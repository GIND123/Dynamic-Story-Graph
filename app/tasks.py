import logging

import redis as redis_module
from neo4j.exceptions import ServiceUnavailable

from app import graph, judge as judge_module, retrieval, vectors
from app.celery_app import celery
from app.config import settings
from app.llm import get_llm
from app.retrieval import format_canon

logger = logging.getLogger(__name__)

# Module-level Redis client for the idempotency lock.
# redis.from_url is lazy (no connection until first use); safe to import without Redis running.
_redis = redis_module.from_url(settings.redis_url, decode_responses=True)

# Lua script: delete the key only if its current value equals our task id.
# Atomic on the Redis server — eliminates the get-compare-delete race.
_LOCK_RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def _release_lock(keys: list[str], args: list[str]) -> None:
    """Atomically delete the lock only if we still own it.

    Resolves the script against the module-level _redis at call time (rather
    than binding at import) so the client is whatever _redis currently is —
    tests monkeypatch app.tasks._redis. Atomic compare-and-delete via Lua.
    """
    _redis.register_script(_LOCK_RELEASE_LUA)(keys=keys, args=args)


@celery.task(
    bind=True,
    autoretry_for=(
        ServiceUnavailable,
    ),  # infra retries only; LLM-quality failures handled inside
    retry_backoff=True,
    max_retries=3,
)
def generate_chapter_task(
    self,
    manuscript_id: str,
    chapter_number: int,
    scene_hint: str | None,
) -> dict:
    """Main generation pipeline with two-tier validation gate.

    Idempotency protocol (CLAUDE.md §3.5):
      1. Redis SET NX acquires the per-chapter lock.
      2. COMMITTED check short-circuits duplicate Celery deliveries.
      3. try/finally releases the lock using get-compare-delete.
    """
    lock_key = f"lock:{manuscript_id}:chapter:{chapter_number}"
    acquired = _redis.set(lock_key, self.request.id, nx=True, ex=900)
    if not acquired:
        logger.info("Lock held by another worker — duplicate suppressed: %s", lock_key)
        return {"status": "duplicate_suppressed"}

    try:
        # Idempotency: already committed by a previous run
        if graph.is_chapter_committed(manuscript_id, chapter_number):
            logger.info("Chapter %d already committed — skipping", chapter_number)
            return {"status": "already_committed", "chapter_number": chapter_number}

        # Context assembly
        canon = graph.get_canon(manuscript_id)
        passages = vectors.similar_passages(
            manuscript_id, scene_hint or format_canon(canon)[:200], k=3
        )
        context = retrieval.assemble_context(canon, passages)

        llm = get_llm()
        feedback: list[str] | None = None
        last_draft: str | None = None

        for attempt in range(1, 4):
            feedback_block = (
                "\n\nPREVIOUS ATTEMPT FEEDBACK (fix these issues):\n"
                + "\n".join(f"- {f}" for f in feedback)
                if feedback
                else ""
            )

            draft = llm.draft_chapter(context + feedback_block, scene_hint)
            last_draft = draft
            extraction = llm.extract(draft)

            # Tier 1: deterministic checks against canon (free, no LLM call)
            t1_violations = graph.tier1_check(extraction, canon)
            if t1_violations:
                feedback = t1_violations
                logger.warning(
                    "Tier-1 violations on attempt %d: %s", attempt, t1_violations
                )
                continue

            # Tier 2: LLM judge (uses tokens; skipped when Tier 1 fails)
            verdict = judge_module.evaluate(canon, draft, llm)
            if verdict.verdict == "FAIL":
                feedback = [
                    f"{c.violated_fact} | {c.draft_evidence}"
                    for c in verdict.contradictions
                ] or [verdict.reasoning]
                logger.warning("Tier-2 FAIL on attempt %d: %s", attempt, feedback)
                continue

            # PASS: commit prose, graph updates, and embedding in one transaction
            emb = vectors.embed_one(draft)
            graph.commit_chapter(manuscript_id, chapter_number, draft, extraction, emb)
            logger.info("Chapter %d committed on attempt %d", chapter_number, attempt)
            return {
                "status": "committed",
                "chapter_number": chapter_number,
                "attempts": attempt,
                "coherence_score": verdict.coherence_score,
            }

        # All 3 attempts failed — store the last draft for human review
        graph.mark_needs_review(
            manuscript_id, chapter_number, last_draft or "", feedback or []
        )
        return {
            "status": "needs_review",
            "chapter_number": chapter_number,
            "attempts": 3,
        }

    finally:
        # Atomic compare-and-delete via Lua (see _LOCK_RELEASE_LUA).
        _release_lock(keys=[lock_key], args=[self.request.id])
