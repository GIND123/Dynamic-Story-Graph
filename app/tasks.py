import logging

import redis as redis_module
from neo4j.exceptions import ServiceUnavailable

from app import graph, retrieval, vectors
from app.celery_app import celery
from app.config import settings
from app.llm import get_llm
from app.retrieval import format_canon

logger = logging.getLogger(__name__)

# Module-level Redis client for the idempotency lock.
# redis.from_url is lazy (no connection until first use); safe to import without Redis running.
_redis = redis_module.from_url(settings.redis_url, decode_responses=True)


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
    """Main generation pipeline.

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

        for attempt in range(1, 4):
            feedback_block = (
                "\n\nPREVIOUS ATTEMPT FEEDBACK (fix these issues):\n"
                + "\n".join(f"- {f}" for f in feedback)
                if feedback
                else ""
            )

            draft = llm.draft_chapter(context + feedback_block, scene_hint)
            extraction = llm.extract(draft)

            # Gate is added in Phase 3; for now every extraction is committed directly
            emb = vectors.embed_one(draft)
            graph.commit_chapter(manuscript_id, chapter_number, draft, extraction, emb)
            logger.info("Chapter %d committed on attempt %d", chapter_number, attempt)
            return {
                "status": "committed",
                "chapter_number": chapter_number,
                "attempts": attempt,
            }

        # Unreachable in Phase 2 (no gate means commit always happens on attempt 1),
        # but required for Phase 3 when the gate may exhaust all retries.
        graph.mark_needs_review(manuscript_id, chapter_number, feedback or [])
        return {
            "status": "needs_review",
            "chapter_number": chapter_number,
            "attempts": 3,
        }

    finally:
        # Get-compare-delete release. Not atomic without Lua; the narrow race:
        # if this task's lock expires and another worker acquires it between GET
        # and DELETE, we would wrongly delete the new lock. The 900 s TTL makes
        # this extremely unlikely in practice.
        current = _redis.get(lock_key)
        if current == self.request.id:
            _redis.delete(lock_key)
