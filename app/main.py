import asyncio
import logging
from contextlib import asynccontextmanager

import redis as redis_module
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app import graph, tasks
from app.celery_app import celery
from app.config import settings
from app.models import ChapterRequest, ManuscriptCreate

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure Neo4j constraints are in place before serving traffic.

    Retries up to 15 times (45 s) because the worker may restart after Neo4j
    has already been healthy once; the Docker healthcheck alone is not enough
    for that case (CLAUDE.md §13.2).
    """
    max_attempts = 15
    for attempt in range(1, max_attempts + 1):
        try:
            await asyncio.to_thread(graph.ensure_constraints)
            logger.info("Neo4j ready — constraints ensured")
            break
        except Exception as exc:
            if attempt == max_attempts:
                logger.error(
                    "Could not reach Neo4j after %d attempts; proceeding anyway: %s",
                    max_attempts,
                    exc,
                )
            else:
                logger.warning(
                    "Neo4j not ready (attempt %d/%d): %s", attempt, max_attempts, exc
                )
                await asyncio.sleep(3)
    yield


app = FastAPI(title="Manuscript Memory Engine", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _check_neo4j() -> str:
    try:
        graph.init_driver().verify_connectivity()
        return "ok"
    except Exception:
        return "error"


def _check_redis() -> str:
    try:
        redis_module.from_url(settings.redis_url).ping()
        return "ok"
    except Exception:
        return "error"


@app.get("/health")
def health() -> dict:
    components = {"neo4j": _check_neo4j(), "redis": _check_redis()}
    overall = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return {"status": overall, "components": components}


# ---------------------------------------------------------------------------
# Manuscripts
# ---------------------------------------------------------------------------


@app.post("/manuscripts")
def create_manuscript(body: ManuscriptCreate) -> dict:
    manuscript_id = graph.create_manuscript(body)
    return {"manuscript_id": manuscript_id}


@app.get("/manuscripts/{mid}/state")
def get_state(mid: str) -> dict:
    return graph.get_state(mid)


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------


@app.post("/manuscripts/{mid}/chapters")
def request_chapter(mid: str, body: ChapterRequest) -> JSONResponse:
    chapter_number = graph.next_chapter_number(mid)
    task = tasks.generate_chapter_task.delay(mid, chapter_number, body.scene_hint)
    return JSONResponse(
        content={"job_id": task.id, "chapter_number": chapter_number},
        status_code=202,
    )


@app.get("/manuscripts/{mid}/chapters/{n}")
def get_chapter(mid: str, n: int) -> dict:
    chapter = graph.get_chapter(mid, n)
    if chapter is None:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return chapter


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    result = AsyncResult(job_id, app=celery)
    return {
        "job_id": job_id,
        "status": result.state,
        "result": result.result if result.ready() else None,
    }
