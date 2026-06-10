from app.celery_app import celery


@celery.task(bind=True)
def generate_chapter_task(
    self,
    manuscript_id: str,
    chapter_number: int,
    scene_hint: str | None,
) -> dict:
    """Phase 1 echo task — replaced with the full pipeline in Phase 2."""
    return {
        "status": "pending_pipeline",
        "chapter_number": chapter_number,
        "manuscript_id": manuscript_id,
    }
