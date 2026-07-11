from celery import Celery
from celery.signals import worker_process_init

from app.config import settings

celery = Celery(
    "mme",
    broker=settings.redis_url,
    backend=settings.result_backend,
    include=["app.tasks"],
)

celery.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,  # pairs with acks_late: crashed worker => requeue
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=1,  # long tasks: do not hoard
    task_time_limit=600,
    task_soft_time_limit=540,
)


@worker_process_init.connect
def _warmup_embedder(**kwargs: object) -> None:
    """Download and compile the fastembed model before the first task runs.

    Fires in each worker child process. The deferred import keeps fastembed
    out of the API process, which also imports celery_app.
    """
    from app.vectors import warmup

    warmup()
