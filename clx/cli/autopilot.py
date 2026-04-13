import logging
import signal
import time

import click

from clx import init_django

logger = logging.getLogger("clx.autopilot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [autopilot] %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)

TASK_PROMPT_TEMPLATE = """\
[New Task: {prompt_name}]

{prompt_content}

[Once you have fully completed your task, call the CompleteTask tool.]
"""


def _resolve_prompt(task):
    """Get prompt content for a task, checking project customizations first."""
    from clx.app.models import Prompt
    from clx.app.prompts import prompt_registry

    try:
        prompt = Prompt.objects.get(
            project=task.project, prompt_id=task.prompt_id
        )
        return prompt.name, prompt.content
    except Prompt.DoesNotExist:
        pass

    entry = prompt_registry.get(task.prompt_id)
    if entry:
        return entry["name"], entry["content"]

    return task.prompt_id, f"(Unknown prompt: {task.prompt_id})"


def _get_thread_for_task(task):
    """Get (or lazily create) the autopilot thread for a task."""
    from clx.app.models import Thread

    if task.label:
        label = task.label
    else:
        label = task.project.active_label or task.project.labels.first()

    if not label:
        return None

    if not label.autopilot_thread_id:
        thread = Thread.objects.create(label=label)
        label.autopilot_thread = thread
        label.save(update_fields=["autopilot_thread", "updated_at"])

    return label.autopilot_thread


def _process_task(task, resume=False):
    """Run a single task through the autopilot agent."""
    from clx.app.agent import CLXAgent
    from clx.app.tools import CompleteTask

    thread = _get_thread_for_task(task)
    if not thread:
        logger.warning(f"No thread available for task {task.id}, skipping")
        return

    # Lock the thread and mark task in-progress.
    thread.autopilot_locked = True
    thread.save(update_fields=["autopilot_locked", "updated_at"])
    task.status = task.Status.IN_PROGRESS
    task.save(update_fields=["status", "updated_at"])

    try:
        # Build the agent with CompleteTask injected.
        tools = CLXAgent.default_tools + [CompleteTask]
        agent = CLXAgent(thread, tools=tools)

        # Compact before starting/resuming the task if needed.
        agent.compact_if_needed()

        # For resume, the user's message is already in the thread history
        # (saved by send_message_api). Pass None so the agent just responds.
        if resume:
            message = None
        else:
            prompt_name, prompt_content = _resolve_prompt(task)
            message = TASK_PROMPT_TEMPLATE.format(
                prompt_name=prompt_name,
                prompt_content=prompt_content,
            )

        logger.info(f"Running task: {task.prompt_id} (label={task.label})")
        result = agent.autopilot_run(message)

        if result == "completed":
            logger.info(f"Task completed: {task.prompt_id}")
            task.delete()
            task.project.update_tasks()
        else:
            logger.info(f"Task awaiting input: {task.prompt_id}")
            task.status = task.Status.AWAITING_INPUT
            task.save(update_fields=["status", "updated_at"])
    finally:
        thread.autopilot_locked = False
        thread.save(update_fields=["autopilot_locked", "updated_at"])


# Track last-processed label per project for round-robin scheduling.
_last_label = {}  # project_id -> label_id


def _process_cycle():
    """Run one autopilot cycle across all enabled projects."""
    from django.db import close_old_connections

    from clx.app.models import Project, Task

    close_old_connections()
    projects = Project.objects.filter(autopilot_enabled=True)
    for project in projects:
        project.update_tasks()

        # Check for awaiting_input tasks where the user has replied.
        # The web UI saves user messages directly without running the agent,
        # so we just check for new messages since the task paused.
        for task in project.tasks.filter(status=Task.Status.AWAITING_INPUT):
            thread = _get_thread_for_task(task)
            if not thread:
                continue
            # Check for any messages after the task entered awaiting_input.
            new_msgs = thread.messages.filter(
                created_at__gt=task.updated_at
            ).exists()
            if new_msgs:
                _process_task(task, resume=True)
                _last_label[str(project.id)] = str(task.label_id)
                project.refresh_from_db(fields=["autopilot_enabled"])
                if not project.autopilot_enabled:
                    break

        # Refresh to check if autopilot is still enabled.
        project.refresh_from_db(fields=["autopilot_enabled"])
        if not project.autopilot_enabled:
            continue

        # Pick the next pending task, but only if its label's autopilot
        # thread doesn't already have an active task.
        # Round-robin: prefer labels we haven't worked on recently.
        active_labels = set(
            project.tasks.filter(
                status__in=[
                    Task.Status.IN_PROGRESS,
                    Task.Status.AWAITING_INPUT,
                ]
            ).values_list("label_id", flat=True)
        )
        candidates = list(
            project.tasks.filter(status=Task.Status.PENDING)
            .exclude(label_id__in=active_labels)
            .order_by("created_at")
        )
        if not candidates:
            continue

        # Pick the first candidate whose label differs from last-processed.
        last = _last_label.get(str(project.id))
        task = None
        for c in candidates:
            if str(c.label_id) != last:
                task = c
                break
        if task is None:
            task = candidates[0]

        updated = Task.objects.filter(
            id=task.id, status=Task.Status.PENDING
        ).update(status=Task.Status.IN_PROGRESS)
        if updated:
            task.refresh_from_db()
            _last_label[str(project.id)] = str(task.label_id)
            _process_task(task)


def _cleanup():
    """Clear stale locks and reset in-progress tasks."""
    from django.db import close_old_connections

    from clx.app.models import Task, Thread

    close_old_connections()
    Thread.objects.filter(autopilot_locked=True).update(autopilot_locked=False)
    Task.objects.filter(status=Task.Status.IN_PROGRESS).update(
        status=Task.Status.PENDING
    )


@click.command()
@click.option("--interval", default=10, help="Seconds between polling cycles")
def autopilot(interval):
    """Run the autopilot loop, processing tasks for enabled projects."""
    init_django()

    _cleanup()
    logger.info(f"Autopilot started. Polling every {interval}s...")

    def handle_shutdown(signum, frame):
        logger.info("Shutting down, cleaning up locks...")
        _cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while True:
        try:
            _process_cycle()
        except SystemExit:
            raise
        except Exception:
            logger.exception("Error in autopilot cycle")
        time.sleep(interval)
