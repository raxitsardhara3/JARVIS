"""Central task execution engine for JARVIS workflows.

The engine is intentionally independent from the existing action modules.  Action
handlers are injected by the caller, which lets current single-action tools keep
working while future planners can compose many ``TaskStep`` objects into a
reliable multi-step workflow.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Iterable


class TaskStatus(str, Enum):
    """Lifecycle states shared by tasks and task steps."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TaskStep:
    """A single executable action inside a task workflow."""

    action_name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    timeout: float | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(slots=True)
class Task:
    """A user or AI requested workflow made of one or more ordered steps."""

    title: str
    description: str = ""
    steps: list[TaskStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING


@dataclass(slots=True)
class TaskResult:
    """Final execution report for a task."""

    task_id: str
    status: TaskStatus
    step_results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time: float = 0.0


@dataclass(slots=True)
class ExecutionContext:
    """Shared state that allows later steps to read earlier outputs."""

    task: Task
    data: dict[str, Any] = field(default_factory=dict)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    def store_step_output(self, step: TaskStep, output: Any) -> None:
        self.step_outputs[step.id] = output
        self.data[step.action_name] = output


@dataclass(slots=True)
class ProgressSnapshot:
    """Point-in-time task progress data suitable for UI updates."""

    task_id: str
    current_step: str | None
    completed_steps: int
    remaining_steps: int
    percentage: float
    execution_time: float


class TaskLogger:
    """Structured task logger wrapper used by the execution engine."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("jarvis.task_engine")

    def event(self, message: str, **fields: Any) -> None:
        suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
        self._logger.info("%s%s", message, f" {suffix}" if suffix else "")

    def exception(self, message: str, **fields: Any) -> None:
        suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
        self._logger.exception("%s%s", message, f" {suffix}" if suffix else "")


class ProgressTracker:
    """Calculates and publishes task progress."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}

    def start(self, task: Task) -> None:
        self._starts[task.id] = time.monotonic()

    def snapshot(self, task: Task, current_step: TaskStep | None = None) -> ProgressSnapshot:
        completed = sum(1 for step in task.steps if step.status == TaskStatus.COMPLETED)
        total = len(task.steps) or 1
        return ProgressSnapshot(
            task_id=task.id,
            current_step=current_step.id if current_step else None,
            completed_steps=completed,
            remaining_steps=max(len(task.steps) - completed, 0),
            percentage=round((completed / total) * 100, 2),
            execution_time=time.monotonic() - self._starts.get(task.id, time.monotonic()),
        )


class RetryHandler:
    """Determines whether failed steps should be retried."""

    def __init__(self, default_retries: int = 0, delay_seconds: float = 0.0) -> None:
        self.default_retries = default_retries
        self.delay_seconds = delay_seconds

    def attempts_for(self, step: TaskStep) -> int:
        return max(step.retry_count, self.default_retries) + 1

    async def wait_before_retry(self) -> None:
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)


class ErrorHandler:
    """Normalizes exceptions and marks failed execution entities."""

    def fail_step(self, step: TaskStep, error: BaseException) -> None:
        step.status = TaskStatus.FAILED
        step.error = str(error)
        step.completed_at = datetime.utcnow()

    def fail_task(self, task: Task) -> None:
        task.status = TaskStatus.FAILED


ActionHandler = Callable[[TaskStep, ExecutionContext], Any | Awaitable[Any]]
EventHook = Callable[[str, Task, TaskStep | None, ProgressSnapshot | None], None]


class TaskExecutor:
    """Executes task steps sequentially through injected action handlers."""

    def __init__(self, handlers: dict[str, ActionHandler], retry_handler: RetryHandler | None = None,
                 error_handler: ErrorHandler | None = None, logger: TaskLogger | None = None,
                 progress: ProgressTracker | None = None, hooks: Iterable[EventHook] | None = None) -> None:
        self.handlers = handlers
        self.retry_handler = retry_handler or RetryHandler()
        self.error_handler = error_handler or ErrorHandler()
        self.logger = logger or TaskLogger()
        self.progress = progress or ProgressTracker()
        self.hooks = list(hooks or [])

    async def execute(self, task: Task, context: ExecutionContext | None = None) -> TaskResult:
        context = context or ExecutionContext(task=task)
        started = datetime.utcnow()
        self.progress.start(task)
        task.status = TaskStatus.RUNNING
        self._emit("task_started", task, None)
        self.logger.event("Task started", task_id=task.id, title=task.title)

        try:
            for step in task.steps:
                if context.cancelled:
                    step.status = TaskStatus.CANCELLED
                    task.status = TaskStatus.CANCELLED
                    break
                await self._execute_step(step, context)
            if task.status not in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                task.status = TaskStatus.COMPLETED
        except Exception as exc:
            self.error_handler.fail_task(task)
            self.logger.exception("Task failed", task_id=task.id, error=str(exc))
        completed = datetime.utcnow()
        result = TaskResult(task.id, task.status, dict(context.step_outputs), started_at=started,
                            completed_at=completed, execution_time=(completed - started).total_seconds())
        self._emit("task_finished", task, None)
        self.logger.event("Task finished", task_id=task.id, status=task.status.value)
        return result

    async def _execute_step(self, step: TaskStep, context: ExecutionContext) -> None:
        handler = self.handlers.get(step.action_name)
        if handler is None:
            raise ValueError(f"No action handler registered for {step.action_name!r}")
        for attempt in range(1, self.retry_handler.attempts_for(step) + 1):
            try:
                step.status = TaskStatus.RUNNING
                step.started_at = datetime.utcnow()
                self._emit("step_started", context.task, step)
                output = handler(step, context)
                if inspect.isawaitable(output):
                    if step.timeout is not None:
                        output = await asyncio.wait_for(output, timeout=step.timeout)
                    else:
                        output = await output
                step.result = output
                step.status = TaskStatus.COMPLETED
                step.completed_at = datetime.utcnow()
                context.store_step_output(step, output)
                self._emit("step_completed", context.task, step)
                return
            except Exception as exc:
                self.error_handler.fail_step(step, exc)
                self.logger.exception("Step failed", step_id=step.id, action=step.action_name, attempt=attempt)
                if attempt >= self.retry_handler.attempts_for(step):
                    context.task.status = TaskStatus.FAILED
                    raise
                self._emit("step_retry", context.task, step)
                await self.retry_handler.wait_before_retry()

    def _emit(self, event: str, task: Task, step: TaskStep | None) -> None:
        snapshot = self.progress.snapshot(task, step)
        for hook in self.hooks:
            hook(event, task, step, snapshot)


class TaskManager:
    """Registers, queues, runs, and cancels JARVIS tasks."""

    def __init__(self, executor: TaskExecutor) -> None:
        self.executor = executor
        self.tasks: dict[str, Task] = {}
        self.queue: Deque[str] = deque()
        self.contexts: dict[str, ExecutionContext] = {}

    def register(self, task: Task) -> Task:
        self.tasks[task.id] = task
        self.contexts[task.id] = ExecutionContext(task=task)
        self.executor.logger.event("Task created", task_id=task.id, title=task.title)
        return task

    def enqueue(self, task: Task) -> Task:
        self.register(task)
        self.queue.append(task.id)
        return task

    async def execute(self, task: Task) -> TaskResult:
        if task.id not in self.tasks:
            self.register(task)
        return await self.executor.execute(task, self.contexts[task.id])

    async def run_next(self) -> TaskResult | None:
        if not self.queue:
            return None
        return await self.execute(self.tasks[self.queue.popleft()])

    def cancel(self, task_id: str) -> bool:
        context = self.contexts.get(task_id)
        task = self.tasks.get(task_id)
        if context is None or task is None:
            return False
        context.cancelled = True
        task.status = TaskStatus.CANCELLED
        self.executor.logger.event("Task cancelled", task_id=task_id)
        return True
