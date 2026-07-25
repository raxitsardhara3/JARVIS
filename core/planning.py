"""Intent parsing and planning primitives for JARVIS.

This module adds a planning layer between natural-language user requests and the
existing task execution engine.  It does not execute actions.  Instead it uses the
Action Registry for capability discovery/validation and emits serializable plans
that can be converted into ``core.task_engine.Task`` objects by callers.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from core.action_registry import UniversalActionRegistry
from core.task_engine import Task, TaskStep


class IntentType(str, Enum):
    """Broad request categories used to route planning behavior."""

    SINGLE_ACTION = "single_action"
    MULTI_STEP_WORKFLOW = "multi_step_workflow"
    INFORMATIONAL = "informational"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class PlanningContext:
    """Context supplied to the planner for safe, registry-aware planning."""

    user_request: str
    available_actions: list[dict[str, Any]] = field(default_factory=list)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    session_metadata: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 12


@dataclass(slots=True)
class ExecutionStep:
    """One planned action with dependencies and execution metadata."""

    id: str
    action_name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    required_permissions: list[str] = field(default_factory=list)
    estimated_duration_seconds: float | None = None
    fallback_actions: list[str] = field(default_factory=list)
    success_conditions: list[str] = field(default_factory=list)
    failure_conditions: list[str] = field(default_factory=list)
    retry_count: int = 0
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_task_step(self) -> TaskStep:
        """Convert this planning step into a Task Engine step without running it."""
        return TaskStep(
            id=self.id,
            action_name=self.action_name,
            description=self.description,
            parameters=dict(self.parameters),
            retry_count=self.retry_count,
            timeout=self.timeout_seconds,
        )


@dataclass(slots=True)
class ExecutionPlan:
    """Serializable plan that the existing Task Engine can later execute."""

    goal: str
    steps: list[ExecutionStep] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    intent_type: IntentType = IntentType.UNKNOWN
    required_actions: list[str] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    estimated_duration_seconds: float | None = None
    permissions: list[str] = field(default_factory=list)
    fallback_actions: list[str] = field(default_factory=list)
    success_conditions: list[str] = field(default_factory=list)
    failure_conditions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        data = asdict(self)
        data["intent_type"] = self.intent_type.value
        return data

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the plan as JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_task(self) -> Task:
        """Convert the plan to a Task Engine task for future execution."""
        return Task(
            id=self.id,
            title=self.goal,
            description=f"Planned workflow for: {self.goal}",
            steps=[step.to_task_step() for step in self.steps],
            metadata={"source": "intent_planner", **self.metadata},
        )


@dataclass(slots=True)
class PlannerResult:
    """Planner API response containing the plan and diagnostics."""

    plan: ExecutionPlan | None
    intent_type: IntentType
    confidence: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.errors


class IntentClassifier:
    """Classifies raw user text before detailed plan construction."""

    _workflow_markers = (" and ", ",", " then ", " after ", " before ", "install", "clone", "create", "generate")
    _action_markers = ("open", "launch", "start", "search", "play", "send", "set", "run", "clone", "create", "generate")

    def classify(self, request: str) -> tuple[IntentType, float]:
        text = request.lower().strip()
        if not text:
            return IntentType.UNKNOWN, 0.0
        if sum(marker in text for marker in self._workflow_markers) >= 2:
            return IntentType.MULTI_STEP_WORKFLOW, 0.78
        if any(marker in text for marker in self._action_markers):
            return IntentType.SINGLE_ACTION, 0.72
        if text.endswith("?") or text.startswith(("what", "who", "when", "where", "why", "how")):
            return IntentType.INFORMATIONAL, 0.66
        return IntentType.UNKNOWN, 0.35


class IntentParser:
    """Turns a natural-language request into normalized planning hints.

    A structured LLM callable can be injected later.  When absent or invalid, the
    parser falls back to deterministic rules so the architecture remains usable
    in tests and offline development.
    """

    def __init__(self, llm_plan_generator: Callable[[PlanningContext], Mapping[str, Any] | str] | None = None) -> None:
        self.llm_plan_generator = llm_plan_generator
        self.classifier = IntentClassifier()

    def parse(self, context: PlanningContext) -> dict[str, Any]:
        if self.llm_plan_generator is not None:
            raw = self.llm_plan_generator(context)
            if isinstance(raw, str):
                raw = json.loads(raw)
            return dict(raw)
        intent_type, confidence = self.classifier.classify(context.user_request)
        return {"goal": context.user_request.strip(), "intent_type": intent_type.value, "confidence": confidence}


class TaskPlanner:
    """Builds ordered execution plans using parsed intent and registry metadata."""

    def __init__(self, registry: UniversalActionRegistry, parser: IntentParser | None = None) -> None:
        self.registry = registry
        self.parser = parser or IntentParser()

    def plan(self, request: str, *, context: PlanningContext | None = None) -> PlannerResult:
        context = context or PlanningContext(user_request=request, available_actions=self._action_docs())
        parsed = self.parser.parse(context)
        intent_type = IntentType(parsed.get("intent_type", IntentType.UNKNOWN))
        steps = self._steps_from_parsed(parsed, context)
        plan = ExecutionPlan(goal=parsed.get("goal") or request, intent_type=intent_type, steps=steps)
        self._finalize(plan)
        result = PlanValidator(self.registry).validate(plan)
        return PlannerResult(plan=plan, intent_type=intent_type, confidence=float(parsed.get("confidence", 0.5)),
                             errors=result.errors, warnings=result.warnings, metadata={"parsed": parsed})

    def _steps_from_parsed(self, parsed: Mapping[str, Any], context: PlanningContext) -> list[ExecutionStep]:
        raw_steps = parsed.get("steps")
        if isinstance(raw_steps, list) and raw_steps:
            return [self._coerce_step(item, i) for i, item in enumerate(raw_steps[:context.max_steps], 1)]
        return PlanBuilder(self.registry).from_request(context.user_request)[:context.max_steps]

    def _coerce_step(self, item: Mapping[str, Any], index: int) -> ExecutionStep:
        return ExecutionStep(
            id=str(item.get("id") or f"step-{index}"),
            action_name=str(item.get("action_name") or item.get("action") or ""),
            description=str(item.get("description") or item.get("goal") or ""),
            parameters=dict(item.get("parameters") or {}),
            dependencies=list(item.get("dependencies") or ([] if index == 1 else [f"step-{index-1}"])),
            fallback_actions=list(item.get("fallback_actions") or []),
            success_conditions=list(item.get("success_conditions") or []),
            failure_conditions=list(item.get("failure_conditions") or []),
            retry_count=int(item.get("retry_count") or 0),
            timeout_seconds=item.get("timeout_seconds"),
            metadata=dict(item.get("metadata") or {}),
        )

    def _finalize(self, plan: ExecutionPlan) -> None:
        for step in plan.steps:
            entry = self.registry.get(step.action_name)
            if entry:
                step.required_permissions = list(entry.metadata.required_permissions)
        plan.required_actions = list(dict.fromkeys(step.action_name for step in plan.steps))
        plan.dependencies = {step.id: list(step.dependencies) for step in plan.steps if step.dependencies}
        plan.permissions = sorted({p for step in plan.steps for p in step.required_permissions})
        durations = [step.estimated_duration_seconds for step in plan.steps if step.estimated_duration_seconds]
        plan.estimated_duration_seconds = sum(durations) if durations else None
        plan.success_conditions = ["All required steps complete successfully"] if plan.steps else []
        plan.failure_conditions = ["A required step fails without a fallback"] if plan.steps else ["No executable action found"]

    def _action_docs(self) -> list[dict[str, Any]]:
        return [entry.metadata.to_dict() for entry in self.registry.list()]


class PlanBuilder:
    """Deterministic plan builder used as fallback and for tests/examples."""

    def __init__(self, registry: UniversalActionRegistry) -> None:
        self.registry = registry

    def from_request(self, request: str) -> list[ExecutionStep]:
        parts = [p.strip(" .") for p in re.split(r"\bthen\b|,|\band\b", request, flags=re.I) if p.strip(" .")]
        steps: list[ExecutionStep] = []
        for index, part in enumerate(parts or [request], 1):
            step = self._step_for_part(part, index)
            if step is not None:
                steps.append(step)
        return steps

    def _step_for_part(self, text: str, index: int) -> ExecutionStep | None:
        lower = text.lower()
        action, params = "web_search", {"query": text, "mode": "search"}
        if lower.startswith(("open ", "launch ", "start ")):
            action, params = "open_app", {"app_name": re.sub(r"^(open|launch|start)\s+", "", text, flags=re.I).strip()}
        elif "youtube" in lower and "search" in lower:
            action, params = "youtube_video", {"action": "play", "query": re.sub(r"search\s+youtube\s+for\s+", "", text, flags=re.I)}
        elif "clone" in lower or "install dependencies" in lower or "run the project" in lower or "react project" in lower or "portfolio website" in lower:
            action, params = "dev_agent", {"description": text}
        desc = f"{text[0].upper() + text[1:] if text else action}"
        return ExecutionStep(id=f"step-{index}", action_name=action, description=desc, parameters=params,
                             dependencies=[] if index == 1 else [f"step-{index-1}"],
                             success_conditions=[f"{action} reports success"], failure_conditions=[f"{action} returns an error"])


class PlanValidator:
    """Validates plans against action availability, parameters, and ordering."""

    def __init__(self, registry: UniversalActionRegistry) -> None:
        self.registry = registry

    def validate(self, plan: ExecutionPlan) -> PlannerResult:
        errors: list[str] = []
        warnings: list[str] = []
        ids = {step.id for step in plan.steps}
        for step in plan.steps:
            errors.extend(f"{step.id}: {error}" for error in self.registry.validate(step.action_name, step.parameters))
            for dep in step.dependencies:
                if dep not in ids:
                    errors.append(f"{step.id}: unknown dependency {dep}")
        if self._has_cycle(plan.steps):
            errors.append("Plan contains a dependency cycle")
        if not plan.steps:
            warnings.append("Plan contains no executable steps")
        return PlannerResult(plan=plan, intent_type=plan.intent_type, confidence=1.0, errors=errors, warnings=warnings)

    def _has_cycle(self, steps: Iterable[ExecutionStep]) -> bool:
        graph = {step.id: step.dependencies for step in steps}
        visiting: set[str] = set(); visited: set[str] = set()
        def visit(node: str) -> bool:
            if node in visiting: return True
            if node in visited: return False
            visiting.add(node)
            if any(dep in graph and visit(dep) for dep in graph.get(node, [])): return True
            visiting.remove(node); visited.add(node); return False
        return any(visit(node) for node in graph)


class PlannerAPI:
    """Facade for application code that needs plans or Task Engine objects."""

    def __init__(self, registry: UniversalActionRegistry, planner: TaskPlanner | None = None) -> None:
        self.planner = planner or TaskPlanner(registry)

    def create_plan(self, request: str, **metadata: Any) -> PlannerResult:
        context = PlanningContext(user_request=request, session_metadata=metadata)
        return self.planner.plan(request, context=context)

    def create_task(self, request: str, **metadata: Any) -> Task:
        result = self.create_plan(request, **metadata)
        if not result.ok or result.plan is None:
            raise ValueError("Cannot create task from invalid plan: " + "; ".join(result.errors))
        return result.plan.to_task()
