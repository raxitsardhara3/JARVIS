"""Universal Action Registry for JARVIS.

The registry gives planners and tools one place to discover what JARVIS can do
without knowing which Python module owns an action.  Existing actions can be
loaded from Gemini tool declarations for backward compatibility, while new
plugins can use the ``@register_action`` decorator to self-register metadata at
import time.
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import platform
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Callable, Iterable


class ActionCategory(str, Enum):
    """High-level capability groups used by planners and documentation."""

    APPLICATION = "application"
    AUTOMATION = "automation"
    BROWSER = "browser"
    CODING = "coding"
    COMMUNICATION = "communication"
    DESKTOP = "desktop"
    DEVELOPER_TOOLS = "developer_tools"
    FILES = "files"
    GAMES = "games"
    MEMORY = "memory"
    MONITORING = "monitoring"
    SYSTEM = "system"
    VISION = "vision"
    WEATHER = "weather"
    WEB = "web"
    PLUGIN = "plugin"
    GENERAL = "general"


@dataclass(slots=True)
class ActionParameter:
    """Schema details for one action parameter."""

    name: str
    type: str = "ANY"
    description: str = ""
    required: bool = False
    default: Any = None


@dataclass(slots=True)
class ActionMetadata:
    """Describes an executable JARVIS action for discovery and planning.

    Fields intentionally mirror what a future planner needs: identity, human and
    model-facing descriptions, safety information, platform fit, schema, examples,
    versioning, and developer-only notes.
    """

    name: str
    description: str
    category: ActionCategory | str = ActionCategory.GENERAL
    parameters: dict[str, ActionParameter] = field(default_factory=dict)
    required_permissions: list[str] = field(default_factory=list)
    platform_support: list[str] = field(default_factory=lambda: ["Windows", "Darwin", "Linux"])
    return_type: str = "str"
    examples: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    developer_notes: str = ""
    module: str | None = None
    callable_name: str | None = None

    def validate_parameters(self, provided: dict[str, Any]) -> list[str]:
        """Return validation errors for missing required parameters."""
        errors: list[str] = []
        for param in self.parameters.values():
            if param.required and param.name not in provided:
                errors.append(f"Missing required parameter: {param.name}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata for dashboards, docs, or API responses."""
        category = self.category.value if isinstance(self.category, ActionCategory) else str(self.category)
        return {
            "name": self.name,
            "description": self.description,
            "category": category,
            "parameters": {
                key: {
                    "type": value.type,
                    "description": value.description,
                    "required": value.required,
                    "default": value.default,
                }
                for key, value in self.parameters.items()
            },
            "required_permissions": list(self.required_permissions),
            "platform_support": list(self.platform_support),
            "return_type": self.return_type,
            "examples": list(self.examples),
            "tags": list(self.tags),
            "version": self.version,
            "developer_notes": self.developer_notes,
            "module": self.module,
            "callable_name": self.callable_name,
        }


@dataclass(slots=True)
class RegisteredAction:
    """Registry entry containing metadata and an optional callable."""

    metadata: ActionMetadata
    handler: Callable[..., Any] | None = None


_PENDING_ACTIONS: list[RegisteredAction] = []


def register_action(metadata: ActionMetadata) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for future plugins and action modules to self-register.

    The decorated function is added to a pending global list during import; any
    ``UniversalActionRegistry`` can later consume that list during discovery.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        metadata.module = metadata.module or func.__module__
        metadata.callable_name = metadata.callable_name or func.__name__
        _PENDING_ACTIONS.append(RegisteredAction(metadata=metadata, handler=func))
        return func

    return decorator


class PermissionChecker:
    """Validates platform and permission requirements before execution."""

    def __init__(self, granted_permissions: Iterable[str] | None = None, current_platform: str | None = None) -> None:
        default_permissions = {
            "communication",
            "desktop_control",
            "filesystem",
            "network",
            "vision_capture",
        }
        self.granted_permissions = set(granted_permissions) if granted_permissions is not None else default_permissions
        self.current_platform = current_platform or platform.system()

    def check(self, metadata: ActionMetadata) -> list[str]:
        """Return unmet permission/platform requirements for an action."""
        errors: list[str] = []
        if metadata.platform_support and self.current_platform not in metadata.platform_support:
            errors.append(f"Unsupported platform: {self.current_platform}")
        missing = [permission for permission in metadata.required_permissions if permission not in self.granted_permissions]
        if missing:
            errors.append("Missing permissions: " + ", ".join(missing))
        return errors


class ActionDocumentationGenerator:
    """Produces human-readable or JSON documentation from registry metadata."""

    def markdown(self, actions: Iterable[RegisteredAction]) -> str:
        sections = ["# JARVIS Action Registry", ""]
        for entry in sorted(actions, key=lambda item: item.metadata.name):
            meta = entry.metadata
            category = meta.category.value if isinstance(meta.category, ActionCategory) else str(meta.category)
            sections.extend([
                f"## {meta.name}",
                f"- Category: `{category}`",
                f"- Description: {meta.description}",
                f"- Return type: `{meta.return_type}`",
                f"- Platforms: {', '.join(meta.platform_support) or 'any'}",
                f"- Tags: {', '.join(meta.tags) or 'none'}",
                "",
            ])
        return "\n".join(sections).strip() + "\n"

    def json(self, actions: Iterable[RegisteredAction]) -> str:
        return json.dumps([entry.metadata.to_dict() for entry in actions], indent=2, sort_keys=True)


class UniversalActionRegistry:
    """Central catalog for action metadata, lookup, validation, and discovery."""

    def __init__(self, permission_checker: PermissionChecker | None = None) -> None:
        self._actions: dict[str, RegisteredAction] = {}
        self.permission_checker = permission_checker or PermissionChecker()
        self.documentation = ActionDocumentationGenerator()

    def register(self, metadata: ActionMetadata, handler: Callable[..., Any] | None = None, *, replace: bool = True) -> None:
        """Register an action and optional executable handler."""
        if not metadata.name:
            raise ValueError("Action metadata requires a name")
        if not metadata.description:
            raise ValueError(f"Action {metadata.name!r} requires a description")
        if metadata.name in self._actions and not replace:
            raise ValueError(f"Action already registered: {metadata.name}")
        self._actions[metadata.name] = RegisteredAction(metadata=metadata, handler=handler)

    def consume_pending(self) -> None:
        """Register actions captured by the decorator during module imports."""
        while _PENDING_ACTIONS:
            entry = _PENDING_ACTIONS.pop(0)
            self.register(entry.metadata, entry.handler)

    def discover(self, package_name: str = "actions") -> None:
        """Import action modules so decorator-based and inferred actions appear."""
        package = importlib.import_module(package_name)
        for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            try:
                module = importlib.import_module(module_info.name)
            except Exception:
                # Optional dependencies should not prevent the registry from loading.
                continue
            self._register_inferred_module_action(module)
        self.consume_pending()

    def load_from_tool_declarations(self, declarations: Iterable[dict[str, Any]]) -> None:
        """Seed registry metadata from existing Gemini tool declarations."""
        for declaration in declarations:
            name = str(declaration.get("name", "")).strip()
            if not name:
                continue
            metadata = ActionMetadata(
                name=name,
                description=str(declaration.get("description", "")),
                category=_infer_category(name),
                parameters=_parameters_from_schema(declaration.get("parameters", {})),
                required_permissions=_infer_permissions(name),
                platform_support=_infer_platforms(name),
                return_type="str",
                examples=[],
                tags=_infer_tags(name, declaration.get("description", "")),
                version="1.0.0",
                developer_notes="Loaded from Gemini Live tool declaration for backward compatibility.",
            )
            self.register(metadata, replace=True)

    def get(self, name: str) -> RegisteredAction | None:
        return self._actions.get(name)

    def require(self, name: str) -> RegisteredAction:
        action = self.get(name)
        if action is None:
            raise KeyError(f"Unknown action: {name}")
        return action

    def list(self, category: ActionCategory | str | None = None) -> list[RegisteredAction]:
        actions = list(self._actions.values())
        if category is None:
            return sorted(actions, key=lambda item: item.metadata.name)
        category_value = category.value if isinstance(category, ActionCategory) else str(category)
        return sorted(
            [entry for entry in actions if _category_value(entry.metadata.category) == category_value],
            key=lambda item: item.metadata.name,
        )

    def search(self, query: str = "", *, category: ActionCategory | str | None = None,
               tags: Iterable[str] | None = None) -> list[RegisteredAction]:
        """Find actions by text, category, and tags."""
        query_lower = query.lower().strip()
        tag_set = {tag.lower() for tag in tags or []}
        results: list[RegisteredAction] = []
        for entry in self.list(category=category):
            meta = entry.metadata
            haystack = " ".join([meta.name, meta.description, " ".join(meta.tags)]).lower()
            if query_lower and query_lower not in haystack:
                continue
            if tag_set and not tag_set.issubset({tag.lower() for tag in meta.tags}):
                continue
            results.append(entry)
        return results

    def validate(self, name: str, parameters: dict[str, Any]) -> list[str]:
        """Validate action existence, parameters, permissions, and platform."""
        action = self.get(name)
        if action is None:
            return [f"Unknown action: {name}"]
        return action.metadata.validate_parameters(parameters) + self.permission_checker.check(action.metadata)

    def docs_markdown(self) -> str:
        return self.documentation.markdown(self.list())

    def docs_json(self) -> str:
        return self.documentation.json(self.list())

    def _register_inferred_module_action(self, module: ModuleType) -> None:
        base_name = module.__name__.rsplit(".", 1)[-1]
        handler = getattr(module, base_name, None)
        if not callable(handler) or base_name in self._actions:
            return
        doc = inspect.getdoc(handler) or inspect.getdoc(module) or f"Action inferred from {module.__name__}."
        metadata = ActionMetadata(
            name=base_name,
            description=doc.splitlines()[0],
            category=_infer_category(base_name),
            required_permissions=_infer_permissions(base_name),
            platform_support=_infer_platforms(base_name),
            tags=_infer_tags(base_name, doc),
            module=module.__name__,
            callable_name=getattr(handler, "__name__", base_name),
            developer_notes="Inferred during module discovery. Add @register_action for richer metadata.",
        )
        self.register(metadata, handler=handler, replace=False)


_DEFAULT_PLATFORM_SUPPORT = ["Windows", "Darwin", "Linux"]


def _parameters_from_schema(schema: dict[str, Any]) -> dict[str, ActionParameter]:
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", []) if isinstance(schema, dict) else [])
    parameters: dict[str, ActionParameter] = {}
    for name, details in properties.items():
        details = details if isinstance(details, dict) else {}
        parameters[name] = ActionParameter(
            name=name,
            type=str(details.get("type", "ANY")),
            description=str(details.get("description", "")),
            required=name in required,
            default=details.get("default"),
        )
    return parameters


def _infer_category(name: str) -> ActionCategory:
    lower = name.lower()
    if "browser" in lower:
        return ActionCategory.BROWSER
    if "computer_control" in lower or "computer_settings" in lower:
        return ActionCategory.AUTOMATION
    if "desktop" in lower or "open_app" in lower:
        return ActionCategory.DESKTOP
    if "file" in lower:
        return ActionCategory.FILES
    if "code" in lower or "dev_agent" in lower:
        return ActionCategory.CODING
    if "message" in lower:
        return ActionCategory.COMMUNICATION
    if "screen" in lower or "camera" in lower:
        return ActionCategory.VISION
    if "weather" in lower:
        return ActionCategory.WEATHER
    if "web_search" in lower or "youtube" in lower or "flight" in lower:
        return ActionCategory.WEB
    if "memory" in lower:
        return ActionCategory.MEMORY
    if "monitor" in lower or "system_status" in lower:
        return ActionCategory.MONITORING
    if "game" in lower:
        return ActionCategory.GAMES
    if "shutdown" in lower:
        return ActionCategory.SYSTEM
    return ActionCategory.GENERAL


def _infer_permissions(name: str) -> list[str]:
    lower = name.lower()
    permissions: list[str] = []
    if any(token in lower for token in ("computer", "desktop", "open_app", "browser", "game")):
        permissions.append("desktop_control")
    if any(token in lower for token in ("screen", "camera")):
        permissions.append("vision_capture")
    if any(token in lower for token in ("file", "code", "dev_agent")):
        permissions.append("filesystem")
    if any(token in lower for token in ("web", "weather", "youtube", "flight")):
        permissions.append("network")
    if "message" in lower:
        permissions.append("communication")
    return permissions


def _infer_platforms(name: str) -> list[str]:
    if name.lower() == "shutdown_jarvis":
        return _DEFAULT_PLATFORM_SUPPORT.copy()
    return _DEFAULT_PLATFORM_SUPPORT.copy()


def _infer_tags(name: str, description: str) -> list[str]:
    words = {part for part in name.lower().replace("-", "_").split("_") if part}
    for keyword in ("browser", "desktop", "file", "web", "vision", "memory", "system", "automation", "plugin"):
        if keyword in description.lower():
            words.add(keyword)
    return sorted(words)


def _category_value(category: ActionCategory | str) -> str:
    return category.value if isinstance(category, ActionCategory) else str(category)
