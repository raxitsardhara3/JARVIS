"""Action wrapper for the Universal Computer Control Engine."""
from __future__ import annotations

import json
from typing import Any

from core.action_registry import ActionCategory, ActionMetadata, ActionParameter, register_action
from core.computer_control_engine import AutomationContext, ComputerController


@register_action(
    ActionMetadata(
        name="universal_computer",
        description=(
            "Universal OS computer control facade for mouse, keyboard, clipboard, windows, "
            "applications, desktop, screen, monitors, file explorer, and guarded system operations."
        ),
        category=ActionCategory.AUTOMATION,
        parameters={
            "capability": ActionParameter(
                name="capability",
                type="STRING",
                description="mouse | keyboard | clipboard | window | application | desktop | screen | monitor | file_explorer | system",
                required=True,
            ),
            "action": ActionParameter(
                name="action",
                type="STRING",
                description="Capability-specific action, e.g. click, hotkey, paste, focus, launch, screenshot.",
                required=True,
            ),
            "parameters": ActionParameter(
                name="parameters",
                type="OBJECT",
                description="Capability-specific parameters passed to the selected manager.",
                required=False,
                default={},
            ),
            "dry_run": ActionParameter(
                name="dry_run",
                type="BOOLEAN",
                description="Plan/log the operation without touching the OS.",
                required=False,
                default=False,
            ),
        },
        required_permissions=["desktop_control"],
        return_type="dict",
        tags=["computer", "automation", "mouse", "keyboard", "window", "clipboard", "screen"],
    )
)
def universal_computer(parameters: dict[str, Any] | None = None, response=None, player=None, session_memory=None) -> str:
    """Run one Universal Computer Control Engine operation and return JSON."""
    params = parameters or {}
    capability = str(params.get("capability", "")).strip()
    action = str(params.get("action", "")).strip()
    operation_parameters = dict(params.get("parameters") or {})
    dry_run = bool(params.get("dry_run", False))

    if player:
        player.write_log(f"[UniversalComputer] {capability}.{action}")

    controller = ComputerController(AutomationContext(dry_run=dry_run))
    result = controller.execute(capability, action, operation_parameters)
    payload = {
        "ok": result.ok,
        "message": result.message,
        "data": result.data,
        "permission_required": result.permission_required.name,
    }
    return json.dumps(payload, ensure_ascii=False)
