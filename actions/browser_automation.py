"""Action wrapper for Browser Automation Engine 2.0."""
from __future__ import annotations

import json
from typing import Any

from actions.browser_control import browser_control
from core.action_registry import ActionCategory, ActionMetadata, ActionParameter, register_action
from core.browser_automation_engine import AutomationContext, BrowserController


@register_action(
    ActionMetadata(
        name="browser_automation",
        description=(
            "Browser Automation Engine 2.0 facade for sessions, profiles, tabs, navigation, "
            "elements, forms, uploads, downloads, cookies, auth, history, bookmarks, and recovery."
        ),
        category=ActionCategory.BROWSER,
        parameters={
            "capability": ActionParameter(
                name="capability",
                type="STRING",
                description="browser | navigation | tab | window | element | input | dom | cookies | download | upload | auth | history | bookmark | page",
                required=True,
            ),
            "action": ActionParameter(
                name="action",
                type="STRING",
                description="Capability-specific action, e.g. open, search, click, type, wait, screenshot.",
                required=True,
            ),
            "parameters": ActionParameter(
                name="parameters",
                type="OBJECT",
                description="Capability-specific parameters for BrowserController.execute().",
                required=False,
                default={},
            ),
            "dry_run": ActionParameter(
                name="dry_run",
                type="BOOLEAN",
                description="Plan/log the operation without invoking the live browser adapter.",
                required=False,
                default=False,
            ),
        },
        required_permissions=["network", "desktop_control"],
        return_type="dict",
        tags=["browser", "automation", "session", "tabs", "dom", "web"],
    )
)
def browser_automation(parameters: dict[str, Any] | None = None, response=None, player=None, session_memory=None) -> str:
    """Execute one Browser Automation Engine 2.0 operation and return JSON."""
    params = parameters or {}
    capability = str(params.get("capability", "")).strip()
    action = str(params.get("action", "")).strip()
    operation_parameters = dict(params.get("parameters") or {})
    dry_run = bool(params.get("dry_run", False))

    if player:
        player.write_log(f"[Browser2] {capability}.{action}")

    adapter = None if dry_run else browser_control
    controller = BrowserController(adapter=adapter, context=AutomationContext(dry_run=dry_run))
    result = controller.execute(capability, action, operation_parameters)
    payload = {
        "ok": result.ok,
        "message": result.message,
        "data": result.data,
        "state": result.state.value if result.state else None,
        "permission_required": result.permission_required.name,
    }
    return json.dumps(payload, ensure_ascii=False)
