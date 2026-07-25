"""Browser Automation Engine 2.0 for JARVIS.

This module provides a structured browser-operation layer for future autonomous
web workflows while reusing the existing browser action implementation for live
execution.  The engine models browsers, profiles, sessions, tabs, navigation,
elements, downloads/uploads, authentication, history/bookmarks, permissions,
logging, and recovery as separate injectable components.
"""
from __future__ import annotations

import logging
import platform
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Protocol


class BrowserPermissionLevel(IntEnum):
    """Risk levels for browser operations."""

    SAFE = 0
    AUTHENTICATED = 1
    FILE_TRANSFER = 2
    DESTRUCTIVE = 3


class BrowserState(str, Enum):
    """Lifecycle states for browser sessions."""

    CREATED = "created"
    RUNNING = "running"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(slots=True)
class AutomationContext:
    """Shared context for browser automation operations."""

    user: str | None = None
    default_browser: str | None = None
    profile_name: str | None = None
    session_id: str | None = None
    dry_run: bool = False
    timeout_seconds: float = 60.0
    retries: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrowserResult:
    """Structured browser-operation result."""

    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    state: BrowserState | None = None
    permission_required: BrowserPermissionLevel = BrowserPermissionLevel.SAFE


@dataclass(slots=True)
class BrowserProfile:
    """Browser profile metadata for session/cookie reuse."""

    browser: str
    name: str = "default"
    path: Path | None = None
    persistent: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrowserSession:
    """A logical browser automation session."""

    browser: str
    profile: BrowserProfile
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: BrowserState = BrowserState.CREATED
    current_url: str | None = None
    title: str | None = None
    active_tab_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrowserElement:
    """DOM element abstraction used by planners and future web agents."""

    selector: str | None = None
    text: str | None = None
    role: str | None = None
    description: str | None = None
    index: int = 0


class BrowserActionAdapter(Protocol):
    """Callable adapter for existing browser action implementations."""

    def __call__(self, parameters: dict[str, Any], response: Any = None, player: Any = None,
                 session_memory: Any = None) -> str: ...


class BrowserLogger:
    """Structured logger for browser automation."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("jarvis.browser_engine")

    def event(self, message: str, **fields: Any) -> None:
        suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
        self._logger.info("%s%s", message, f" {suffix}" if suffix else "")


class BrowserPermissionManager:
    """Classifies and gates sensitive browser operations."""

    _levels = {
        "download": BrowserPermissionLevel.FILE_TRANSFER,
        "upload": BrowserPermissionLevel.FILE_TRANSFER,
        "login": BrowserPermissionLevel.AUTHENTICATED,
        "logout": BrowserPermissionLevel.AUTHENTICATED,
        "delete": BrowserPermissionLevel.DESTRUCTIVE,
        "clear_history": BrowserPermissionLevel.DESTRUCTIVE,
        "clear_cookies": BrowserPermissionLevel.DESTRUCTIVE,
        "history.clear": BrowserPermissionLevel.DESTRUCTIVE,
        "cookies.clear": BrowserPermissionLevel.DESTRUCTIVE,
    }

    def level_for(self, action: str) -> BrowserPermissionLevel:
        return max((level for token, level in self._levels.items() if token in action.lower()), default=BrowserPermissionLevel.SAFE)

    def check(self, action: str, context: AutomationContext) -> BrowserResult | None:
        level = self.level_for(action)
        if context.dry_run and level > BrowserPermissionLevel.SAFE:
            return BrowserResult(True, f"Dry run approved for browser action {action}.", permission_required=level)
        if level == BrowserPermissionLevel.DESTRUCTIVE:
            return BrowserResult(False, f"Confirmation required for browser action {action}.", permission_required=level)
        return None


class BrowserRecoveryEngine:
    """Retries transient browser failures and records recovery attempts."""

    def __init__(self, logger: BrowserLogger | None = None) -> None:
        self.logger = logger or BrowserLogger()

    def run(self, action: str, attempts: int, operation: Callable[[], BrowserResult]) -> BrowserResult:
        last: BrowserResult | None = None
        for attempt in range(1, max(attempts, 1) + 1):
            result = operation()
            if result.ok:
                if attempt > 1:
                    self.logger.event("Browser action recovered", action=action, attempt=attempt)
                return result
            last = result
            self.logger.event("Browser action failed", action=action, attempt=attempt, message=result.message)
            time.sleep(min(0.25 * attempt, 1.0))
        return last or BrowserResult(False, f"Browser action failed: {action}", state=BrowserState.FAILED)


class BrowserProfileManager:
    """Discovers and creates reusable browser profile locations."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.home() / ".jarvis_browser_profiles"

    def get_profile(self, browser: str, profile_name: str | None = None) -> BrowserProfile:
        name = profile_name or "default"
        path = self.base_dir / browser.lower() / name
        path.mkdir(parents=True, exist_ok=True)
        return BrowserProfile(browser=browser, name=name, path=path)


class BrowserManager:
    """Detects, opens, closes, switches, and restarts browsers."""

    _known_bins = {
        "chrome": ["google-chrome", "google-chrome-stable", "chrome", "chromium", "chromium-browser"],
        "edge": ["microsoft-edge", "microsoft-edge-stable", "msedge"],
        "firefox": ["firefox"],
        "brave": ["brave-browser", "brave"],
        "opera": ["opera", "opera-stable"],
        "vivaldi": ["vivaldi", "vivaldi-stable"],
        "safari": [],
    }

    def __init__(self, adapter: BrowserActionAdapter | None = None, logger: BrowserLogger | None = None) -> None:
        self.adapter = adapter
        self.logger = logger or BrowserLogger()

    def detect_installed(self) -> BrowserResult:
        system = platform.system()
        found: list[dict[str, str]] = []
        for name, bins in self._known_bins.items():
            if name == "safari" and system == "Darwin":
                found.append({"name": name, "path": "/Applications/Safari.app"})
                continue
            for binary in bins:
                resolved = shutil.which(binary)
                if resolved:
                    found.append({"name": name, "path": resolved})
                    break
        return BrowserResult(True, "Installed browsers detected.", {"browsers": found})

    def open(self, browser: str | None = None, url: str | None = None) -> BrowserResult:
        return self._call_existing({"action": "go_to", "browser": browser or "", "url": url or ""}, "Browser opened")

    def close(self, browser: str | None = None) -> BrowserResult:
        return self._call_existing({"action": "close", "browser": browser or ""}, "Browser closed")

    def close_all(self) -> BrowserResult:
        return self._call_existing({"action": "close_all"}, "All browser sessions closed")

    def switch(self, browser: str) -> BrowserResult:
        return self._call_existing({"action": "switch", "browser": browser}, "Browser switched")

    def restart(self, browser: str | None = None) -> BrowserResult:
        closed = self.close(browser)
        opened = self.open(browser)
        return BrowserResult(closed.ok and opened.ok, f"Restart: {closed.message}; {opened.message}", {"closed": closed.data, "opened": opened.data})

    def _call_existing(self, params: dict[str, Any], fallback: str) -> BrowserResult:
        if self.adapter is None:
            return BrowserResult(True, fallback, {"planned_parameters": params})
        text = self.adapter(params)
        return BrowserResult(not _looks_failed(text), text, {"legacy_result": text})


class TabManager:
    """Manages tabs in the active browser session."""

    def __init__(self, adapter: BrowserActionAdapter | None = None) -> None:
        self.adapter = adapter

    def new_tab(self, url: str | None = None, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "new_tab", "url": url or "", "browser": browser or ""}, "New tab planned")

    def close_tab(self, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "close_tab", "browser": browser or ""}, "Close tab planned")

    def switch_tab(self, index: int) -> BrowserResult:
        return BrowserResult(True, "Switch tab planned.", {"index": index, "future": "Playwright page selection adapter"})


class WindowManager:
    """Manages browser windows."""

    def new_window(self, browser: str | None = None, url: str | None = None) -> BrowserResult:
        return BrowserResult(True, "New browser window planned.", {"browser": browser, "url": url, "future": "new context/window adapter"})

    def close_window(self, browser: str | None = None) -> BrowserResult:
        return BrowserResult(True, "Close browser window planned.", {"browser": browser})


class NavigationEngine:
    """Handles URL navigation, history movement, refresh, and waiting."""

    def __init__(self, adapter: BrowserActionAdapter | None = None) -> None:
        self.adapter = adapter

    def open_url(self, url: str, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "go_to", "url": url, "browser": browser or ""}, "Open URL planned")

    def search(self, query: str, engine: str = "google", browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "search", "query": query, "engine": engine, "browser": browser or ""}, "Search planned")

    def back(self, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "back", "browser": browser or ""}, "Back planned")

    def forward(self, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "forward", "browser": browser or ""}, "Forward planned")

    def refresh(self, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "reload", "browser": browser or ""}, "Refresh planned")

    def wait_for_load(self, strategy: str = "domcontentloaded", timeout_seconds: float = 30) -> BrowserResult:
        return BrowserResult(True, "Wait strategy planned.", {"strategy": strategy, "timeout_seconds": timeout_seconds})


class ElementFinder:
    """Locates DOM elements by selector, text, role, or natural-language description."""

    def find(self, element: BrowserElement) -> BrowserResult:
        return BrowserResult(True, "Element lookup planned.", {"element": _element_dict(element)})


class InteractionEngine:
    """Executes clicks, typing, hotkeys, hover, drag/drop, and clipboard interactions."""

    def __init__(self, adapter: BrowserActionAdapter | None = None) -> None:
        self.adapter = adapter

    def click(self, element: BrowserElement, browser: str | None = None) -> BrowserResult:
        params = {"action": "smart_click" if element.description else "click", "browser": browser or "", "selector": element.selector, "text": element.text, "description": element.description or ""}
        return _adapter_result(self.adapter, params, "Click planned")

    def double_click(self, element: BrowserElement) -> BrowserResult:
        return BrowserResult(True, "Double click planned.", {"element": _element_dict(element), "future": "locator.dblclick adapter"})

    def right_click(self, element: BrowserElement) -> BrowserResult:
        return BrowserResult(True, "Right click planned.", {"element": _element_dict(element), "future": "locator.click(button='right') adapter"})

    def hover(self, element: BrowserElement) -> BrowserResult:
        return BrowserResult(True, "Hover planned.", {"element": _element_dict(element)})

    def drag_and_drop(self, source: BrowserElement, target: BrowserElement) -> BrowserResult:
        return BrowserResult(True, "Drag and drop planned.", {"source": _element_dict(source), "target": _element_dict(target)})

    def type_text(self, text: str, element: BrowserElement | None = None, browser: str | None = None) -> BrowserResult:
        params = {"action": "smart_type" if element and element.description else "type", "browser": browser or "", "selector": element.selector if element else None, "description": element.description if element else "", "text": text}
        return _adapter_result(self.adapter, params, "Type planned")

    def press(self, key: str, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "press", "key": key, "browser": browser or ""}, "Key press planned")

    def hotkey(self, keys: list[str]) -> BrowserResult:
        return BrowserResult(True, "Browser hotkey planned.", {"keys": keys, "future": "page.keyboard.press chord adapter"})


class DOMInspector:
    """Reads page text, source, current URL, title, and state."""

    def __init__(self, adapter: BrowserActionAdapter | None = None) -> None:
        self.adapter = adapter

    def text(self, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "get_text", "browser": browser or ""}, "Read page text planned")

    def current_url(self, browser: str | None = None) -> BrowserResult:
        return _adapter_result(self.adapter, {"action": "get_url", "browser": browser or ""}, "Current URL planned")

    def page_source(self) -> BrowserResult:
        return BrowserResult(True, "Page source planned.", {"future": "page.content adapter"})

    def title(self) -> BrowserResult:
        return BrowserResult(True, "Page title planned.", {"future": "page.title adapter"})

    def state(self) -> BrowserResult:
        return BrowserResult(True, "Browser state planned.", {"future": "session registry snapshot"})


class CookieManager:
    """Manages cookies without hardcoding credentials."""

    def save(self, session: BrowserSession) -> BrowserResult:
        return BrowserResult(True, "Cookie save planned.", {"session_id": session.id, "secure_storage": "future encrypted storage"})

    def load(self, profile: BrowserProfile) -> BrowserResult:
        return BrowserResult(True, "Cookie load planned.", {"profile": str(profile.path), "secure_storage": "future encrypted storage"})

    def clear(self) -> BrowserResult:
        return BrowserResult(False, "Confirmation required for clearing cookies.", permission_required=BrowserPermissionLevel.DESTRUCTIVE)


class DownloadManager:
    """Handles download lifecycle and future download directory policies."""

    def wait_for_download(self, timeout_seconds: float = 120) -> BrowserResult:
        return BrowserResult(True, "Download wait planned.", {"timeout_seconds": timeout_seconds})


class UploadManager:
    """Handles file upload flows."""

    def upload(self, path: str, element: BrowserElement | None = None) -> BrowserResult:
        return BrowserResult(True, "File upload planned.", {"path": path, "element": _element_dict(element) if element else None})


class AuthenticationManager:
    """Coordinates saved sessions, login detection, and future MFA workflows."""

    def login(self, service: str, profile: BrowserProfile) -> BrowserResult:
        return BrowserResult(True, "Login flow planned; credentials are not hardcoded.", {"service": service, "profile": profile.name, "future": "secure credential provider + MFA callback"}, permission_required=BrowserPermissionLevel.AUTHENTICATED)

    def detect_expired(self) -> BrowserResult:
        return BrowserResult(True, "Session-expiration detection planned.", {"future": "DOM/auth-state detector"})


class BrowserHistory:
    """Provides browser history actions."""

    def show(self) -> BrowserResult:
        return BrowserResult(True, "History view planned.", {"future": "browser-specific history adapter"})

    def clear(self) -> BrowserResult:
        return BrowserResult(False, "Confirmation required for clearing browser history.", permission_required=BrowserPermissionLevel.DESTRUCTIVE)


class BookmarkManager:
    """Provides bookmark actions."""

    def add(self, url: str, title: str | None = None) -> BrowserResult:
        return BrowserResult(True, "Bookmark add planned.", {"url": url, "title": title})

    def list(self) -> BrowserResult:
        return BrowserResult(True, "Bookmark listing planned.", {"future": "browser-specific bookmark adapter"})


class BrowserController:
    """Facade coordinating Browser Automation Engine 2.0 components."""

    def __init__(self, adapter: BrowserActionAdapter | None = None, context: AutomationContext | None = None,
                 logger: BrowserLogger | None = None, permissions: BrowserPermissionManager | None = None,
                 recovery: BrowserRecoveryEngine | None = None, profiles: BrowserProfileManager | None = None) -> None:
        self.adapter = adapter
        self.context = context or AutomationContext()
        self.logger = logger or BrowserLogger()
        self.permissions = permissions or BrowserPermissionManager()
        self.recovery = recovery or BrowserRecoveryEngine(self.logger)
        self.profiles = profiles or BrowserProfileManager()
        self.browser = BrowserManager(adapter, self.logger)
        self.tabs = TabManager(adapter)
        self.windows = WindowManager()
        self.navigation = NavigationEngine(adapter)
        self.elements = ElementFinder()
        self.interactions = InteractionEngine(adapter)
        self.dom = DOMInspector(adapter)
        self.cookies = CookieManager()
        self.downloads = DownloadManager()
        self.uploads = UploadManager()
        self.auth = AuthenticationManager()
        self.history = BrowserHistory()
        self.bookmarks = BookmarkManager()

    def execute(self, capability: str, action: str, parameters: dict[str, Any] | None = None) -> BrowserResult:
        """Execute one browser automation command by capability/action."""
        params = parameters or {}
        denied = self.permissions.check(f"{capability}.{action}", self.context)
        if denied and not denied.ok:
            return denied
        if denied and self.context.dry_run:
            return denied

        def operation() -> BrowserResult:
            return self._dispatch(capability.lower().strip(), action.lower().strip(), params)

        return self.recovery.run(action, self.context.retries + 1, operation)

    def _dispatch(self, capability: str, action: str, params: dict[str, Any]) -> BrowserResult:
        browser = params.get("browser") or self.context.default_browser
        profile = self.profiles.get_profile(browser or "default", params.get("profile") or self.context.profile_name)
        if capability == "browser":
            if action == "detect": return self.browser.detect_installed()
            if action == "open": return self.browser.open(browser, params.get("url"))
            if action == "close": return self.browser.close(browser)
            if action == "close_all": return self.browser.close_all()
            if action == "switch": return self.browser.switch(str(browser or params.get("target", "")))
            if action == "restart": return self.browser.restart(browser)
        if capability == "navigation":
            if action == "open_url": return self.navigation.open_url(str(params.get("url", "")), browser)
            if action == "search": return self.navigation.search(str(params.get("query", "")), str(params.get("engine", "google")), browser)
            if action == "back": return self.navigation.back(browser)
            if action == "forward": return self.navigation.forward(browser)
            if action == "refresh": return self.navigation.refresh(browser)
            if action == "wait": return self.navigation.wait_for_load(str(params.get("strategy", "domcontentloaded")), float(params.get("timeout_seconds", 30)))
        if capability == "tab":
            if action == "new": return self.tabs.new_tab(params.get("url"), browser)
            if action == "close": return self.tabs.close_tab(browser)
            if action == "switch": return self.tabs.switch_tab(int(params.get("index", 0)))
        if capability == "window":
            if action == "new": return self.windows.new_window(browser, params.get("url"))
            if action == "close": return self.windows.close_window(browser)
        if capability == "element":
            if action == "find": return self.elements.find(_element_from_params(params))
            if action == "click": return self.interactions.click(_element_from_params(params), browser)
            if action == "double_click": return self.interactions.double_click(_element_from_params(params))
            if action == "right_click": return self.interactions.right_click(_element_from_params(params))
            if action == "hover": return self.interactions.hover(_element_from_params(params))
            if action == "drag_drop": return self.interactions.drag_and_drop(_element_from_params(params.get("source", {})), _element_from_params(params.get("target", {})))
        if capability == "input":
            if action == "type": return self.interactions.type_text(str(params.get("text", "")), _element_from_params(params) if any(k in params for k in ("selector", "description", "text_selector")) else None, browser)
            if action == "press": return self.interactions.press(str(params.get("key", "Enter")), browser)
            if action == "hotkey": return self.interactions.hotkey(_keys_from_params(params))
            if action in {"copy", "paste", "select_text"}: return BrowserResult(True, f"Input {action} planned.", {"future": "clipboard/selection adapter"})
        if capability == "dom":
            if action == "text": return self.dom.text(browser)
            if action == "url": return self.dom.current_url(browser)
            if action == "source": return self.dom.page_source()
            if action == "title": return self.dom.title()
            if action == "state": return self.dom.state()
        if capability == "cookies":
            if action == "save": return self.cookies.save(BrowserSession(browser or "default", profile))
            if action == "load": return self.cookies.load(profile)
            if action == "clear": return self.cookies.clear()
        if capability == "download" and action == "wait": return self.downloads.wait_for_download(float(params.get("timeout_seconds", 120)))
        if capability == "upload" and action == "file": return self.uploads.upload(str(params.get("path", "")), _element_from_params(params) if params else None)
        if capability == "auth":
            if action == "login": return self.auth.login(str(params.get("service", browser or "website")), profile)
            if action == "detect_expired": return self.auth.detect_expired()
        if capability == "history":
            if action == "show": return self.history.show()
            if action == "clear": return self.history.clear()
        if capability == "bookmark":
            if action == "add": return self.bookmarks.add(str(params.get("url", "")), params.get("title"))
            if action == "list": return self.bookmarks.list()
        if capability == "page" and action == "screenshot":
            return _adapter_result(self.adapter, {"action": "screenshot", "path": params.get("path"), "browser": browser or ""}, "Screenshot planned")
        return BrowserResult(False, f"Unsupported browser command: {capability}.{action}")


def _adapter_result(adapter: BrowserActionAdapter | None, params: dict[str, Any], fallback: str) -> BrowserResult:
    if adapter is None:
        return BrowserResult(True, fallback, {"planned_parameters": params})
    text = adapter(params)
    return BrowserResult(not _looks_failed(text), text, {"legacy_result": text})


def _looks_failed(text: str) -> bool:
    lowered = str(text).lower()
    return any(token in lowered for token in ("error", "failed", "could not", "timeout", "unknown action"))


def _element_from_params(params: dict[str, Any]) -> BrowserElement:
    return BrowserElement(
        selector=params.get("selector") or params.get("text_selector"),
        text=params.get("text_match") or params.get("text"),
        role=params.get("role"),
        description=params.get("description"),
        index=int(params.get("index", 0) or 0),
    )


def _element_dict(element: BrowserElement) -> dict[str, Any]:
    return {
        "selector": element.selector,
        "text": element.text,
        "role": element.role,
        "description": element.description,
        "index": element.index,
    }


def _keys_from_params(params: dict[str, Any]) -> list[str]:
    keys = params.get("keys") or []
    if isinstance(keys, str):
        return [part.strip() for part in keys.replace("+", ",").split(",") if part.strip()]
    return [str(key) for key in keys]
