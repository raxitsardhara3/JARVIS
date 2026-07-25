"""Universal computer control engine for JARVIS.

The engine is a reusable foundation for desktop automation.  It coordinates
mouse, keyboard, window, clipboard, screen, application, desktop, monitor, file
explorer, and system-control adapters behind one high-level ``ComputerController``.
Dangerous operations are guarded by ``PermissionManager`` so future autonomous
workflows can request confirmation before performing destructive actions.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
except Exception:  # pragma: no cover - optional runtime/headless dependency
    pyautogui = None  # type: ignore[assignment]

try:
    import pyperclip
except ImportError:  # pragma: no cover - optional runtime dependency
    pyperclip = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None  # type: ignore[assignment]


class PermissionLevel(IntEnum):
    """Risk levels for computer-control operations."""

    SAFE = 0
    SENSITIVE = 1
    DESTRUCTIVE = 2
    ADMIN = 3


class AutomationCapability(str, Enum):
    """Supported capability families for registry/planner discovery."""

    MOUSE = "mouse"
    KEYBOARD = "keyboard"
    CLIPBOARD = "clipboard"
    WINDOW = "window"
    APPLICATION = "application"
    DESKTOP = "desktop"
    SCREEN = "screen"
    MONITOR = "monitor"
    FILE_EXPLORER = "file_explorer"
    SYSTEM = "system"


@dataclass(slots=True)
class AutomationContext:
    """Shared context passed through the universal control engine."""

    user: str | None = None
    dry_run: bool = False
    require_confirmation: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AutomationResult:
    """Structured result returned by every manager operation."""

    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    permission_required: PermissionLevel = PermissionLevel.SAFE


@dataclass(slots=True)
class MonitorInfo:
    """Basic monitor/screen geometry."""

    index: int
    width: int
    height: int
    x: int = 0
    y: int = 0
    primary: bool = True


@dataclass(slots=True)
class WindowInfo:
    """Best-effort window description across operating systems."""

    title: str
    handle: str | int | None = None
    process: str | None = None
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    focused: bool = False


class ConfirmationProvider(Protocol):
    """Protocol for future UI/voice confirmation providers."""

    def __call__(self, operation: str, level: PermissionLevel, details: dict[str, Any]) -> bool: ...


class AutomationLogger:
    """Thin logging wrapper for automation telemetry."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("jarvis.computer_control")

    def event(self, message: str, **fields: Any) -> None:
        suffix = " ".join(f"{key}={value!r}" for key, value in fields.items())
        self._logger.info("%s%s", message, f" {suffix}" if suffix else "")


class PermissionManager:
    """Evaluates and gates sensitive computer-control actions."""

    _dangerous_keywords = {
        "delete": PermissionLevel.DESTRUCTIVE,
        "remove": PermissionLevel.DESTRUCTIVE,
        "shutdown": PermissionLevel.ADMIN,
        "restart": PermissionLevel.ADMIN,
        "terminate": PermissionLevel.SENSITIVE,
        "kill": PermissionLevel.SENSITIVE,
        "registry": PermissionLevel.ADMIN,
        "admin": PermissionLevel.ADMIN,
    }

    def __init__(self, confirmation_provider: ConfirmationProvider | None = None,
                 default_allowed_level: PermissionLevel = PermissionLevel.SENSITIVE) -> None:
        self.confirmation_provider = confirmation_provider
        self.default_allowed_level = default_allowed_level

    def level_for(self, operation: str) -> PermissionLevel:
        lower = operation.lower()
        return max((level for word, level in self._dangerous_keywords.items() if word in lower), default=PermissionLevel.SAFE)

    def ensure_allowed(self, operation: str, context: AutomationContext, details: dict[str, Any] | None = None) -> AutomationResult | None:
        level = self.level_for(operation)
        if context.dry_run:
            return AutomationResult(True, f"Dry run approved for {operation}.", permission_required=level)
        if level <= self.default_allowed_level:
            return None
        if context.require_confirmation and self.confirmation_provider:
            approved = self.confirmation_provider(operation, level, details or {})
            if approved:
                return None
        return AutomationResult(False, f"Confirmation required for {operation}.", permission_required=level)


class BaseManager:
    """Base class for managers sharing context, permissions, and logging."""

    def __init__(self, context: AutomationContext | None = None, permissions: PermissionManager | None = None,
                 logger: AutomationLogger | None = None) -> None:
        self.context = context or AutomationContext()
        self.permissions = permissions or PermissionManager()
        self.logger = logger or AutomationLogger()

    def _pyautogui(self):
        if pyautogui is None:
            raise RuntimeError("PyAutoGUI is not installed. Run: pip install pyautogui")
        return pyautogui

    def _ok(self, message: str, **data: Any) -> AutomationResult:
        self.logger.event(message, **data)
        return AutomationResult(True, message, data)


class MouseController(BaseManager):
    """Controls pointer movement, clicks, dragging, and scrolling."""

    def move(self, x: int, y: int, duration: float = 0.2) -> AutomationResult:
        if self.context.dry_run:
            return self._ok("Mouse move planned", x=x, y=y)
        self._pyautogui().moveTo(x, y, duration=duration)
        return self._ok("Mouse moved", x=x, y=y)

    def click(self, x: int | None = None, y: int | None = None, button: str = "left", clicks: int = 1) -> AutomationResult:
        if self.context.dry_run:
            return self._ok("Mouse click planned", x=x, y=y, button=button, clicks=clicks)
        gui = self._pyautogui()
        if x is not None and y is not None:
            gui.click(x, y, button=button, clicks=clicks)
        else:
            gui.click(button=button, clicks=clicks)
        return self._ok("Mouse clicked", x=x, y=y, button=button, clicks=clicks)

    def double_click(self, x: int | None = None, y: int | None = None) -> AutomationResult:
        return self.click(x=x, y=y, clicks=2)

    def right_click(self, x: int | None = None, y: int | None = None) -> AutomationResult:
        return self.click(x=x, y=y, button="right")

    def middle_click(self, x: int | None = None, y: int | None = None) -> AutomationResult:
        return self.click(x=x, y=y, button="middle")

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.4) -> AutomationResult:
        if self.context.dry_run:
            return self._ok("Mouse drag planned", x1=x1, y1=y1, x2=x2, y2=y2)
        gui = self._pyautogui()
        gui.moveTo(x1, y1, duration=0.1)
        gui.dragTo(x2, y2, duration=duration, button="left")
        return self._ok("Mouse dragged", x1=x1, y1=y1, x2=x2, y2=y2)

    def scroll(self, amount: int, horizontal: bool = False) -> AutomationResult:
        if self.context.dry_run:
            return self._ok("Mouse scroll planned", amount=amount, horizontal=horizontal)
        gui = self._pyautogui()
        gui.hscroll(amount) if horizontal else gui.scroll(amount)
        return self._ok("Mouse scrolled", amount=amount, horizontal=horizontal)


class KeyboardController(BaseManager):
    """Controls typing, single keys, shortcuts, and hotkeys."""

    def type_text(self, text: str, interval: float = 0.02) -> AutomationResult:
        if self.context.dry_run:
            return self._ok("Keyboard typing planned", text=text[:80])
        self._pyautogui().write(text, interval=interval)
        return self._ok("Text typed", text=text[:80])

    def press(self, key: str) -> AutomationResult:
        if self.context.dry_run:
            return self._ok("Key press planned", key=key)
        self._pyautogui().press(key)
        return self._ok("Key pressed", key=key)

    def hotkey(self, *keys: str) -> AutomationResult:
        if self.context.dry_run:
            return self._ok("Hotkey planned", keys=list(keys))
        self._pyautogui().hotkey(*keys)
        return self._ok("Hotkey pressed", keys=list(keys))


class ClipboardManager(BaseManager):
    """Provides copy/paste plus an in-memory clipboard history."""

    def __init__(self, *args: Any, history_limit: int = 25, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history: deque[str] = deque(maxlen=history_limit)

    def copy(self) -> AutomationResult:
        if pyperclip is not None:
            value = pyperclip.paste()
        else:
            KeyboardController(self.context, self.permissions, self.logger).hotkey("ctrl", "c")
            time.sleep(0.2)
            value = ""
        if value:
            self.history.appendleft(value)
        return self._ok("Clipboard copied", text=value[:80], history_size=len(self.history))

    def set_text(self, text: str) -> AutomationResult:
        if pyperclip is None:
            return AutomationResult(False, "pyperclip is not installed; clipboard text cannot be set directly.")
        if not self.context.dry_run:
            pyperclip.copy(text)
        self.history.appendleft(text)
        return self._ok("Clipboard text set", text=text[:80], history_size=len(self.history))

    def paste(self, text: str | None = None) -> AutomationResult:
        if text is not None:
            result = self.set_text(text)
            if not result.ok:
                return result
        paste_key = "command" if platform.system() == "Darwin" else "ctrl"
        return KeyboardController(self.context, self.permissions, self.logger).hotkey(paste_key, "v")

    def get_history(self) -> AutomationResult:
        return self._ok("Clipboard history returned", items=list(self.history))


class WindowManager(BaseManager):
    """Best-effort cross-platform window focus/move/resize controls."""

    def list_windows(self) -> AutomationResult:
        if pyautogui is not None and hasattr(pyautogui, "getAllWindows"):
            windows = [asdict(WindowInfo(title=w.title, x=w.left, y=w.top, width=w.width, height=w.height))
                       for w in pyautogui.getAllWindows() if getattr(w, "title", "")]
            return self._ok("Windows detected", windows=windows)
        return self._ok("Window listing unavailable on this platform", windows=[])

    def focus(self, title: str) -> AutomationResult:
        if self.context.dry_run:
            return self._ok("Window focus planned", title=title)
        if pyautogui is not None and hasattr(pyautogui, "getWindowsWithTitle"):
            matches = pyautogui.getWindowsWithTitle(title)
            if matches:
                matches[0].activate()
                return self._ok("Window focused", title=title)
        return self._platform_window_command("focus", title)

    def close(self, title: str | None = None) -> AutomationResult:
        denied = self.permissions.ensure_allowed("close window", self.context, {"title": title})
        if denied:
            return denied
        if title:
            focused = self.focus(title)
            if not focused.ok:
                return focused
        close_key = "command" if platform.system() == "Darwin" else "alt"
        close_arg = "w" if platform.system() == "Darwin" else "f4"
        return KeyboardController(self.context, self.permissions, self.logger).hotkey(close_key, close_arg)

    def maximize(self, title: str | None = None) -> AutomationResult:
        return self._window_method_or_hotkey(title, "maximize", ["win", "up"])

    def minimize(self, title: str | None = None) -> AutomationResult:
        return self._window_method_or_hotkey(title, "minimize", ["win", "down"])

    def move(self, title: str, x: int, y: int) -> AutomationResult:
        if pyautogui is not None and hasattr(pyautogui, "getWindowsWithTitle"):
            matches = pyautogui.getWindowsWithTitle(title)
            if matches and not self.context.dry_run:
                matches[0].moveTo(x, y)
                return self._ok("Window moved", title=title, x=x, y=y)
        return self._ok("Window move planned", title=title, x=x, y=y)

    def resize(self, title: str, width: int, height: int) -> AutomationResult:
        if pyautogui is not None and hasattr(pyautogui, "getWindowsWithTitle"):
            matches = pyautogui.getWindowsWithTitle(title)
            if matches and not self.context.dry_run:
                matches[0].resizeTo(width, height)
                return self._ok("Window resized", title=title, width=width, height=height)
        return self._ok("Window resize planned", title=title, width=width, height=height)

    def _window_method_or_hotkey(self, title: str | None, method: str, fallback: list[str]) -> AutomationResult:
        if title:
            self.focus(title)
        if pyautogui is not None and hasattr(pyautogui, "getActiveWindow"):
            window = pyautogui.getActiveWindow()
            if window and hasattr(window, method) and not self.context.dry_run:
                getattr(window, method)()
                return self._ok(f"Window {method}d", title=title)
        return KeyboardController(self.context, self.permissions, self.logger).hotkey(*fallback)

    def _platform_window_command(self, action: str, title: str) -> AutomationResult:
        system = platform.system()
        if system == "Linux":
            for command in (["wmctrl", "-a", title], ["xdotool", "search", "--name", title, "windowactivate"]):
                try:
                    subprocess.run(command, capture_output=True, timeout=5)
                    return self._ok("Window focused", title=title, method=command[0])
                except FileNotFoundError:
                    continue
        if system == "Darwin":
            script = f'tell application "System Events" to set frontmost of (first process whose name contains "{title}") to true'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
            return self._ok("Window focused", title=title, method="osascript")
        if system == "Windows":
            script = f'(New-Object -ComObject WScript.Shell).AppActivate("{title}")'
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, timeout=5)
            return self._ok("Window focused", title=title, method="powershell")
        return AutomationResult(False, f"Window {action} unsupported on {system}.")


class ApplicationManager(BaseManager):
    """Launches, detects, lists, and closes applications/processes."""

    def launch(self, name: str) -> AutomationResult:
        if not name:
            return AutomationResult(False, "Application name is required.")
        if self.context.dry_run:
            return self._ok("Application launch planned", app=name)
        system = platform.system()
        try:
            if system == "Darwin":
                result = subprocess.run(["open", "-a", name], capture_output=True, timeout=8)
                if result.returncode == 0:
                    return self._ok("Application launched", app=name)
            elif system == "Windows":
                subprocess.Popen(name, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self._ok("Application launched", app=name)
            else:
                binary = shutil.which(name) or shutil.which(name.lower()) or shutil.which(name.lower().replace(" ", "-"))
                if binary:
                    subprocess.Popen([binary], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return self._ok("Application launched", app=name, binary=binary)
                subprocess.run(["xdg-open", name], capture_output=True, timeout=5)
                return self._ok("Application launch requested", app=name)
        except Exception as exc:
            return AutomationResult(False, f"Could not launch {name}: {exc}")
        return AutomationResult(False, f"Could not launch {name}.")

    def close(self, name: str) -> AutomationResult:
        denied = self.permissions.ensure_allowed("terminate process", self.context, {"app": name})
        if denied:
            return denied
        if psutil is None:
            return AutomationResult(False, "psutil is required to close applications by process name.")
        closed = 0
        for proc in psutil.process_iter(["name"]):
            proc_name = (proc.info.get("name") or "").lower()
            if name.lower() in proc_name:
                if not self.context.dry_run:
                    proc.terminate()
                closed += 1
        return self._ok("Application close requested", app=name, processes=closed)

    def is_running(self, name: str) -> AutomationResult:
        if psutil is None:
            return AutomationResult(False, "psutil is required for process detection.")
        matches = [proc.info for proc in psutil.process_iter(["pid", "name"]) if name.lower() in (proc.info.get("name") or "").lower()]
        return self._ok("Application detection complete", app=name, running=bool(matches), matches=matches)

    def running_processes(self, limit: int = 50) -> AutomationResult:
        if psutil is None:
            return AutomationResult(False, "psutil is required for process listing.")
        processes = list(psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]))[:limit]
        return self._ok("Running processes listed", processes=[p.info for p in processes])


class DesktopManager(BaseManager):
    """Handles desktop icons, taskbar placeholders, and desktop directory data."""

    def desktop_path(self) -> Path:
        if platform.system() == "Linux" and os.environ.get("XDG_DESKTOP_DIR"):
            return Path(os.environ["XDG_DESKTOP_DIR"]).expanduser()
        return Path.home() / "Desktop"

    def icons(self) -> AutomationResult:
        desktop = self.desktop_path()
        items = []
        if desktop.exists():
            items = [{"name": item.name, "path": str(item), "directory": item.is_dir()} for item in desktop.iterdir() if not item.name.startswith(".")]
        return self._ok("Desktop icons listed", desktop=str(desktop), icons=items)

    def show_desktop(self) -> AutomationResult:
        if platform.system() == "Darwin":
            return KeyboardController(self.context, self.permissions, self.logger).hotkey("command", "f3")
        return KeyboardController(self.context, self.permissions, self.logger).hotkey("win", "d")

    def taskbar(self) -> AutomationResult:
        return self._ok("Taskbar automation placeholder", supported=False, future="OS-specific taskbar adapter")

    def notification_center(self) -> AutomationResult:
        if platform.system() == "Darwin":
            return KeyboardController(self.context, self.permissions, self.logger).hotkey("control", "command", "n")
        if platform.system() == "Windows":
            return KeyboardController(self.context, self.permissions, self.logger).hotkey("win", "n")
        return self._ok("Notification center placeholder", supported=False)


class ScreenManager(BaseManager):
    """Captures screenshots and exposes recording architecture hooks."""

    def screenshot(self, path: str | None = None) -> AutomationResult:
        target = Path(path).expanduser() if path else Path.home() / "Desktop" / "jarvis_screenshot.png"
        if self.context.dry_run:
            return self._ok("Screenshot planned", path=str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        image = self._pyautogui().screenshot()
        image.save(str(target))
        return self._ok("Screenshot captured", path=str(target))

    def start_recording(self, path: str | None = None) -> AutomationResult:
        return self._ok("Screen recording start placeholder", path=path, supported=False, future="ffmpeg/mss adapter")

    def stop_recording(self) -> AutomationResult:
        return self._ok("Screen recording stop placeholder", supported=False, future="recording session adapter")


class MonitorManager(BaseManager):
    """Detects monitor geometry for current and future multi-monitor automation."""

    def list_monitors(self) -> AutomationResult:
        if pyautogui is None:
            return AutomationResult(False, "PyAutoGUI is required for monitor detection.")
        width, height = pyautogui.size()
        return self._ok("Monitors detected", monitors=[asdict(MonitorInfo(0, width, height))])


class FileExplorerController(BaseManager):
    """Automates common file explorer actions safely."""

    def open_path(self, path: str) -> AutomationResult:
        target = Path(path).expanduser()
        if self.context.dry_run:
            return self._ok("File explorer open planned", path=str(target))
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["explorer", str(target)])
        elif system == "Darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return self._ok("File explorer opened", path=str(target))

    def copy_path(self, source: str, destination: str) -> AutomationResult:
        src = Path(source).expanduser(); dst = Path(destination).expanduser()
        if self.context.dry_run:
            return self._ok("File copy planned", source=str(src), destination=str(dst))
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return self._ok("File copied", source=str(src), destination=str(dst))

    def move_path(self, source: str, destination: str) -> AutomationResult:
        denied = self.permissions.ensure_allowed("move file", self.context, {"source": source, "destination": destination})
        if denied:
            return denied
        if not self.context.dry_run:
            shutil.move(str(Path(source).expanduser()), str(Path(destination).expanduser()))
        return self._ok("File moved", source=source, destination=destination)


class SystemController(BaseManager):
    """System-level operations guarded by permissions."""

    def shutdown(self) -> AutomationResult:
        denied = self.permissions.ensure_allowed("shutdown PC", self.context)
        if denied:
            return denied
        if self.context.dry_run:
            return self._ok("Shutdown planned")
        command = ["shutdown", "/s", "/t", "0"] if platform.system() == "Windows" else ["shutdown", "-h", "now"]
        subprocess.Popen(command)
        return self._ok("Shutdown requested")

    def restart(self) -> AutomationResult:
        denied = self.permissions.ensure_allowed("restart PC", self.context)
        if denied:
            return denied
        if self.context.dry_run:
            return self._ok("Restart planned")
        command = ["shutdown", "/r", "/t", "0"] if platform.system() == "Windows" else ["shutdown", "-r", "now"]
        subprocess.Popen(command)
        return self._ok("Restart requested")


class ComputerController:
    """Facade that routes universal computer actions to focused managers."""

    def __init__(self, context: AutomationContext | None = None, permissions: PermissionManager | None = None,
                 logger: AutomationLogger | None = None) -> None:
        self.context = context or AutomationContext()
        self.permissions = permissions or PermissionManager()
        self.logger = logger or AutomationLogger()
        self.mouse = MouseController(self.context, self.permissions, self.logger)
        self.keyboard = KeyboardController(self.context, self.permissions, self.logger)
        self.clipboard = ClipboardManager(self.context, self.permissions, self.logger)
        self.windows = WindowManager(self.context, self.permissions, self.logger)
        self.applications = ApplicationManager(self.context, self.permissions, self.logger)
        self.desktop = DesktopManager(self.context, self.permissions, self.logger)
        self.screen = ScreenManager(self.context, self.permissions, self.logger)
        self.monitors = MonitorManager(self.context, self.permissions, self.logger)
        self.files = FileExplorerController(self.context, self.permissions, self.logger)
        self.system = SystemController(self.context, self.permissions, self.logger)

    def execute(self, capability: str, action: str, parameters: dict[str, Any] | None = None) -> AutomationResult:
        """Execute a universal control operation by capability/action names."""
        params = parameters or {}
        target = capability.lower().strip()
        action_name = action.lower().strip()
        routes: dict[tuple[str, str], Callable[..., AutomationResult]] = {
            ("mouse", "move"): self.mouse.move,
            ("mouse", "click"): self.mouse.click,
            ("mouse", "double_click"): self.mouse.double_click,
            ("mouse", "right_click"): self.mouse.right_click,
            ("mouse", "middle_click"): self.mouse.middle_click,
            ("mouse", "drag"): self.mouse.drag,
            ("mouse", "scroll"): self.mouse.scroll,
            ("keyboard", "type"): self.keyboard.type_text,
            ("keyboard", "press"): self.keyboard.press,
            ("keyboard", "hotkey"): lambda **kw: self.keyboard.hotkey(*_keys_from_params(kw)),
            ("clipboard", "copy"): self.clipboard.copy,
            ("clipboard", "paste"): self.clipboard.paste,
            ("clipboard", "set_text"): self.clipboard.set_text,
            ("clipboard", "history"): self.clipboard.get_history,
            ("window", "list"): self.windows.list_windows,
            ("window", "focus"): self.windows.focus,
            ("window", "close"): self.windows.close,
            ("window", "resize"): self.windows.resize,
            ("window", "move"): self.windows.move,
            ("window", "minimize"): self.windows.minimize,
            ("window", "maximize"): self.windows.maximize,
            ("application", "launch"): self.applications.launch,
            ("application", "close"): self.applications.close,
            ("application", "detect"): self.applications.is_running,
            ("application", "processes"): self.applications.running_processes,
            ("desktop", "icons"): self.desktop.icons,
            ("desktop", "show"): self.desktop.show_desktop,
            ("desktop", "taskbar"): self.desktop.taskbar,
            ("desktop", "notifications"): self.desktop.notification_center,
            ("screen", "screenshot"): self.screen.screenshot,
            ("screen", "start_recording"): self.screen.start_recording,
            ("screen", "stop_recording"): self.screen.stop_recording,
            ("monitor", "list"): self.monitors.list_monitors,
            ("file_explorer", "open"): self.files.open_path,
            ("file_explorer", "copy"): self.files.copy_path,
            ("file_explorer", "move"): self.files.move_path,
            ("system", "shutdown"): self.system.shutdown,
            ("system", "restart"): self.system.restart,
        }
        handler = routes.get((target, action_name))
        if handler is None:
            return AutomationResult(False, f"Unsupported computer control action: {capability}.{action}")
        try:
            return handler(**params)
        except TypeError as exc:
            return AutomationResult(False, f"Invalid parameters for {capability}.{action}: {exc}")
        except Exception as exc:
            return AutomationResult(False, f"Computer control failed for {capability}.{action}: {exc}")


def _keys_from_params(params: dict[str, Any]) -> list[str]:
    keys = params.get("keys") or params.get("sequence") or []
    if isinstance(keys, str):
        return [part.strip() for part in keys.replace("+", ",").split(",") if part.strip()]
    return [str(key) for key in keys]
