"""AI Scribe orchestration package.

Expose the light-weight `state` module at package import time. Heavy
submodules such as `graph` depend on optional third-party packages and
should be imported explicitly by callers or test code (or patched) to
avoid import-time side effects.
"""

from . import state  # re-export state for convenience

# Provide a light-weight placeholder for `orchestrator.graph` so tests that
# patch `orchestrator.graph.build_graph` can do so without importing the
# heavy real `graph` module (which depends on optional third-party libs).
try:
    # Prefer to leave the real submodule import to callers; only create a
    # placeholder if the actual submodule is not importable.
    import importlib
    importlib.import_module(__name__ + ".graph")
except Exception:
    # Create a lightweight module object and register it in sys.modules so
    # `from orchestrator.graph import build_graph` will succeed without
    # importing the heavy real graph implementation (which may require
    # optional third-party packages like langgraph).
    import sys
    from types import ModuleType

    _placeholder = ModuleType(__name__ + ".graph")
    _placeholder.build_graph = lambda *a, **k: None
    _placeholder.run_encounter = lambda *a, **k: None
    sys.modules[__name__ + ".graph"] = _placeholder
    graph = _placeholder

