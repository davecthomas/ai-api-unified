# Refactor Conditional Dependencies Using Plugin/Factory Pattern

## The Problem
Currently, our library handles optional dependencies defined in `pyproject.toml` using brittle conditional imports. The core library aggressively attempts to import these optional modules (e.g., heavy NLP models, vector database clients) and uses global flags (like `FEATURE_AVAILABLE`) and `try/except ImportError` blocks to survive failures. 

This approach violates several best practices:
1. **Polluted Core Logic:** Core classes are littered with `if/else` branches checking if an optional dependency is available before executing logic.
2. **Static Analysis Nightmares:** Type checkers (like Pyright/Mypy) constantly flag missing imports when evaluating the core library in environments where the optional extras aren't installed. This forces us to suppress errors using `# type: ignore` declarations throughout the codebase.
3. **Leaky Abstractions:** The core library shouldn't need to know the implementation details or import locations of libraries that the user hasn't actively opted into using.

## How We Can Fix This
To resolve this systemic mess while keeping heavy dependencies strictly optional, we need to **invert the control**. Instead of the core library surviving import failures, we should use a Plugin or Factory Pattern backed by formal Protocols (ABCs).

Here is the general concept of how we should clean this up across all conditional dependencies in the repository:

### 1. Stable Factory/Base API, Intentional Import Contract Change
**Current Constraint:** The factory and base-interface API exposed to callers of
`ai_api_unified` should remain stable, but import paths for optional
concrete provider classes are allowed to change.

In practice, this means:
- Factory entry points and base classes remain the primary public contract.
- Optional concrete provider classes are no longer re-exported from package
  `__init__.py` modules.
- Callers that use concrete providers directly should import from implementation
  modules, or use factories for runtime provider selection.

This still follows the Plugin/Factory encapsulation goal: the routing to
dynamically loaded optional dependencies stays in one internal orchestration
path, while package export behavior remains explicit and deterministic.

### 2. Keep the Core Clean
The core library orchestrator (e.g., a Middleware class or a Base Client) should know nothing about the specific third-party library (like Presidio, Pinecone, etc.). It should only interact with a standard Python `Protocol` or `ABC` that defines the required interface (e.g., `sanitize_text()`, `store_embedding()`).

### 3. Move the Heavy Lifting (The Plugin)
Create a separate, internal sub-module specifically for the concrete implementation of the optional dependency (e.g., `impl/_third_party_engine.py`). 
- This file **assumes the dependency is installed**.
- It contains **no** `try/except ImportError` blocks.
- This keeps static type checkers perfectly happy when evaluating that specific isolated file, provided the extra is installed in the check environment.

### 4. Dynamic Resolution (The Factory)
When the core orchestrator starts up and sees that the user has requested the feature (via configuration or environment variables), it calls a central factory function.
- The factory attempts to **dynamically import** the concrete implementation module using Python's `importlib`.
- **If the import succeeds:** It returns the fully instantiated engine/plugin.
- **If the import fails due to missing optional dependency extras:** The factory/loader converts the failure to one typed domain exception (`AiProviderDependencyUnavailableError`) with actionable install commands.
- **If the import fails for another reason:** The loader raises `AiProviderRuntimeError` so internal defects are not disguised as dependency misses.
- **No No-Op fallback is used in this design:** provider selection fails fast and explicitly so callers can remediate configuration immediately.

## Why This Pattern is Better

*   **Zero `# type: ignore` Required:** The core files only reference standard internal types and our custom Protocols. The implementation files reference the third-party types directly. Static analysis works perfectly.
*   **No Global Flags:** Variables like `DEPENDENCY_AVAILABLE` disappear completely from the global state.
*   **Clean Core Logic:** The core orchestrator's primary method just calls `self.engine.do_work()`. It doesn't need complex `if/else` branching. If the extra is missing, one typed exception is raised with direct install guidance.
