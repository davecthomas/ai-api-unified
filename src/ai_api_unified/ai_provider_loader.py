"""
Lazy-loading helpers for optional provider implementations.

Prime directives for this refactor:
- Delay provider imports until runtime so optional dependencies are truly optional.
- Translate low-level import failures into stable, domain-specific exceptions.
- Validate provider classes against expected capability interfaces before use.

This module intentionally exposes one loading boundary:
- `load_ai_provider_class(...)`

Everything else in this module is private helper logic that supports that one
public function. This keeps dependency management and dynamic importing in one
place for all provider-capability factories.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TypeVar, cast

from .ai_provider_exceptions import (
    AiProviderDependencyUnavailableError,
    AiProviderRuntimeError,
)
from .ai_provider_registry import AiProviderSpec

TypeExpectedBase = TypeVar("TypeExpectedBase")


def _extract_missing_module_name(exception: ModuleNotFoundError) -> str:
    """
    Extracts a missing module token from a ModuleNotFoundError.

    Why:
        ModuleNotFoundError can be raised for several import paths, and the
        loader needs a stable string token to classify the failure.

    How:
        Reads the structured `exception.name` field instead of parsing message
        text, then normalizes whitespace.

    Args:
        exception: Captured ModuleNotFoundError from lazy import.

    Returns:
        The missing module name when available, otherwise an empty string.
    """
    str_missing_module_name: str = ""
    if exception.name:
        str_missing_module_name = str(exception.name).strip()
    # Normal return with missing module identifier (may be empty).
    return str_missing_module_name


def _is_dependency_missing_for_spec(
    str_missing_module_name: str, ai_provider_spec: AiProviderSpec
) -> bool:
    """
    Checks whether a missing module appears to be an optional dependency miss.

    Why:
        Not every ModuleNotFoundError means "install an extra". Some indicate an
        internal bug in provider code. This check separates user-remediable
        dependency misses from unexpected runtime failures.

    How:
        Compares the missing module token against dependency roots configured in
        registry metadata, including nested imports under each root.

    Args:
        str_missing_module_name: Missing module token from ModuleNotFoundError.
        ai_provider_spec: AiProvider metadata including dependency module roots.

    Returns:
        True when the missing module matches configured dependency roots.
    """
    if not str_missing_module_name:
        # Early return because empty module names cannot be matched safely.
        return False

    # Loop over configured dependency roots to find a direct or prefixed module match.
    for str_dependency_root in ai_provider_spec.set_str_dependency_roots:
        if (
            str_missing_module_name == str_dependency_root
            or str_missing_module_name.startswith(f"{str_dependency_root}.")
        ):
            # Early return because this miss maps to an optional dependency root.
            return True

    # Normal return because no dependency-root match was found.
    return False


def _build_dependency_message(
    ai_provider_spec: AiProviderSpec, str_missing_module_name: str
) -> str:
    """
    Builds a user-facing dependency installation error message.

    Why:
        When optional dependencies are missing, the fastest path to recovery is
        an explicit installation message tailored to the failing provider.

    Args:
        ai_provider_spec: AiProvider metadata with install command guidance.
        str_missing_module_name: Missing dependency module token.

    Returns:
        Formatted message describing the missing dependency and how to install.
    """
    str_missing_suffix: str = ""
    if str_missing_module_name:
        str_missing_suffix = f" Missing module: {str_missing_module_name}."
    str_error_message: str = (
        f"AiProvider '{ai_provider_spec.str_engine}' for capability "
        f"'{ai_provider_spec.str_capability}' requires optional extra "
        f"'{ai_provider_spec.str_required_extra}'.{str_missing_suffix} "
        f"Install for consumers: {ai_provider_spec.str_consumer_install_command}. "
        f"Install for local development: {ai_provider_spec.str_local_install_command}."
    )
    # Normal return with actionable dependency remediation guidance.
    return str_error_message


def load_ai_provider_class(
    ai_provider_spec: AiProviderSpec, class_expected_base: type[TypeExpectedBase]
) -> type[TypeExpectedBase]:
    """
    Lazily imports a provider module, resolves the provider class, and validates type.

    Why this is the one loader entry point:
        This function is the central safety gate for provider loading. It keeps
        all dependency classification, import behavior, and interface validation
        in one place so factories share one deterministic behavior.

    How this differs from non-factory imports:
        - Factory path: resolve provider metadata from the registry, call this
          function, then instantiate the returned class.
        - Direct module import path: callers import provider modules themselves
          and accept normal Python import behavior.
        This split keeps the library internals simple: one managed lazy-loader
        path for runtime provider selection, standard Python imports otherwise.

    When:
        Called by factory methods right before a provider is needed for a
        specific capability.

    How:
        1. Import module path from registry metadata.
        2. Classify ModuleNotFoundError as optional dependency miss vs runtime bug.
        3. Resolve class symbol by name from module namespace.
        4. Enforce class shape (`type`) and interface (`issubclass`).
        5. Return a precisely typed class for the caller.

    Args:
        ai_provider_spec: AiProvider metadata record from the registry.
        class_expected_base: Expected abstract/base class for this capability.

    Returns:
        The resolved provider class object constrained to class_expected_base type.
        Raises AiProviderDependencyUnavailableError when optional dependencies are
        missing, and AiProviderRuntimeError for all other loader failures.
    """
    module_loaded: ModuleType
    try:
        # Runtime import keeps optional providers decoupled from base installs.
        module_loaded = importlib.import_module(ai_provider_spec.str_module_path)
    except ModuleNotFoundError as exception:
        # ModuleNotFoundError can mean either a missing optional package or an
        # internal provider import bug. We classify before choosing exception type.
        str_missing_module_name: str = _extract_missing_module_name(exception)
        if _is_dependency_missing_for_spec(str_missing_module_name, ai_provider_spec):
            # Raise a user-actionable dependency exception with install guidance.
            raise AiProviderDependencyUnavailableError(
                _build_dependency_message(ai_provider_spec, str_missing_module_name)
            ) from exception
        # Raise runtime error for non-optional misses (usually provider defects).
        raise AiProviderRuntimeError(
            "Failed to load provider module due to internal import error: "
            f"{ai_provider_spec.str_module_path}"
        ) from exception
    except Exception as exception:
        # Normalize unexpected import-time failures into a loader runtime error.
        raise AiProviderRuntimeError(
            "Unexpected failure while importing provider module: "
            f"{ai_provider_spec.str_module_path}"
        ) from exception

    # Resolve the provider class by name from the imported module namespace.
    # Python fundamental: modules are objects with a namespace dictionary
    # (`module.__dict__`), and `vars(module_loaded)` exposes that dictionary.
    # This loader only knows the class name at runtime (from registry metadata),
    # so dictionary lookup is the correct dynamic approach for plugin loading.
    # `.get(...)` avoids a KeyError and lets validation below raise a typed error.
    object_provider_class: object = vars(module_loaded).get(
        ai_provider_spec.str_class_name
    )
    # Validate that lookup produced a class object, not a function/constant/None.
    if not isinstance(object_provider_class, type):
        raise AiProviderRuntimeError(
            "AiProvider class identifier was not found as a class object: "
            f"{ai_provider_spec.str_module_path}.{ai_provider_spec.str_class_name}"
        )

    # Validate interface compatibility so downstream code can rely on base methods.
    if not issubclass(object_provider_class, class_expected_base):
        raise AiProviderRuntimeError(
            "AiProvider class does not implement required base interface: "
            f"{ai_provider_spec.str_module_path}.{ai_provider_spec.str_class_name}"
        )

    class_provider_type: type[TypeExpectedBase] = cast(
        type[TypeExpectedBase], object_provider_class
    )
    # Normal return with validated provider class.
    return class_provider_type
