"""Tests for lazy AiProvider class loading behavior."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import patch

import pytest

from ai_api_unified.ai_provider_exceptions import (
    AiProviderDependencyUnavailableError,
    AiProviderRuntimeError,
)
from ai_api_unified.ai_provider_loader import load_ai_provider_class
from ai_api_unified.ai_provider_registry import AiProviderSpec


class DummyExpectedBase:
    """Simple base class used for loader subclass validation in tests."""


class DummyProviderClass(DummyExpectedBase):
    """Concrete provider class that satisfies DummyExpectedBase."""


class DummyWrongProviderClass:
    """Concrete provider class that does not satisfy DummyExpectedBase."""


class TestAiProviderLoader:
    """Validate dependency classification and class resolution in lazy loader."""

    @staticmethod
    def _build_ai_provider_spec(
        set_str_dependency_roots: set[str] | None = None,
        str_class_name: str = "DummyProviderClass",
    ) -> AiProviderSpec:
        """Builds a minimal AiProviderSpec fixture for loader unit tests."""
        return AiProviderSpec(
            str_capability="completions",
            str_engine="openai",
            str_module_path="tests.fake_provider_module",
            str_class_name=str_class_name,
            str_required_extra="openai",
            str_consumer_install_command="poetry add 'ai-api-unified[openai]'",
            str_local_install_command='poetry install --extras "openai"',
            set_str_dependency_roots=set_str_dependency_roots or {"openai"},
        )

    def test_load_ai_provider_class_returns_valid_class(self) -> None:
        """Loader should return resolved provider class when module and class are valid."""
        ai_provider_spec: AiProviderSpec = self._build_ai_provider_spec()
        module_loaded: ModuleType = ModuleType("tests.fake_provider_module")
        setattr(module_loaded, "DummyProviderClass", DummyProviderClass)

        with patch(
            "ai_api_unified.ai_provider_loader.importlib.import_module",
            return_value=module_loaded,
        ):
            class_provider: type[DummyExpectedBase] = load_ai_provider_class(
                ai_provider_spec,
                DummyExpectedBase,
            )

        assert class_provider is DummyProviderClass

    def test_load_ai_provider_class_raises_dependency_error_for_missing_extra(
        self,
    ) -> None:
        """Loader should raise dependency-unavailable error for missing optional SDK modules."""
        ai_provider_spec: AiProviderSpec = self._build_ai_provider_spec(
            set_str_dependency_roots={"openai"}
        )
        exception_dependency_missing: ModuleNotFoundError = ModuleNotFoundError(
            "No module named 'openai.resources'"
        )
        exception_dependency_missing.name = "openai.resources"

        with patch(
            "ai_api_unified.ai_provider_loader.importlib.import_module",
            side_effect=exception_dependency_missing,
        ):
            with pytest.raises(
                AiProviderDependencyUnavailableError,
                match="requires optional extra 'openai'",
            ):
                load_ai_provider_class(ai_provider_spec, DummyExpectedBase)

    def test_load_ai_provider_class_raises_runtime_error_for_internal_import_failure(
        self,
    ) -> None:
        """Loader should treat non-dependency missing modules as runtime import failures."""
        ai_provider_spec: AiProviderSpec = self._build_ai_provider_spec(
            set_str_dependency_roots={"openai"}
        )
        exception_internal_missing: ModuleNotFoundError = ModuleNotFoundError(
            "No module named 'local_internal_module'"
        )
        exception_internal_missing.name = "local_internal_module"

        with patch(
            "ai_api_unified.ai_provider_loader.importlib.import_module",
            side_effect=exception_internal_missing,
        ):
            with pytest.raises(
                AiProviderRuntimeError,
                match="internal import error",
            ):
                load_ai_provider_class(ai_provider_spec, DummyExpectedBase)

    def test_load_ai_provider_class_raises_runtime_error_when_class_missing(
        self,
    ) -> None:
        """Loader should raise runtime error when configured class symbol is absent."""
        ai_provider_spec: AiProviderSpec = self._build_ai_provider_spec(
            str_class_name="MissingProviderClass"
        )
        module_loaded: ModuleType = ModuleType("tests.fake_provider_module")

        with patch(
            "ai_api_unified.ai_provider_loader.importlib.import_module",
            return_value=module_loaded,
        ):
            with pytest.raises(
                AiProviderRuntimeError,
                match="class identifier was not found",
            ):
                load_ai_provider_class(ai_provider_spec, DummyExpectedBase)

    def test_load_ai_provider_class_raises_runtime_error_for_wrong_base(self) -> None:
        """Loader should raise runtime error when class does not implement expected base interface."""
        ai_provider_spec: AiProviderSpec = self._build_ai_provider_spec(
            str_class_name="DummyWrongProviderClass"
        )
        module_loaded: ModuleType = ModuleType("tests.fake_provider_module")
        setattr(module_loaded, "DummyWrongProviderClass", DummyWrongProviderClass)

        with patch(
            "ai_api_unified.ai_provider_loader.importlib.import_module",
            return_value=module_loaded,
        ):
            with pytest.raises(
                AiProviderRuntimeError,
                match="does not implement required base interface",
            ):
                load_ai_provider_class(ai_provider_spec, DummyExpectedBase)
