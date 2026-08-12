# ruff: noqa: E402
"""Cross-provider guards for voice observability caller-id attribution."""

from __future__ import annotations

import ast
import inspect
import textwrap
from abc import ABC
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

pytest.importorskip("elevenlabs")
pytest.importorskip("openai")
pytest.importorskip("azure.cognitiveservices.speech")
pytest.importorskip("google.genai")
pytest.importorskip("google.cloud.texttospeech")
pytest.importorskip("google.cloud.speech_v1p1beta1")

import ai_api_unified.ai_openai_base as ai_openai_base_module
import ai_api_unified.voice.ai_voice_openai as ai_voice_openai_module
from ai_api_unified.ai_base import RETRY_POLICY_DEFAULT
from ai_api_unified.ai_openai_base import OPENAI_USER_SETTING_KEY, RETRY_POLICY_KEY
from ai_api_unified.voice.ai_voice_azure import AIVoiceAzure
from ai_api_unified.voice.ai_voice_base import AIVoiceBase
from ai_api_unified.voice.ai_voice_elevenlabs import AIVoiceElevenLabs
from ai_api_unified.voice.ai_voice_google import AIVoiceGoogle
from ai_api_unified.voice.ai_voice_openai import AIVoiceOpenAI

TEST_OPENAI_USER: str = "voice-attribution-user"

LIST_VOICE_PROVIDER_CLASSES: list[type[AIVoiceBase]] = [
    AIVoiceOpenAI,
    AIVoiceGoogle,
    AIVoiceAzure,
    AIVoiceElevenLabs,
]
FROZENSET_MRO_BASES_EXEMPT_FROM_INIT_CHECK: frozenset[type] = frozenset(
    {AIVoiceBase, BaseModel, ABC, object}
)


def _collect_self_assigned_attribute_names(callable_initializer: Any) -> set[str]:
    """
    Collects every `self.<name> = ...` target assigned inside one initializer.

    Args:
        callable_initializer: Initializer function to inspect.

    Returns:
        Set of attribute names the initializer assigns onto `self`.
    """
    module_tree: ast.Module = ast.parse(
        textwrap.dedent(inspect.getsource(callable_initializer))
    )
    set_attribute_names: set[str] = set()
    for node in ast.walk(module_tree):
        list_targets: list[ast.expr]
        if isinstance(node, ast.Assign):
            list_targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            list_targets = [node.target]
        else:
            continue
        for node_target in list_targets:
            if (
                isinstance(node_target, ast.Attribute)
                and isinstance(node_target.value, ast.Name)
                and node_target.value.id == "self"
            ):
                set_attribute_names.add(node_target.attr)
    # Normal return with every attribute name the initializer assigns onto self.
    return set_attribute_names


def _build_openai_voice_client() -> AIVoiceOpenAI:
    """
    Builds an AIVoiceOpenAI through its real constructor with the SDK stubbed out.

    Args:
        None

    Returns:
        Fully constructed AIVoiceOpenAI backed by fake environment and SDK objects.
    """
    mock_env_settings: Mock = Mock()
    mock_env_settings.get_setting.side_effect = lambda key, default=None: {
        "OPENAI_API_KEY": "test-openai-api-key",
        OPENAI_USER_SETTING_KEY: TEST_OPENAI_USER,
    }.get(key, default)

    # AIOpenAIBase.get_api_base_url builds its own EnvSettings from its own module,
    # so patching only the voice module would let this read the ambient env and .env.
    with (
        patch.object(
            ai_voice_openai_module, "EnvSettings", return_value=mock_env_settings
        ),
        patch.object(
            ai_openai_base_module, "EnvSettings", return_value=mock_env_settings
        ),
        patch.object(ai_voice_openai_module, "OpenAI", return_value=SimpleNamespace()),
    ):
        # Normal return with a constructor-initialized OpenAI voice client.
        return AIVoiceOpenAI(engine="openai")


@pytest.mark.parametrize(
    "class_voice_provider",
    LIST_VOICE_PROVIDER_CLASSES,
    ids=lambda class_voice_provider: class_voice_provider.__name__,
)
def test_skipped_vendor_initializer_state_is_reestablished(
    class_voice_provider: type[AIVoiceBase],
) -> None:
    """
    Verify state from a skipped vendor initializer is re-established by the provider.

    Voice providers are pydantic models. `BaseModel.__init__` does not chain to
    `super().__init__()`, so a vendor base listed after `AIVoiceBase` has its
    `__init__` skipped and every attribute it would assign stays unset. Reading
    such an attribute raises AttributeError on the first provider call, which is
    exactly how `legacy_caller_id=self.user` broke every OpenAI TTS call.

    Args:
        class_voice_provider: Concrete voice provider class under inspection.

    Returns:
        None after asserting no skipped initializer leaves an attribute unset.
    """
    dict_skipped_initializer_attributes: dict[str, set[str]] = {}
    for class_base in class_voice_provider.__mro__[1:]:
        if class_base in FROZENSET_MRO_BASES_EXEMPT_FROM_INIT_CHECK:
            continue
        if "__init__" not in vars(class_base):
            continue
        dict_skipped_initializer_attributes[class_base.__name__] = (
            _collect_self_assigned_attribute_names(class_base.__init__)
        )

    if not dict_skipped_initializer_attributes:
        pytest.skip(
            f"{class_voice_provider.__name__} has no skipped vendor initializer"
        )

    assert class_voice_provider is AIVoiceOpenAI, (
        f"{class_voice_provider.__name__} gained a vendor base with a skipped "
        "__init__. Add a constructor-based builder for it here so this guard can "
        "check the attributes that initializer would have assigned."
    )
    ai_voice_client: AIVoiceBase = _build_openai_voice_client()
    for (
        str_base_name,
        set_attribute_names,
    ) in dict_skipped_initializer_attributes.items():
        for str_attribute_name in sorted(set_attribute_names):
            assert hasattr(ai_voice_client, str_attribute_name), (
                f"{str_base_name}.__init__ assigns '{str_attribute_name}', but it sits "
                f"behind pydantic's BaseModel in {class_voice_provider.__name__}'s MRO "
                "and never runs. Re-establish that attribute in "
                f"{class_voice_provider.__name__}.__init__ or the first inherited "
                "method that reads it raises AttributeError."
            )


@pytest.mark.parametrize(
    "class_voice_provider",
    LIST_VOICE_PROVIDER_CLASSES,
    ids=lambda class_voice_provider: class_voice_provider.__name__,
)
def test_provider_resolves_legacy_caller_id_without_constructed_state(
    class_voice_provider: type[AIVoiceBase],
) -> None:
    """
    Verify every provider resolves a legacy caller id without touching unset attributes.

    A provider that reads vendor state directly at the observability call site
    raises AttributeError here, before any network call is attempted.

    Args:
        class_voice_provider: Concrete voice provider class under inspection.

    Returns:
        None after asserting the hook returns a string or None on an unconstructed instance.
    """
    ai_voice_client: AIVoiceBase = class_voice_provider.model_construct()

    legacy_caller_id: str | None = ai_voice_client._resolve_legacy_caller_id()

    assert legacy_caller_id is None or isinstance(legacy_caller_id, str)


def test_base_resolver_defaults_to_no_legacy_caller_id() -> None:
    """
    Verify AIVoiceBase attributes nothing by default so vendors without the concept opt out.

    Args:
        None

    Returns:
        None after asserting the base resolver returns None.
    """

    class ConcreteVoiceClient(AIVoiceBase):
        """Minimal concrete AIVoiceBase used to exercise the default resolver."""

        def text_to_voice(self, **_: Any) -> bytes:
            """Unused abstract override."""
            return b""

        def stream_audio(self, *_: Any, **__: Any) -> bytes:
            """Unused abstract override."""
            return b""

        def speech_to_text(self, *_: Any, **__: Any) -> str:
            """Unused abstract override."""
            return ""

    assert ConcreteVoiceClient.model_construct()._resolve_legacy_caller_id() is None


@pytest.mark.parametrize(
    "object_configured_retry_policy",
    ["", "   ", None],
    ids=["blank", "whitespace", "unset"],
)
def test_blank_retry_policy_setting_falls_back_to_default(
    object_configured_retry_policy: object,
) -> None:
    """
    Verify a present-but-blank COMPLETIONS_RETRY_POLICY does not break construction.

    `EnvSettings.get_setting` returns the empty string rather than the default
    when a key is present and blank, and the empty string is not a valid policy.

    Args:
        object_configured_retry_policy: Blank-ish configured retry policy value.

    Returns:
        None after asserting the client builds and reports the default policy.
    """
    mock_env_settings: Mock = Mock()
    mock_env_settings.get_setting.side_effect = lambda key, default=None: {
        "OPENAI_API_KEY": "test-openai-api-key",
        OPENAI_USER_SETTING_KEY: TEST_OPENAI_USER,
        RETRY_POLICY_KEY: object_configured_retry_policy,
    }.get(key, default)

    with (
        patch.object(
            ai_voice_openai_module, "EnvSettings", return_value=mock_env_settings
        ),
        patch.object(
            ai_openai_base_module, "EnvSettings", return_value=mock_env_settings
        ),
        patch.object(ai_voice_openai_module, "OpenAI", return_value=SimpleNamespace()),
    ):
        ai_voice_client: AIVoiceOpenAI = AIVoiceOpenAI(engine="openai")

    assert ai_voice_client.retry_policy == RETRY_POLICY_DEFAULT


def test_vendor_attributes_stay_writable_like_sibling_capabilities() -> None:
    """
    Verify the mirrored AIOpenAIBase attributes accept assignment.

    Existing suites reassign these on completions, embeddings, and images
    clients, so voice keeps the same writable surface.

    Args:
        None

    Returns:
        None after asserting each mirrored attribute round-trips a new value.
    """
    ai_voice_client: AIVoiceOpenAI = _build_openai_voice_client()

    ai_voice_client.backoff_delays = [0]
    ai_voice_client.api_key = "rotated-key"
    ai_voice_client.user = "rotated-user"
    ai_voice_client.retry_policy = RETRY_POLICY_DEFAULT
    ai_voice_client.base_url = "https://example.invalid/v1"

    assert ai_voice_client.backoff_delays == [0]
    assert ai_voice_client.api_key == "rotated-key"
    assert ai_voice_client.user == "rotated-user"
    assert ai_voice_client._resolve_legacy_caller_id() == "rotated-user"
    assert ai_voice_client.base_url == "https://example.invalid/v1"


def test_openai_provider_resolves_configured_user_as_caller_id() -> None:
    """
    Verify the OpenAI provider attributes calls to the configured OPENAI_USER value.

    Args:
        None

    Returns:
        None after asserting the constructed client reports the configured user.
    """
    ai_voice_client: AIVoiceOpenAI = _build_openai_voice_client()

    assert ai_voice_client.user == TEST_OPENAI_USER
    assert ai_voice_client._resolve_legacy_caller_id() == TEST_OPENAI_USER


@pytest.mark.parametrize(
    "class_voice_provider",
    [AIVoiceGoogle, AIVoiceAzure, AIVoiceElevenLabs],
    ids=lambda class_voice_provider: class_voice_provider.__name__,
)
def test_providers_without_vendor_user_setting_inherit_base_resolver(
    class_voice_provider: type[AIVoiceBase],
) -> None:
    """
    Verify providers whose vendor exposes no user setting do not override the resolver.

    Args:
        class_voice_provider: Concrete voice provider class under inspection.

    Returns:
        None after asserting the provider inherits the base resolver unchanged.
    """
    assert "_resolve_legacy_caller_id" not in vars(class_voice_provider)
