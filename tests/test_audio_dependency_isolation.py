# test_audio_dependency_isolation.py
"""
Guards the audio-dependency isolation shipped in 2.16.0: pydub lives in the
`voice` extra, and no text path (package import, completions clients) may
trigger its import or its SyntaxWarning/RuntimeWarning noise.

Import-isolation checks run in a subprocess so this test is immune to other
tests having already imported pydub into the shared pytest process.
"""

import subprocess
import sys

import pytest

from ai_api_unified.ai_provider_exceptions import (
    AiProviderDependencyUnavailableError,
)


def _run_isolated(str_code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", str_code],
        capture_output=True,
        text=True,
    )


class TestAudioDependencyIsolation:
    def test_package_import_stays_audio_free_and_warning_free(self):
        completed = _run_isolated(
            "import warnings; warnings.simplefilter('error')\n"
            "import sys\n"
            "import ai_api_unified\n"
            "assert 'pydub' not in sys.modules, 'pydub imported eagerly'\n"
            "assert 'audioop_lts' not in sys.modules, 'audioop shim imported eagerly'\n"
        )
        assert completed.returncode == 0, completed.stderr

    def test_claude_client_construction_stays_audio_free(self):
        pytest.importorskip("anthropic")
        completed = _run_isolated(
            "import warnings; warnings.simplefilter('error')\n"
            "import os, sys\n"
            "os.environ['ANTHROPIC_API_KEY'] = 'test-key'\n"
            "from ai_api_unified.completions.ai_anthropic_completions import (\n"
            "    AiAnthropicCompletions,\n"
            ")\n"
            "AiAnthropicCompletions(model='claude-opus-4-8')\n"
            "assert 'pydub' not in sys.modules, 'pydub imported by completions'\n"
        )
        assert completed.returncode == 0, completed.stderr

    def test_lazy_pydub_module_import_stays_audio_free(self):
        completed = _run_isolated(
            "import sys\n"
            "from ai_api_unified.util._lazy_pydub import AudioSegment, play\n"
            "assert 'pydub' not in sys.modules, 'proxy import loaded pydub'\n"
        )
        assert completed.returncode == 0, completed.stderr

    def test_audio_segment_proxy_resolves_real_pydub_on_use(self):
        pytest.importorskip("pydub")
        completed = _run_isolated(
            "import sys\n"
            "from ai_api_unified.util._lazy_pydub import AudioSegment\n"
            "segment = AudioSegment.silent(duration=10)\n"
            "assert type(segment).__module__.startswith('pydub'), type(segment)\n"
            "assert 'pydub' in sys.modules\n"
        )
        assert completed.returncode == 0, completed.stderr

    def test_missing_pydub_raises_voice_extra_hint(self):
        # sys.modules['pydub'] = None makes the import machinery raise
        # ImportError, simulating an install without the voice extra.
        completed = _run_isolated(
            "import sys\n"
            "sys.modules['pydub'] = None\n"
            "from ai_api_unified.util._lazy_pydub import AudioSegment\n"
            "from ai_api_unified.ai_provider_exceptions import (\n"
            "    AiProviderDependencyUnavailableError,\n"
            ")\n"
            "try:\n"
            "    AudioSegment.from_file('x')\n"
            "except AiProviderDependencyUnavailableError as exc:\n"
            "    assert 'voice' in str(exc), str(exc)\n"
            "else:\n"
            "    raise AssertionError('expected the typed dependency error')\n"
        )
        assert completed.returncode == 0, completed.stderr

    def test_typed_error_is_exported(self):
        # The hint error is part of the documented failure mode.
        assert issubclass(AiProviderDependencyUnavailableError, RuntimeError)
