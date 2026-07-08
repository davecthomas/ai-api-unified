# ruff: noqa: E402

# test_anthropic_batches.py
"""
Tests for Anthropic Message Batches support on the `claude` completions engine.

Covers the capability gate, request validation, submit/status/cancel/results
normalization, the run_batch blocking wrapper, and PII redaction of batch
prompts, all against a mocked SDK client.
"""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("anthropic")

from ai_api_unified.ai_base import (
    AIBatchItemStatus,
    AIBatchJob,
    AIBatchRequestItem,
    AIBatchStatus,
)
from ai_api_unified.ai_provider_exceptions import AiProviderCapabilityUnsupportedError
from ai_api_unified.completions.ai_anthropic_completions import AiAnthropicCompletions


def _build_client(model: str = "claude-opus-4-8") -> AiAnthropicCompletions:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = AiAnthropicCompletions(model=model)
    client.client = Mock()
    return client


def _provider_batch(
    *,
    batch_id: str = "msgbatch_123",
    processing_status: str = "in_progress",
    processing: int = 2,
    succeeded: int = 0,
    errored: int = 0,
    canceled: int = 0,
    expired: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=batch_id,
        processing_status=processing_status,
        request_counts=SimpleNamespace(
            processing=processing,
            succeeded=succeeded,
            errored=errored,
            canceled=canceled,
            expired=expired,
        ),
        created_at=None,
        ended_at=None,
    )


def _reqs() -> list[AIBatchRequestItem]:
    return [
        AIBatchRequestItem(custom_id="a", prompt="Summarize the mat."),
        AIBatchRequestItem(
            custom_id="b", prompt="Translate hi.", max_response_tokens=64
        ),
    ]


class TestCapabilityGate:
    def test_anthropic_supports_batch(self) -> None:
        client = _build_client()
        assert client.capabilities.supports_batch is True

    def test_unsupported_engine_raises(self) -> None:
        # A client whose capabilities do not declare batch support must refuse.
        client = _build_client()
        with patch.object(
            type(client),
            "capabilities",
            property(
                lambda self: client._capabilities.model_copy(
                    update={"supports_batch": False}
                )
            ),
        ):
            with pytest.raises(AiProviderCapabilityUnsupportedError, match="batch"):
                client.submit_batch(_reqs())


class TestValidation:
    def test_empty_requests_raise(self) -> None:
        client = _build_client()
        with pytest.raises(ValueError, match="at least one"):
            client.submit_batch([])

    def test_duplicate_custom_ids_raise(self) -> None:
        client = _build_client()
        dupes = [
            AIBatchRequestItem(custom_id="x", prompt="one"),
            AIBatchRequestItem(custom_id="x", prompt="two"),
        ]
        with pytest.raises(ValueError, match="unique"):
            client.submit_batch(dupes)

    def test_empty_prompt_raises(self) -> None:
        client = _build_client()
        with pytest.raises(ValueError, match="empty prompt"):
            client.submit_batch([AIBatchRequestItem(custom_id="x", prompt="   ")])

    def test_empty_custom_id_raises(self) -> None:
        client = _build_client()
        with pytest.raises(ValueError, match="custom_id cannot be empty"):
            client.submit_batch([AIBatchRequestItem(custom_id="  ", prompt="hi")])

    def test_non_positive_max_response_tokens_raises(self) -> None:
        client = _build_client()
        with pytest.raises(ValueError, match="non-positive"):
            client.submit_batch(
                [AIBatchRequestItem(custom_id="x", prompt="hi", max_response_tokens=0)]
            )


class TestJobTerminalState:
    def test_canceling_is_not_terminal(self) -> None:
        job = AIBatchJob(
            batch_id="claude:b",
            provider_batch_id="b",
            status=AIBatchStatus.CANCELING,
        )
        assert job.is_terminal is False

    def test_in_progress_is_not_terminal(self) -> None:
        job = AIBatchJob(
            batch_id="claude:b", provider_batch_id="b", status=AIBatchStatus.IN_PROGRESS
        )
        assert job.is_terminal is False

    @pytest.mark.parametrize(
        "status",
        [
            AIBatchStatus.ENDED,
            AIBatchStatus.FAILED,
            AIBatchStatus.EXPIRED,
            AIBatchStatus.CANCELED,
        ],
    )
    def test_stopped_states_are_terminal(self, status: AIBatchStatus) -> None:
        job = AIBatchJob(batch_id="claude:b", provider_batch_id="b", status=status)
        assert job.is_terminal is True


class TestSubmit:
    def test_submit_builds_requests_and_normalizes_job(self) -> None:
        client = _build_client()
        client.client.messages.batches.create.return_value = _provider_batch()

        job: AIBatchJob = client.submit_batch(_reqs())

        assert isinstance(job, AIBatchJob)
        assert job.provider_batch_id == "msgbatch_123"
        assert job.batch_id == "claude:msgbatch_123"
        assert job.status is AIBatchStatus.IN_PROGRESS
        assert job.request_count == 2
        assert job.processing_count == 2
        assert job.provider_engine == "claude"

        create_kwargs = client.client.messages.batches.create.call_args.kwargs
        sent = create_kwargs["requests"]
        assert [r["custom_id"] for r in sent] == ["a", "b"]
        assert sent[0]["params"]["model"] == "claude-opus-4-8"
        assert sent[1]["params"]["max_tokens"] == 64
        assert sent[0]["params"]["messages"][0]["content"] == "Summarize the mat."

    def test_submit_redacts_prompts(self) -> None:
        client = _build_client()
        client.pii_middleware = SimpleNamespace(
            process_input=lambda text: "[REDACTED]",
            process_output=lambda text: text,
        )
        client.client.messages.batches.create.return_value = _provider_batch()

        client.submit_batch([AIBatchRequestItem(custom_id="a", prompt="My SSN is 1")])

        sent = client.client.messages.batches.create.call_args.kwargs["requests"]
        assert sent[0]["params"]["messages"][0]["content"] == "[REDACTED]"


class TestStatusAndCancel:
    def test_get_batch_strips_namespace(self) -> None:
        client = _build_client()
        client.client.messages.batches.retrieve.return_value = _provider_batch(
            processing_status="ended", processing=0, succeeded=2
        )

        job = client.get_batch("claude:msgbatch_123")

        client.client.messages.batches.retrieve.assert_called_once_with("msgbatch_123")
        assert job.status is AIBatchStatus.ENDED
        assert job.is_terminal is True
        assert job.succeeded_count == 2

    def test_cancel_batch(self) -> None:
        client = _build_client()
        client.client.messages.batches.cancel.return_value = _provider_batch(
            processing_status="canceling"
        )

        job = client.cancel_batch(
            AIBatchJob(
                batch_id="claude:msgbatch_123",
                provider_batch_id="msgbatch_123",
                status=AIBatchStatus.IN_PROGRESS,
            )
        )

        client.client.messages.batches.cancel.assert_called_once_with("msgbatch_123")
        assert job.status is AIBatchStatus.CANCELING


class TestResults:
    def test_results_normalize_success_and_error(self) -> None:
        client = _build_client()
        client.client.messages.batches.results.return_value = iter(
            [
                SimpleNamespace(
                    custom_id="a",
                    result=SimpleNamespace(
                        type="succeeded",
                        message=SimpleNamespace(
                            content=[SimpleNamespace(type="text", text="Hello")],
                            usage=SimpleNamespace(
                                input_tokens=10,
                                output_tokens=3,
                                cache_read_input_tokens=None,
                            ),
                        ),
                    ),
                ),
                SimpleNamespace(
                    custom_id="b",
                    result=SimpleNamespace(
                        type="errored",
                        error=SimpleNamespace(type="invalid_request"),
                    ),
                ),
            ]
        )

        results = client.get_batch_results("claude:msgbatch_123")

        assert len(results) == 2
        by_id = {r.custom_id: r for r in results}
        assert by_id["a"].status is AIBatchItemStatus.SUCCEEDED
        assert by_id["a"].text == "Hello"
        assert by_id["a"].provider_prompt_tokens == 10
        assert by_id["a"].provider_completion_tokens == 3
        assert by_id["b"].status is AIBatchItemStatus.ERRORED
        assert by_id["b"].error_message == "invalid_request"


class TestRunBatch:
    def test_run_batch_submits_polls_and_returns_results(self) -> None:
        client = _build_client()
        client.client.messages.batches.create.return_value = _provider_batch(
            processing_status="in_progress"
        )
        # First poll still in progress, second poll ended.
        client.client.messages.batches.retrieve.side_effect = [
            _provider_batch(processing_status="in_progress"),
            _provider_batch(processing_status="ended", processing=0, succeeded=1),
        ]
        client.client.messages.batches.results.return_value = iter(
            [
                SimpleNamespace(
                    custom_id="a",
                    result=SimpleNamespace(
                        type="succeeded",
                        message=SimpleNamespace(
                            content=[SimpleNamespace(type="text", text="done")],
                            usage=SimpleNamespace(
                                input_tokens=5,
                                output_tokens=1,
                                cache_read_input_tokens=None,
                            ),
                        ),
                    ),
                )
            ]
        )

        with patch("ai_api_unified.ai_base.time.sleep", return_value=None):
            results = client.run_batch(
                [AIBatchRequestItem(custom_id="a", prompt="go")],
                poll_interval_seconds=0.01,
            )

        assert [r.custom_id for r in results] == ["a"]
        assert results[0].text == "done"
        assert client.client.messages.batches.retrieve.call_count == 2

    def test_run_batch_times_out(self) -> None:
        client = _build_client()
        client.client.messages.batches.create.return_value = _provider_batch(
            processing_status="in_progress"
        )
        client.client.messages.batches.retrieve.return_value = _provider_batch(
            processing_status="in_progress"
        )

        with patch("ai_api_unified.ai_base.time.sleep", return_value=None):
            with pytest.raises(TimeoutError, match="did not end"):
                client.run_batch(
                    [AIBatchRequestItem(custom_id="a", prompt="go")],
                    timeout_seconds=0.0,
                    poll_interval_seconds=0.01,
                )
