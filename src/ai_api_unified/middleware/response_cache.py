# response_cache.py
"""
Response cache: a stored answer for an identical request.

Off unless `models.json` (or the middleware YAML) lists the `response_cache`
middleware. Once on, every public call that can be cached looks in the
store first: a hit returns the stored result without touching the router,
its classifier, or the provider; a miss runs the call as usual and stores
the result for the operation's TTL.

Where it runs: once per public call, at the outermost client. `AIBase`
wraps the public methods its subclasses define (`send_prompt`,
`generate_embeddings`, ...) through `wrap_cacheable_methods`, the envelope
decorators cover conversations and structured output, and the classifier
templates are decorated directly. A call made from inside another (a
chain's candidate, a router's tier, the router's classifier) sees
`RESPONSE_CACHE_ACTIVE` set and runs through, so nothing is cached twice.

The key: `rc:v1:<client>:<sha256 of the canonical request>`. The client
part is the engine token, model, thinking level, and config_id, so two
clients configured differently never share an entry. The canonical request
is everything that determines the answer: the operation, the prompt or
messages (strings stripped of surrounding whitespace), tools, schema,
token budget, and provider options. Not in it: request ids, timestamps,
prompt-cache hints, the purpose hint, and the caller, session, or tenant
tags, which are labels for reporting, never a boundary.

Per call: `provider_options={"cache": ...}` on conversations and structured
output, `other_params.cache` on prompts, or the `cache_mode(...)` context
manager around any call. "on" (the default) reads and writes; "off" neither
reads nor writes; "bypass" skips the lookup and refreshes the entry.

Never cached: streaming, calls with media attachments, batches, voice.

A hit is metered: one request cost report with `cache_hit` True, a total of
zero, and `usd_saved` set to what the original call cost, emitted on the
cost topic and kept as the client's `last_request_cost`. Result envelopes
carry the same two fields.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from .cache_store import store_get, store_set
from .middleware_config import (
    CACHE_OPERATION_COMPLETIONS,
    CACHE_OPERATION_CONVERSATION,
    CACHE_OPERATION_EMBEDDINGS,
    CACHE_OPERATION_STRUCTURED_OUTPUT,
    MiddlewareConfig,
    ResponseCacheSettingsModel,
)
from .request_cost import (
    REQUEST_STATUS_OK,
    AIRequestCostReport,
    RequestCallCapture,
    capture_request_calls,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

KEY_PREFIX: str = "rc:v1"
# Values of the per-call cache option.
CACHE_MODE_ON: str = "on"
CACHE_MODE_OFF: str = "off"
CACHE_MODE_BYPASS: str = "bypass"
FROZENSET_CACHE_MODES: frozenset[str] = frozenset(
    {CACHE_MODE_ON, CACHE_MODE_OFF, CACHE_MODE_BYPASS}
)
# Reserved provider_options key and other_params field.
CACHE_OPTION: str = "cache"
# Values of the `ai_cache_event` field on this module's log records.
CACHE_EVENT_HIT: str = "hit"
CACHE_EVENT_MISS: str = "miss"
CACHE_EVENT_STORE_FAILED: str = "store_failed"

# Argument names that never enter the key.
FROZENSET_EXCLUDED_ARGS: frozenset[str] = frozenset(
    {"request_timeout_seconds", "prompt_cache", "self"}
)
# provider_options keys that never enter the key.
FROZENSET_EXCLUDED_OPTIONS: frozenset[str] = frozenset({CACHE_OPTION, "purpose"})

# Public methods AIBase wraps on its subclasses, by operation.
DICT_WRAPPED_METHOD_OPERATIONS: dict[str, str] = {
    "send_prompt": CACHE_OPERATION_COMPLETIONS,
    "asend_prompt": CACHE_OPERATION_COMPLETIONS,
    "send_prompt_with_result": CACHE_OPERATION_COMPLETIONS,
    "asend_prompt_with_result": CACHE_OPERATION_COMPLETIONS,
    "generate_embeddings": CACHE_OPERATION_EMBEDDINGS,
    "generate_embeddings_batch": CACHE_OPERATION_EMBEDDINGS,
    "agenerate_embeddings": CACHE_OPERATION_EMBEDDINGS,
    "agenerate_embeddings_batch": CACHE_OPERATION_EMBEDDINGS,
}
DICT_ENVELOPE_OPERATIONS: dict[str, str] = {
    "send_conversation": CACHE_OPERATION_CONVERSATION,
    "asend_conversation": CACHE_OPERATION_CONVERSATION,
    "send_structured_output": CACHE_OPERATION_STRUCTURED_OUTPUT,
    "asend_structured_output": CACHE_OPERATION_STRUCTURED_OUTPUT,
}
_WRAPPED_MARKER: str = "_ai_response_cache_wrapped"

# True while an outer public call owns the cache decision.
RESPONSE_CACHE_ACTIVE: ContextVar[bool] = ContextVar(
    "ai_api_unified_response_cache_active", default=False
)
# The cache_mode() context manager's value, or None.
CACHE_MODE_OVERRIDE: ContextVar[str | None] = ContextVar(
    "ai_api_unified_cache_mode", default=None
)


# ── Settings singleton ────────────────────────────────────────────────────


_LOCK: threading.Lock = threading.Lock()
_SETTINGS: ResponseCacheSettingsModel | None = None
_BOOL_RESOLVED: bool = False


def get_response_cache_settings() -> ResponseCacheSettingsModel | None:
    """The configured settings, resolved once per process; None when off."""
    global _SETTINGS, _BOOL_RESOLVED
    if _BOOL_RESOLVED:
        # Early return with the resolved value.
        return _SETTINGS
    with _LOCK:
        if not _BOOL_RESOLVED:
            try:
                _SETTINGS = MiddlewareConfig().get_response_cache_settings()
            except Exception as exception:
                _LOGGER.warning(
                    "response_cache_config_failed error_type=%s",
                    exception.__class__.__name__,
                )
                _SETTINGS = None
            _BOOL_RESOLVED = True
    # Normal return with the settings, or None.
    return _SETTINGS


def reset_response_cache() -> None:
    """Forgets the resolved settings so the next call re-reads configuration."""
    global _SETTINGS, _BOOL_RESOLVED
    with _LOCK:
        _SETTINGS = None
        _BOOL_RESOLVED = False


def parse_cache_mode(value: Any) -> str | None:
    """
    Normalizes a per-call cache option.

    Raises:
        ValueError: Not on, off, or bypass.
    """
    if value is None:
        # Early return: not set.
        return None
    str_mode: str = str(value).strip().lower()
    if str_mode not in FROZENSET_CACHE_MODES:
        raise ValueError(
            f"Unknown cache option {value!r}; expected one of "
            f"{sorted(FROZENSET_CACHE_MODES)}."
        )
    # Normal return with the normalized mode.
    return str_mode


@contextmanager
def cache_mode(str_mode: str) -> Iterator[None]:
    """
    Sets the cache option for every call inside the block: "on", "off", or
    "bypass". The per-call option, when given, wins over this.
    """
    token: Token[str | None] = CACHE_MODE_OVERRIDE.set(parse_cache_mode(str_mode))
    try:
        yield
    finally:
        CACHE_MODE_OVERRIDE.reset(token)


# ── Canonical request and key ─────────────────────────────────────────────


def _canonical(value: Any) -> Any:
    """JSON-ready form of one argument: models dump, schemas by value, strings stripped."""
    if isinstance(value, str):
        # Early return with the stripped string.
        return value.strip()
    if isinstance(value, BaseModel):
        # Early return with the model's JSON form.
        return _canonical(
            value.model_dump(mode="json", exclude={"prompt_cache", CACHE_OPTION})
        )
    if isinstance(value, type) and issubclass(value, BaseModel):
        # Early return with the schema a response model supplies.
        return value.model_json_schema()
    if isinstance(value, dict):
        # Early return with keys sorted and values canonicalized.
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        # Early return with each element canonicalized.
        return [_canonical(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        # Early return: already JSON.
        return value
    if isinstance(value, bytes):
        # Early return with a digest, never the bytes.
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Decimal):
        # Early return with the number as text.
        return str(value)
    # Normal return with the text form of anything else.
    return repr(value)


def _has_media(dict_args: dict[str, Any]) -> bool:
    """True when any argument carries media attachments."""
    # Loop over the arguments for a params object with media.
    for value in dict_args.values():
        checker: Any = getattr(value, "has_included_media", None)
        if callable(checker) and checker():
            # Early return: an attachment is present.
            return True
    # Normal return: text only.
    return False


def _client_identity(client: Any) -> dict[str, Any]:
    """The parts of the client that change the answer to the same request."""
    thinking: Any = getattr(client, "_thinking_level", None)
    if thinking is None:
        try:
            thinking = getattr(client, "thinking_level", None)
        except Exception:
            thinking = None
    # Normal return with the identity fields.
    return {
        "type": type(client).__name__,
        "engine": _safe_call(client, "_resolve_observability_provider_engine"),
        "model": _safe_attr(client, "model_name"),
        "thinking": getattr(thinking, "value", thinking) if thinking else None,
        "dimensions": getattr(client, "dimensions", None),
        "config_id": _safe_attr(client, "config_id"),
    }


def _safe_attr(obj: Any, str_name: str) -> Any:
    try:
        value: Any = getattr(obj, str_name, None)
        # Normal return with the value when it is plain data.
        return value if isinstance(value, (str, int, float)) else None
    except Exception:
        # Early return: a property that raises on a partial client.
        return None


def _safe_call(obj: Any, str_name: str) -> Any:
    try:
        method: Any = getattr(obj, str_name, None)
        # Normal return with the call's result.
        return method() if callable(method) else None
    except Exception:
        # Early return: a partial client.
        return None


def _sync_method_name(str_method: str) -> str:
    """The sync name of a method, so an async twin shares its entries."""
    if str_method.startswith("a") and str_method[1:] in (
        *DICT_WRAPPED_METHOD_OPERATIONS,
        *DICT_ENVELOPE_OPERATIONS,
        "classify_choice",
        "classify_predicate",
        "classify_score",
    ):
        # Early return with the twin's name.
        return str_method[1:]
    # Normal return: already the sync name.
    return str_method


def build_cache_key(
    client: Any, str_operation: str, str_method: str, dict_args: dict[str, Any]
) -> str:
    """The store key for one call; see the module docstring."""
    dict_options: Any = dict_args.get("provider_options")
    if isinstance(dict_options, dict):
        dict_args = {
            **dict_args,
            "provider_options": {
                k: v
                for k, v in dict_options.items()
                if k not in FROZENSET_EXCLUDED_OPTIONS
            },
        }
    dict_payload: dict[str, Any] = {
        "operation": str_operation,
        "method": _sync_method_name(str_method),
        "client": _client_identity(client),
        "args": _canonical(
            {k: v for k, v in dict_args.items() if k not in FROZENSET_EXCLUDED_ARGS}
        ),
    }
    str_json: str = json.dumps(dict_payload, sort_keys=True, separators=(",", ":"))
    str_digest: str = hashlib.sha256(str_json.encode("utf-8")).hexdigest()
    str_client: str = str(dict_payload["client"].get("config_id") or "direct")
    # Normal return with the namespaced key.
    return f"{KEY_PREFIX}:{str_client}:{str_digest}"


# ── Stored values ─────────────────────────────────────────────────────────


def _serialize_result(result: Any, usd: Decimal | None) -> bytes | None:
    """The stored form of a result, or None when it cannot be stored."""
    dict_entry: dict[str, Any]
    if isinstance(result, BaseModel):
        dict_entry = {
            "kind": "model",
            "type": f"{type(result).__module__}.{type(result).__qualname__}",
            "payload": result.model_dump(mode="json"),
        }
    elif isinstance(result, (str, dict, list)):
        dict_entry = {"kind": "json", "payload": result}
    else:
        # Early return: an unknown shape is not stored.
        return None
    dict_entry["usd"] = str(usd) if usd is not None else None
    try:
        # Normal return with the JSON bytes.
        return json.dumps(dict_entry, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        # Early return: a provider object in the result cannot be stored.
        return None


def _import_type(str_path: str) -> type[BaseModel] | None:
    str_module, _, str_qualname = str_path.rpartition(".")
    try:
        import importlib

        obj: Any = importlib.import_module(str_module)
        # Loop over the qualified name's parts (nested classes).
        for str_part in str_qualname.split("."):
            obj = getattr(obj, str_part)
    except Exception:
        # Early return: the type is gone.
        return None
    # Normal return with the class.
    return obj if isinstance(obj, type) and issubclass(obj, BaseModel) else None


def _deserialize_result(value: bytes) -> tuple[Any, Decimal | None] | None:
    """(result, original cost) from the stored bytes, or None when unreadable."""
    try:
        dict_entry: Any = json.loads(value.decode("utf-8"))
        usd: Decimal | None = (
            Decimal(dict_entry["usd"]) if dict_entry.get("usd") is not None else None
        )
        if dict_entry.get("kind") == "json":
            # Early return with the plain value.
            return dict_entry["payload"], usd
        cls: type[BaseModel] | None = _import_type(str(dict_entry.get("type")))
        if cls is None:
            # Early return: cannot rebuild the model.
            return None
        # Normal return with the rebuilt model.
        return cls.model_validate(dict_entry["payload"]), usd
    except Exception:
        # Early return: a stale or foreign entry is a miss.
        return None


# ── Hits ──────────────────────────────────────────────────────────────────


def _hit_report(
    client: Any, str_operation: str, usd_saved: Decimal | None
) -> AIRequestCostReport:
    """The request cost report for a hit: nothing spent, the original saved."""
    from .observability_runtime import get_observability_context

    context: Any = get_observability_context()
    dict_tags: dict[str, str] = {
        **dict(getattr(client, "_dict_config_tags", {}) or {}),
        **dict(context.tags),
    }
    # Normal return with the zero-cost report.
    return AIRequestCostReport(
        request_id=uuid.uuid4().hex,
        operation=str_operation,
        status=REQUEST_STATUS_OK,
        started_at_utc=datetime.now(timezone.utc),
        elapsed_ms=0.0,
        caller_id=context.caller_id,
        session_id=context.session_id,
        workflow_id=context.workflow_id,
        tags=dict_tags,
        cache_hit=True,
        usd_saved=usd_saved,
    )


def _emit_hit(
    client: Any, str_operation: str, str_key: str, usd_saved: Decimal | None
) -> AIRequestCostReport:
    from .observability import get_observability_middleware

    report: AIRequestCostReport = _hit_report(client, str_operation, usd_saved)
    try:
        get_observability_middleware().emit_request_cost(report)
    except Exception as exception:
        _LOGGER.warning(
            "response_cache_emit_failed error_type=%s", exception.__class__.__name__
        )
    try:
        client._last_request_cost = report
    except Exception:
        # A test double or a frozen client has nowhere to keep it.
        pass
    _LOGGER.info(
        "CACHE HIT: %s served from the response cache (saved %s USD).",
        str_operation,
        usd_saved if usd_saved is not None else "unknown",
        extra={
            "ai_cache_event": CACHE_EVENT_HIT,
            "operation": str_operation,
            "cache_key": str_key.rsplit(":", 1)[-1][:16],
            "usd_saved": str(usd_saved) if usd_saved is not None else None,
            "caller_id": report.caller_id,
            "config_id": report.tags.get("config_id"),
        },
    )
    # Normal return with the hit's report.
    return report


def _mark_hit(result: Any, report: AIRequestCostReport) -> Any:
    """Stamps cache_hit, usd_saved, and the zero-cost report on an envelope."""
    if isinstance(result, BaseModel) and "cache_hit" in type(result).model_fields:
        dict_update: dict[str, Any] = {"cache_hit": True, "usd_saved": report.usd_saved}
        if "cost" in type(result).model_fields:
            dict_update["cost"] = report
        # Early return with the stamped copy.
        return result.model_copy(update=dict_update)
    # Normal return: a plain value carries no envelope.
    return result


# ── The decision ──────────────────────────────────────────────────────────


class _Plan:
    """What one call does with the cache."""

    __slots__ = ("str_key", "bool_read", "bool_write", "int_ttl", "str_operation")

    def __init__(
        self,
        *,
        str_key: str,
        bool_read: bool,
        bool_write: bool,
        int_ttl: int,
        str_operation: str,
    ) -> None:
        self.str_key = str_key
        self.bool_read = bool_read
        self.bool_write = bool_write
        self.int_ttl = int_ttl
        self.str_operation = str_operation


def _resolve_mode(dict_args: dict[str, Any]) -> str:
    """The per-call option, the context manager's, or on."""
    dict_options: Any = dict_args.get("provider_options")
    if isinstance(dict_options, dict) and dict_options.get(CACHE_OPTION) is not None:
        # Early return with the provider_options value.
        return parse_cache_mode(dict_options[CACHE_OPTION]) or CACHE_MODE_ON
    params: Any = dict_args.get("other_params")
    str_params_mode: Any = (
        getattr(params, CACHE_OPTION, None) if params is not None else None
    )
    if str_params_mode is not None:
        # Early return with the params value.
        return parse_cache_mode(str_params_mode) or CACHE_MODE_ON
    # Normal return with the block's mode, or on.
    return CACHE_MODE_OVERRIDE.get() or CACHE_MODE_ON


def plan_for(
    client: Any, str_operation: str, str_method: str, dict_args: dict[str, Any]
) -> _Plan | None:
    """The cache plan for one call, or None when the call is not cached."""
    settings: ResponseCacheSettingsModel | None = get_response_cache_settings()
    if settings is None:
        # Early return: the cache is off.
        return None
    int_ttl: int | None = settings.ttl_for(str_operation)
    if int_ttl is None:
        # Early return: this operation is off.
        return None
    str_mode: str = _resolve_mode(dict_args)
    if str_mode == CACHE_MODE_OFF or _has_media(dict_args):
        # Early return: this call opts out, or carries media.
        return None
    try:
        str_key: str = build_cache_key(client, str_operation, str_method, dict_args)
    except Exception as exception:
        _LOGGER.debug(
            "response_cache_key_failed operation=%s error_type=%s",
            str_operation,
            exception.__class__.__name__,
        )
        # Early return: an unkeyable request is not cached.
        return None
    # Normal return with the plan.
    return _Plan(
        str_key=str_key,
        bool_read=str_mode == CACHE_MODE_ON,
        bool_write=True,
        int_ttl=int_ttl,
        str_operation=str_operation,
    )


def lookup(client: Any, plan: _Plan) -> tuple[bool, Any]:
    """(hit, result) for a plan that reads."""
    if not plan.bool_read:
        # Early return: bypass never reads.
        return False, None
    value: bytes | None = store_get(plan.str_key)
    if value is None:
        # Early return: a miss.
        return False, None
    loaded: tuple[Any, Decimal | None] | None = _deserialize_result(value)
    if loaded is None:
        # Early return: unreadable, so a miss.
        return False, None
    result, usd_saved = loaded
    report: AIRequestCostReport = _emit_hit(
        client, plan.str_operation, plan.str_key, usd_saved
    )
    # Normal return with the stored result, marked as a hit.
    return True, _mark_hit(result, report)


def store(plan: _Plan, result: Any, capture: RequestCallCapture) -> None:
    """Stores a miss's result with the cost the call incurred."""
    capture.mark_served()
    report: AIRequestCostReport = capture.build_report(REQUEST_STATUS_OK)
    usd: Decimal | None = report.total_usd if report.calls else None
    value: bytes | None = _serialize_result(result, usd)
    if value is None:
        _LOGGER.debug(
            "response_cache_not_stored operation=%s reason=unserializable",
            plan.str_operation,
            extra={"ai_cache_event": CACHE_EVENT_STORE_FAILED},
        )
        # Early return: nothing stored.
        return None
    store_set(plan.str_key, value, plan.int_ttl)
    _LOGGER.debug(
        "CACHE MISS: %s stored for %ss.",
        plan.str_operation,
        plan.int_ttl,
        extra={"ai_cache_event": CACHE_EVENT_MISS, "operation": plan.str_operation},
    )
    # Normal return after storing.
    return None


def _bound_args(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """The call's arguments by name, defaults applied, self excluded."""
    try:
        bound: inspect.BoundArguments = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        dict_args: dict[str, Any] = dict(bound.arguments)
    except (TypeError, ValueError):
        dict_args = {"args": list(args), **kwargs}
    dict_args.pop("self", None)
    # A **kwargs catch-all is flattened so its keys are named.
    dict_extra: Any = dict_args.pop("kwargs", None)
    if isinstance(dict_extra, dict):
        dict_args.update(dict_extra)
    # Normal return with the named arguments.
    return dict_args


def cached_call(
    str_operation: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator: a public method consults the cache when it is the outermost
    call, runs through otherwise. Works on sync and async methods.
    """

    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        if getattr(func, _WRAPPED_MARKER, False):
            # Early return: already wrapped.
            return func
        str_method: str = func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async(self: Any, *args: Any, **kwargs: Any) -> Any:
                if RESPONSE_CACHE_ACTIVE.get():
                    # Early return: an outer call owns the decision.
                    return await func(self, *args, **kwargs)
                token: Token[bool] = RESPONSE_CACHE_ACTIVE.set(True)
                try:
                    plan: _Plan | None = plan_for(
                        self,
                        str_operation,
                        str_method,
                        _bound_args(func, (self, *args), kwargs),
                    )
                    if plan is None:
                        # Early return: not cached.
                        return await func(self, *args, **kwargs)
                    bool_hit, hit = lookup(self, plan)
                    if bool_hit:
                        # Early return with the stored result.
                        return hit
                    with capture_request_calls(str_method) as capture:
                        result: Any = await func(self, *args, **kwargs)
                    store(plan, result, capture)
                    # Normal return with the live result.
                    return result
                finally:
                    RESPONSE_CACHE_ACTIVE.reset(token)

            setattr(_async, _WRAPPED_MARKER, True)
            return _async

        @functools.wraps(func)
        def _sync(self: Any, *args: Any, **kwargs: Any) -> Any:
            if RESPONSE_CACHE_ACTIVE.get():
                # Early return: an outer call owns the decision.
                return func(self, *args, **kwargs)
            token: Token[bool] = RESPONSE_CACHE_ACTIVE.set(True)
            try:
                plan: _Plan | None = plan_for(
                    self,
                    str_operation,
                    str_method,
                    _bound_args(func, (self, *args), kwargs),
                )
                if plan is None:
                    # Early return: not cached.
                    return func(self, *args, **kwargs)
                bool_hit, hit = lookup(self, plan)
                if bool_hit:
                    # Early return with the stored result.
                    return hit
                with capture_request_calls(str_method) as capture:
                    result: Any = func(self, *args, **kwargs)
                store(plan, result, capture)
                # Normal return with the live result.
                return result
            finally:
                RESPONSE_CACHE_ACTIVE.reset(token)

        setattr(_sync, _WRAPPED_MARKER, True)
        return _sync

    return _decorate


def wrap_cacheable_methods(cls: type) -> None:
    """
    Wraps the cacheable public methods a class defines itself; called from
    AIBase.__init_subclass__ so every engine, chain, and router is covered
    without each remembering to decorate.
    """
    # Loop over the methods this class defines in its own body.
    for str_name, str_operation in DICT_WRAPPED_METHOD_OPERATIONS.items():
        func: Any = cls.__dict__.get(str_name)
        if (
            func is None
            or not callable(func)
            or isinstance(func, (staticmethod, classmethod))
        ):
            continue
        setattr(cls, str_name, cached_call(str_operation)(func))
    # Normal return after wrapping.
    return None
