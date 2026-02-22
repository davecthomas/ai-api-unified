# ai_bedrock_completions.py

import json
from typing import Any, ClassVar, Type


BEDROCK_DEPENDENCIES_AVAILABLE: bool = False
try:
    from ..ai_bedrock_base import (
        AIBedrockBase,
        ClientError,
    )

    BEDROCK_DEPENDENCIES_AVAILABLE = True
except ImportError:
    BEDROCK_DEPENDENCIES_AVAILABLE = False


if BEDROCK_DEPENDENCIES_AVAILABLE:
    from pydantic import ValidationError

    from ..ai_base import (
        AIBaseCompletions,
        AIStructuredPrompt,
        AICompletionsPromptParamsBase,
        SupportedDataType,
    )
    from ..util.env_settings import EnvSettings

    class AiBedrockCompletions(AIBedrockBase, AIBaseCompletions):
        """
        Completion client for Amazon Bedrock via the Converse API, with
        structured-output prompts.
        """

        def __init__(self, model: str = "", **kwargs: Any):
            settings = EnvSettings()
            resolved_model: str = (
                model
                if model
                else settings.get("COMPLETIONS_MODEL_NAME", "amazon.nova-lite-v1:0")
            )
            self.completions_model: str = resolved_model
            AIBedrockBase.__init__(self, model=resolved_model, **kwargs)
            AIBaseCompletions.__init__(self, model=resolved_model)
            # Reuse the existing retry schedule from the base but allow overrides via kwargs
            custom_backoff: list[float] | None = kwargs.get("backoff_delays")
            if custom_backoff is not None:
                self.backoff_delays = custom_backoff

        @property
        def max_context_tokens(self) -> int:
            """
            Look up the Bedrock model context window for the current model_name.
            Falls back to 0 if unknown (i.e. no guard will occur).
            """
            return self.DICT_CONTEXT_WINDOWS.get(self.completions_model, 0)

        @property
        def price_per_1k_tokens(self) -> float:
            """
            Look up the cost-per-1 k tokens for this model.
            Returns 0.0 if unknown (no guard).
            """
            return self.DICT_PRICES.get(self.completions_model, 0.0)

        DICT_CONTEXT_WINDOWS: dict[str, int] = {
            "amazon.nova-micro-v1:0": 4_096_000,
            "amazon.nova-lite-v1:0": 8_192_000,
            "amazon.nova-pro-v1:0": 16_384_000,
            "amazon.nova-premier-v1:0": 32_768_000,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0": 8_192_000,
        }

        # dollars per 1 k tokens for each supported model
        DICT_PRICES: ClassVar[dict[str, float]] = {
            "amazon.nova-micro-v1:0": 0.0004,
            "amazon.nova-lite-v1:0": 0.0008,
            "amazon.nova-pro-v1:0": 0.0016,
            "amazon.nova-premier-v1:0": 0.0032,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0": 0.0015,
        }

        @property
        def list_model_names(self) -> list[str]:
            # As of Oct 2025, aggregated from AWS release notes:
            return [
                "amazon.nova-micro-v1:0",
                "amazon.nova-lite-v1:0",
                "amazon.nova-pro-v1:0",
                "amazon.nova-premier-v1:0",
                "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            ]

        def _extract_json_text_from_converse_response(
            self, resp: dict[str, Any]
        ) -> str:
            content = resp.get("output", {}).get("message", {}).get("content", [])
            if not content or "text" not in content[0]:
                raise RuntimeError("No text in response")
            return content[0]["text"]

        def strict_schema_prompt(
            self,
            prompt: str,
            response_model: Type[AIStructuredPrompt],
            max_response_tokens: int = 512,
            *,
            other_params: AICompletionsPromptParamsBase | None = None,
        ) -> AIStructuredPrompt:
            """
            Free-form JSON generation with Pydantic v2 post-validation.
            Guarantees variety by using sampling instead of tool-locking.
            """
            prompt += self.generate_prompt_entropy_tag()

            # 1) Auto-append the generic JSON-schema instruction—no assumptions
            prompt_addendum: str = (
                self.generate_prompt_addendum_json_schema_instruction(
                    response_model, code_fence=False
                )
            )
            full_prompt = f"{prompt}\n\n{prompt_addendum}"
            system_prompt: str = (
                other_params.system_prompt
                if other_params is not None and other_params.system_prompt is not None
                else AICompletionsPromptParamsBase.DEFAULT_STRICT_SCHEMA_SYSTEM_PROMPT
            )

            # 2. Build plain messages
            user_content: list[dict[str, Any]] = self._build_user_content(
                full_prompt, other_params
            )
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"text": "```json"}]},
            ]

            # 3. Sampling settings for maximum variety
            inference_config = {
                "maxTokens": max_response_tokens * 2,  # TO DO - fix this hack
                "temperature": 0.9,  # high randomness
                "stopSequences": ["```"],  # stop at JSON end marker
                # "topP": 0.9,  # nucleus sampling
            }

            # 4. Retry-loop on AWS, JSON, or validation errors
            for attempt, delay in enumerate(self.backoff_delays, start=1):
                try:
                    resp = self.client.converse(
                        modelId=self.model,
                        messages=messages,
                        system=[{"text": system_prompt}],
                        inferenceConfig=inference_config,
                    )

                    raw_json = self._extract_json_text_from_converse_response(resp)
                    raw_json = raw_json.rstrip("```").strip()
                    parsed = json.loads(raw_json)
                    if isinstance(parsed, dict) and "properties" in parsed:
                        parsed = parsed["properties"]
                    return response_model.model_validate(parsed)

                except json.JSONDecodeError as jde:
                    fixed = self._repair_json(raw_json)
                    try:
                        parsed = json.loads(fixed)
                        return response_model.model_validate(parsed)

                    except json.JSONDecodeError:
                        if attempt < len(self.backoff_delays):
                            self._sleep_with_backoff(delay)
                            continue
                        raise RuntimeError(
                            f"JSON parse (even after repair) failed after {attempt} tries: {jde}"
                        ) from jde

                except ValidationError as ve:
                    errors = ve.errors()
                    print("❌ raw_json:", raw_json)
                    print("❌ parsed payload:", parsed)
                    print("❌ validation errors:", errors)
                    raise RuntimeError(
                        f"{response_model.__name__}.model_validate() failed:\n"
                        f"  raw_json: {raw_json}\n"
                        f"  parsed: {parsed}\n"
                        f"  errors: {errors}"
                    ) from ve

                except self.client.exceptions.ModelErrorException as me:
                    print(f"ModelErrorException: {me}")
                    if attempt < len(self.backoff_delays):
                        self._sleep_with_backoff(delay)
                        continue
                    raise RuntimeError(
                        f"Bedrock ModelErrorException after {attempt} tries: {me}"
                    ) from me

                except ClientError as ce:
                    if attempt < len(self.backoff_delays):
                        self._sleep_with_backoff(delay)
                        continue
                    raise RuntimeError(
                        f"Bedrock error after {attempt} tries: {ce}"
                    ) from ce

                except Exception as e:
                    if attempt < len(self.backoff_delays):
                        self._sleep_with_backoff(delay)
                        continue
                    raise RuntimeError(
                        f"Unexpected failure after {attempt} tries: {e}"
                    ) from e

        def send_prompt(
            self,
            prompt: str,
            *,
            other_params: AICompletionsPromptParamsBase | None = None,
        ) -> str:
            # AWS Bedrock expects messages in this format for the Converse API
            user_content: list[dict[str, Any]] = self._build_user_content(
                prompt, other_params
            )
            messages = [{"role": "user", "content": user_content}]

            system_prompt: str = (
                other_params.system_prompt
                if other_params is not None and other_params.system_prompt is not None
                else AICompletionsPromptParamsBase.DEFAULT_SYSTEM_PROMPT
            )

            inference_config = {"temperature": 0.2, "topP": 0.85, "maxTokens": 256}

            for attempt, delay in enumerate(self.backoff_delays, start=1):
                try:
                    response = self.client.converse(
                        modelId=self.model,
                        messages=messages,
                        system=[{"text": system_prompt}],
                        inferenceConfig=inference_config,
                    )
                    content = (
                        response.get("output", {}).get("message", {}).get("content", [])
                    )
                    if content and "text" in content[0]:
                        return content[0]["text"]
                    return ""
                except Exception as e:
                    if attempt == len(self.backoff_delays):
                        raise RuntimeError(f"Bedrock converse failed: {e}")
                    self._sleep_with_backoff(delay)

        def _build_user_content(
            self,
            prompt: str,
            other_params: AICompletionsPromptParamsBase | None,
        ) -> list[dict[str, Any]]:
            """
            Build AWS Bedrock message content supporting optional image attachments.
            """

            content_parts: list[dict[str, Any]] = [{"text": prompt}]
            if other_params is None or not other_params.has_included_media:
                return content_parts

            for (
                _,
                media_type,
                media_bytes,
                mime_type,
            ) in other_params.iter_included_media():
                if media_type is not SupportedDataType.IMAGE:
                    continue

                if len(media_bytes) > AICompletionsPromptParamsBase.MAX_IMAGE_BYTES:
                    raise ValueError(
                        "Image attachment exceeds the maximum allowed size of "
                        f"{AICompletionsPromptParamsBase.MAX_IMAGE_BYTES} bytes."
                    )

                content_parts.append(
                    {
                        "image": {
                            "format": self._derive_image_format(mime_type),
                            "source": {"bytes": media_bytes},
                        }
                    }
                )

            return content_parts

        @staticmethod
        def _derive_image_format(mime_type: str) -> str:
            """
            Extract the provider-specific image format token from a MIME type.
            """

            return mime_type.split("/", maxsplit=1)[-1]

        @staticmethod
        def _repair_json(raw_json: str) -> str:
            """
            Attempt a minimal repair of malformed JSON returned by the model.

            The current implementation simply returns the original string, acting as a
            placeholder hook for more advanced repair strategies if needed later.
            """

            return raw_json

else:  # pragma: no cover - fallback when boto3 missing

    class AiBedrockCompletions:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "Amazon Bedrock completions require installing the 'bedrock' extra."
            )
