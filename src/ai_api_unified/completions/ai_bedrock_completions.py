# ai_bedrock_completions.py

import json
import time
import boto3  # AWS SDK for Python to call Bedrock runtime service
from botocore.exceptions import ClientError
from typing import ClassVar, Dict, List, Type
from pydantic import ValidationError

from ..ai_base import AIBaseCompletions, AIStructuredPrompt

from ..util.env_settings import EnvSettings


class AiBedrockCompletions(AIBaseCompletions):
    """
    Completion client for Amazon Bedrock via the Converse API, with
    structured-output prompts.
    """

    def __init__(self, model: str = "", dimensions: int = 0):
        settings = EnvSettings()
        # self.model = model if model else "amazon.nova-pro-v1:0"
        self.model = (
            model
            if model
            else settings.get("COMPLETIONS_MODEL_NAME", "amazon.nova-lite-v1:0")
        )
        self.completions_model = model
        self.dimensions = dimensions
        # AWS Region: determines which Bedrock endpoint is used; loaded from environment via EnvSettings
        self.region_name = settings.get("AWS_REGION", "us-east-1")

        try:
            # Initialize the Bedrock Runtime client. boto3 will pick up AWS credentials
            # from environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or attached IAM role.
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to create Bedrock runtime client. Check your AWS credentials and AWS_REGION."
            ) from e

        # Retries with exponential backoff delays for transient API errors
        self.backoff_delays = [1, 2, 4, 8]

    @property
    def max_context_tokens(self) -> int:
        """
        Look up the OpenAI context window for the current model_name.
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
        "amazon.nova-micro-v1:0": 4096000,  # 4k tokens
        "amazon.nova-lite-v1:0": 8192000,  # 8k tokens
        "amazon.nova-pro-v1:0": 16384000,  # 16k tokens
        "amazon.nova-premier-v1:0": 32768000,  # 32k tokens
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": 8192000,  # 8k tokens
        # …add any others we support…
    }

    # dollars per 1 k tokens for each supported model
    DICT_PRICES: ClassVar[Dict[str, float]] = {
        "amazon.nova-micro-v1:0": 0.0004,  # $0.40 per million tokens
        "amazon.nova-lite-v1:0": 0.0008,  # $0.80 per million tokens
        "amazon.nova-pro-v1:0": 0.0016,  # $1.60 per million tokens
        "amazon.nova-premier-v1:0": 0.0032,  # $3.20 per million tokens
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": 0.0015,  # $1.50 per million tokens
    }

    @property
    def list_model_names(self) -> List[str]:
        # As of May 2025, aggregated from OpenAI docs and release notes:
        return [
            "amazon.nova-micro-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-pro-v1:0",
            "amazon.nova-premier-v1:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ]

    def _extract_json_text_from_converse_response(self, resp: dict) -> str:
        content = resp.get("output", {}).get("message", {}).get("content", [])
        if not content or "text" not in content[0]:
            raise RuntimeError("No text in response")
        return content[0]["text"]

    def strict_schema_prompt(
        self,
        prompt: str,
        response_model: Type[AIStructuredPrompt],
        max_response_tokens: int = 512,
    ) -> AIStructuredPrompt:
        """
        Free-form JSON generation with Pydantic v2 post-validation.
        Guarantees variety by using sampling instead of tool-locking.
        """
        prompt += self.generate_prompt_entropy_tag()

        # 1) Auto-append the generic JSON-schema instruction—no assumptions
        prompt_addendum: str = self.generate_prompt_addendum_json_schema_instruction(
            response_model, code_fence=False
        )
        full_prompt = f"{prompt}\n\n{prompt_addendum}"

        # 2. Build plain messages
        messages = [
            {"role": "user", "content": [{"text": full_prompt}]},
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
                    inferenceConfig=inference_config,
                )

                # TO DO - check stop reason. Should be "end_turn" when successful. "max_tokens" means we overran the limit.
                # 5. Pull free-form text and parse JSON
                raw_json = self._extract_json_text_from_converse_response(resp)
                # Remove trailing ``` from raw_json
                raw_json = raw_json.rstrip("```").strip()
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict) and "properties" in parsed:
                    parsed = parsed["properties"]
                return response_model.model_validate(parsed)

            except json.JSONDecodeError as jde:
                # malformed JSON → try to self-heal
                fixed = self._repair_json(raw_json)
                try:
                    parsed = json.loads(fixed)
                    return response_model.model_validate(parsed)

                except json.JSONDecodeError:
                    # still broken → sleep & retry or give up
                    if attempt < len(self.backoff_delays):
                        time.sleep(delay)
                        continue
                    raise RuntimeError(
                        f"JSON parse (even after repair) failed after {attempt} tries: {jde}"
                    ) from jde

            except ValidationError as ve:
                errors = ve.errors()
                # You can print, log, or include these in a new exception
                print("❌ raw_json:", raw_json)
                print("❌ parsed payload:", parsed)
                print("❌ validation errors:", errors)
                # Option A: re-raise a richer error
                raise RuntimeError(
                    f"{response_model.__name__}.model_validate() failed:\n"
                    f"  raw_json: {raw_json}\n"
                    f"  parsed: {parsed}\n"
                    f"  errors: {errors}"
                ) from ve

            except self.client.exceptions.ModelErrorException as me:
                # AWS Bedrock error → retry
                print(f"ModelErrorException: {me}")
                if attempt < len(self.backoff_delays):
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Bedrock ModelErrorException after {attempt} tries: {me}"
                ) from me

            except ClientError as ce:
                # AWS transient → retry
                if attempt < len(self.backoff_delays):
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Bedrock error after {attempt} tries: {ce}") from ce

            except Exception as e:
                # catch-all transient → retry
                if attempt < len(self.backoff_delays):
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Unexpected failure after {attempt} tries: {e}"
                ) from e

    def send_prompt(self, prompt: str) -> str:
        # AWS Bedrock expects messages in this format for the Converse API
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        # Inference settings define how the model generates text:
        #   temperature (0.0–1.0): controls randomness of output. Lower values (e.g., 0.2) make responses more predictable;
        #       higher values (e.g., 0.8) increase variation by sampling less likely tokens.
        #   topP (0.0–1.0): limits the model’s next-token choices to the smallest set of tokens whose cumulative probability
        #       mass is at least topP. For example, topP=0.9 restricts choices to tokens accounting for 90% of total probability,
        #       excluding very unlikely options. This balances consistency and diversity.
        #   maxTokens: maximum number of tokens to generate in the response. Tokens roughly correspond to words or word parts;
        #       setting this prevents excessively long outputs.
        #
        # Example values:
        #   temperature: 0.2    # low randomness for focused, reliable outputs
        #   topP: 0.85          # restrict next-token choices to top 85% cumulative probability
        #   maxTokens:  256     # cap response length for concise, JSON-formatted results
        inference_config = {"temperature": 0.2, "topP": 0.85, "maxTokens": 256}

        for attempt, delay in enumerate(self.backoff_delays, start=1):
            try:
                response = self.client.converse(
                    modelId=self.model,
                    messages=messages,
                    inferenceConfig=inference_config,
                )
                # Extract the text from the first content block in the response
                content = (
                    response.get("output", {}).get("message", {}).get("content", [])
                )
                if content and "text" in content[0]:
                    return content[0]["text"]
                return ""
            except Exception as e:
                if attempt == len(self.backoff_delays):
                    raise RuntimeError(f"Bedrock converse failed: {e}")
                # Wait before retrying
                time.sleep(delay)
