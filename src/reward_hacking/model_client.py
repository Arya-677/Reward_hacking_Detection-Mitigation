import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class CallMetrics(BaseModel):
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    retries: int = 0
    error: Optional[str] = None
    raw_response: Optional[str] = None


class ModelClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 180.0,
        keep_alive: str = "30m",
        num_ctx: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def check_ollama_running(self) -> bool:
        """Check if Ollama server is running and reachable."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_installed_models(self) -> List[Dict[str, Any]]:
        """List all models installed locally in Ollama."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
        except httpx.ConnectError:
            raise ConnectionError(
                f"Ollama is not running: start it with 'ollama serve' or launch the Ollama app (tried {self.base_url})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to list models from Ollama: {e}")

    def is_model_installed(self, model_name: str) -> bool:
        """Check if a specific model tag or name is installed."""
        models = self.list_installed_models()
        target = model_name.strip().lower()
        target_base = target.split(":")[0]
        for m in models:
            name = m.get("name", "").lower()
            model_id = m.get("model", "").lower()
            if target in (name, model_id):
                return True
            # Also match without tag if exact base name matches single model
            if name.split(":")[0] == target_base and ":" not in target:
                return True
        return False

    def get_model_digest(self, model_name: str) -> Optional[str]:
        """Get the sha256 digest for a model."""
        models = self.list_installed_models()
        target = model_name.strip().lower()
        for m in models:
            name = m.get("name", "").lower()
            if target == name or target == m.get("model", "").lower():
                return m.get("digest")
        return None

    def test_model_connection(self, model_name: str) -> bool:
        """Run a minimal generation to test model readiness."""
        if not self.is_model_installed(model_name):
            raise ValueError(f"Model '{model_name}' is not installed: run 'ollama pull {model_name}'")
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "options": {"num_predict": 5, "temperature": 0.0},
                "keep_alive": self.keep_alive,
            }
            resp = self._client.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
            return resp.status_code == 200
        except Exception as e:
            raise RuntimeError(f"Error testing model '{model_name}': {e}")

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Strip <think>...</think> blocks emitted by thinking models like Qwen3."""
        # Remove complete think blocks
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # If closing tag exists without opening or vice-versa
        cleaned = re.sub(r"^.*?<\/think>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    @staticmethod
    def _extract_json_string(text: str) -> str:
        """Extract JSON substring if model enclosed it in markdown code fences or conversational text."""
        stripped = text.strip()
        # Look for markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Look for first { to last }
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return stripped[start : end + 1].strip()
        return stripped

    def generate_structured(
        self,
        model: str,
        messages: List[Dict[str, str]],
        schema_class: Type[T],
        temperature: float = 0.4,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
        retries: int = 1,
        disable_thinking: bool = True,
    ) -> Tuple[Optional[T], CallMetrics]:
        """Generate structured response validated against a Pydantic schema with retry support."""
        if not self.is_model_installed(model):
            raise ValueError(f"Model '{model}' is not installed: run 'ollama pull {model}'")

        schema_json = schema_class.model_json_schema()
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.num_ctx,
        }
        if seed is not None:
            options["seed"] = seed

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0
        retry_count = 0
        current_messages = list(messages)

        for attempt in range(retries + 1):
            payload = {
                "model": model,
                "messages": current_messages,
                "format": schema_json,
                "stream": False,
                "options": options,
                "keep_alive": self.keep_alive,
            }
            if disable_thinking:
                payload["think"] = False

            t0 = time.perf_counter()
            try:
                resp = self._client.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds
                )
                latency = time.perf_counter() - t0
                total_latency += latency

                if resp.status_code != 200:
                    err_msg = f"Ollama HTTP {resp.status_code}: {resp.text}"
                    return None, CallMetrics(
                        model=model,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        latency_seconds=total_latency,
                        retries=retry_count,
                        error=err_msg,
                    )

                data = resp.json()
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                raw_content = data.get("message", {}).get("content", "")
                cleaned_content = self._strip_thinking_tags(raw_content)
                extracted_json = self._extract_json_string(cleaned_content)

                try:
                    parsed_dict = json.loads(extracted_json)
                    validated_obj = schema_class.model_validate(parsed_dict)
                    return validated_obj, CallMetrics(
                        model=model,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        latency_seconds=total_latency,
                        retries=retry_count,
                        raw_response=cleaned_content,
                    )
                except (json.JSONDecodeError, ValidationError) as parse_err:
                    if attempt < retries:
                        retry_count += 1
                        # One controlled repair retry
                        current_messages.append({"role": "assistant", "content": raw_content})
                        current_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response was invalid. "
                                    f"Please return valid JSON matching this schema: {json.dumps(schema_json)}"
                                ),
                            }
                        )
                        continue
                    else:
                        return None, CallMetrics(
                            model=model,
                            prompt_tokens=total_prompt_tokens,
                            completion_tokens=total_completion_tokens,
                            total_tokens=total_prompt_tokens + total_completion_tokens,
                            latency_seconds=total_latency,
                            retries=retry_count,
                            error=f"Schema validation failed: {parse_err}",
                            raw_response=cleaned_content,
                        )

            except Exception as req_err:
                latency = time.perf_counter() - t0
                total_latency += latency
                return None, CallMetrics(
                    model=model,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    latency_seconds=total_latency,
                    retries=retry_count,
                    error=str(req_err),
                )

        return None, CallMetrics(
            model=model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            latency_seconds=total_latency,
            retries=retry_count,
            error="Exhausted retries without valid output",
        )
