"""
NVIDIA NIM Inference Client.
OpenAI-compatible API wrapper for NVIDIA NIM microservice.
"""

import os
import time
from typing import List, Optional

import requests

from .config import DEFAULT_NIM_MODEL, MAX_RETRIES, NIM_BASE_URL, RETRY_DELAY, SEED


class NIMInferenceClient:
    """
    Client for querying NVIDIA NIM microservice endpoints.
    Follows the OpenAI-compatible chat completions API pattern.
    """

    def __init__(
        self,
        model: str = DEFAULT_NIM_MODEL,
        api_key: Optional[str] = None,
        base_url: str = NIM_BASE_URL,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY not set. "
                "Get your key at https://build.nvidia.com"
            )

        print(f"[INFO] NIM client initialized — model: {self.model}")

    def query(
        self,
        prompt: str,
        system_prompt: str = (
            "You are a precise financial analyst. "
            "Answer questions accurately based on the provided context."
        ),
        temperature: float = 0.1,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> str:
        """Send a single query to the NIM endpoint."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": SEED,
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"[WARN] Retry {attempt + 1}/{MAX_RETRIES}: {e}"
                    )
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(
                        f"[ERROR] Failed after {MAX_RETRIES} retries: {e}"
                    )
                    return f"[ERROR] {str(e)}"

    def batch_query(
        self,
        prompts: List[str],
        system_prompt: str = (
            "You are a precise financial analyst. "
            "Answer questions accurately based on the provided context."
        ),
        delay: float = 0.5,
        **kwargs,
    ) -> List[str]:
        """Query multiple prompts sequentially with rate limiting."""
        results = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            print(f"[INFO] Querying {i + 1}/{total}...", end="\r")
            result = self.query(
                prompt, system_prompt=system_prompt, **kwargs
            )
            results.append(result)
            if i < total - 1:
                time.sleep(delay)

        print(
            f"[INFO] Batch query complete: {total} responses collected."
        )
        return results
