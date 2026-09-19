import json
from typing import Any
import httpx
from journal_ai.config.settings import settings


class LLMClient:
    """
    Unified Multi-Model Client supporting OpenAI, Google Gemini, Groq, and Heuristic Fallbacks.
    Uses async httpx for low latency and zero external SDK dependencies.
    """

    def __init__(self):
        self.openai_key = settings.openai_api_key
        self.gemini_key = settings.gemini_api_key
        self.groq_key = settings.groq_api_key
        self.provider = settings.llm_provider

    def is_llm_available(self) -> bool:
        """Returns True if at least one valid LLM API key is configured."""
        return bool(self.openai_key or self.gemini_key or self.groq_key)

    async def generate_json(self, prompt: str, system_prompt: str = "") -> dict[str, Any] | None:
        """Generate a JSON response from the active LLM provider."""
        text = await self.generate_text(prompt, system_prompt=system_prompt + "\nOutput MUST be valid raw JSON.")
        if not text:
            return None

        # Clean JSON markdown blocks if any
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            # Try finding first { and last }
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(cleaned[start:end+1])
                except Exception:
                    pass
            return None

    async def generate_text(self, prompt: str, system_prompt: str = "") -> str | None:
        """Generate text using the best available configured LLM."""
        # 1. Try OpenAI if key is present
        if self.openai_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json",
                }
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": settings.openai_model or "gpt-4o-mini",
                    "messages": messages,
                    "temperature": 0.3,
                }

                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                print(f"[LLMClient] OpenAI call notice: {exc}")

        # 2. Try Gemini if key is present
        if self.gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent?key={self.gemini_key}"
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                payload = {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": 0.3},
                }
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as exc:
                print(f"[LLMClient] Gemini call notice: {exc}")

        # 3. Try Groq if key is present
        if self.groq_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                }
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                print(f"[LLMClient] Groq call notice: {exc}")

        return None
