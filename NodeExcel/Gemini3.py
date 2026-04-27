import base64
import io
import json
import math
import os
import secrets
import urllib.request
import urllib.error
import ssl
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

MISSING_GOOGLE_GENAI_MSG = (
    "Missing dependency 'google-genai'. Install it in the same Python environment that runs ComfyUI, "
    "then restart ComfyUI. Example commands:\n"
    f"  python -m pip install -r custom_nodes/{os.path.basename(os.path.dirname(__file__))}/requirements.txt\n"
    "  python -m pip install \"google-genai>=1.7.0,<2.0.0\""
)

try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError as exc:
    if (exc.name or "").startswith("google"):
        raise ModuleNotFoundError(MISSING_GOOGLE_GENAI_MSG) from exc
    raise
except ImportError as exc:
    raise ImportError(f"{MISSING_GOOGLE_GENAI_MSG}\nOriginal import error: {exc}") from exc


CONFIG_FILENAME = "config.json"
PROVIDER_BASE_URLS = {
    "apistudio": "https://apistudio.cc",
    "apistudio.cc": "https://apistudio.cc",
    "mmchat": "https://llm-api.mmchat.xyz/gemini",
    "mmchat-gemini": "https://llm-api.mmchat.xyz/gemini",
}


def _normalize_api_url(api_url: str) -> str:
    """
    Normalize provider/base URL input so users can pass:
    - provider alias: apistudio / mmchat
    - bare host/path: apistudio.cc or llm-api.mmchat.xyz/gemini
    - full endpoint URL: https://.../v1/chat/completions
    """
    url = (api_url or "").strip().strip("\"'")
    # Prevent malformed endpoints like .../v1/chat/completions:
    url = url.rstrip(" \t\r\n:;,")
    if not url:
        return ""

    alias = url.lower().strip("/")
    if alias in PROVIDER_BASE_URLS:
        return PROVIDER_BASE_URLS[alias]

    if "://" not in url:
        url = f"https://{url}"
    return url.rstrip("/")


def _dedupe_keys(values: List[str]) -> List[str]:
    import re
    seen = set()
    cleaned = []
    for value in values:
        if not isinstance(value, str):
            continue
        key = (value or "").strip().strip("\"'")
        key = re.sub(r"\s+", "", key)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def _load_api_keys_from_config() -> List[str]:
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Failed to load {CONFIG_FILENAME}: {exc}") from exc

    keys = data.get("api_keys", [])
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        raise ValueError(f"{CONFIG_FILENAME} api_keys must be a list of strings.")
    return _dedupe_keys(keys)


def _resolve_api_keys(user_key: str) -> List[str]:
    env_key = os.getenv("GEMINI_API_KEY", "")
    keys = _dedupe_keys([user_key] + _load_api_keys_from_config() + [env_key])
    if not keys:
        raise ValueError(
            "API key is required. Provide it in the node, add it to config.json, or set GEMINI_API_KEY."
        )
    return keys


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(token in message for token in ("resource_exhausted", "quota", "rate limit", "429")):
        return True

    for attr in ("status_code", "code"):
        code = getattr(exc, attr, None)
        if callable(code):
            try:
                code = code()
            except Exception:
                code = None
        if code == 429 or str(code).upper() == "RESOURCE_EXHAUSTED":
            return True
    return False


def _is_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    model_tokens = (
        "unsupported model",
        "unknown model",
        "invalid model",
        "model not found",
        "model_not_found",
        "no such model",
        "model does not exist",
        "is not supported for",
        "not found for api version",
        "no available channel for model",
    )
    if any(token in message for token in model_tokens):
        return True
    if "model" in message and ("not found" in message or "does not exist" in message):
        return True

    for attr in ("status_code", "code"):
        code = getattr(exc, attr, None)
        if callable(code):
            try:
                code = code()
            except Exception:
                code = None
        if code == 404 or str(code).upper() == "NOT_FOUND":
            return True
    return False


def _run_with_key_rotation(api_keys: List[str], request_fn, api_url: str = ""):
    last_quota_error = None
    for key in api_keys:
        client_kwargs = {"api_key": key}
        if api_url:
            client_kwargs["http_options"] = types.HttpOptions(base_url=api_url)
        client = genai.Client(**client_kwargs)
        try:
            return request_fn(client)
        except Exception as exc:
            if _is_quota_error(exc):
                last_quota_error = exc
                continue
            raise
    if last_quota_error is not None:
        raise last_quota_error
    raise ValueError("No valid API key available.")


# ---------------------------------------------------------------------------
# API mode detection and multi-mode call helpers
# ---------------------------------------------------------------------------

def _detect_api_mode(api_url: str, node_type: str = "image") -> str:
    """
    Detect which API mode to use based on the api_url input.

    node_type: "image" or "text" — determines default format for third-party URLs.
      - image nodes → Gemini REST format (/v1beta/models/{model}:generateContent)
      - text nodes  → OpenAI-compatible format (/v1/chat/completions)

    Returns:
        "sdk"      - empty string → use google-genai SDK with default endpoint
        "official" - Gemini REST format (user provided full Gemini URL)
        "openai"   - OpenAI-compatible format (user provided full OpenAI URL or /openai path)
        "gemini"   - Third-party URL for image node → auto-append Gemini REST path
        "openai_default" - Third-party URL for text node → auto-append OpenAI path
    """
    url = _normalize_api_url(api_url)
    if not url:
        return "sdk"

    url_lower = url.lower()

    # OpenAI-compatible path (e.g. googleapis.com/v1beta/openai/)
    if "/openai" in url_lower:
        return "openai"

    # OpenAI full endpoint (e.g. /v1/chat/completions, /v1/images/generations)
    if url_lower.endswith("/chat/completions") or url_lower.endswith("/images/generations"):
        return "openai"

    # Gemini REST format indicators: :generateContent or /v1beta/models/ or /v1/models/
    if ":generatecontent" in url_lower or "/v1beta/models/" in url_lower or "/v1/models/" in url_lower:
        return "official"

    # Google official domain → Gemini REST
    if "googleapis.com" in url_lower:
        return "official"

    # Third-party URL: image → Gemini REST, text → OpenAI
    if node_type == "text":
        return "openai_default"
    return "gemini"


def _urlopen_with_retry(url: str, data: bytes, headers: dict, max_retries: int = 3) -> dict:
    """
    Robust HTTP POST with automatic fallback:
    1. Normal SSL → if SSLError, retry with relaxed SSL
    2. Bearer auth → if 401, try x-api-key header
    3. x-api-key → if 401, try ?key= URL param
    4. 429 rate limit → retry with backoff
    """
    import time

    last_error = None
    attempt_errors = []

    # Auth strategies to try: (auth_type, header_dict, url_with_key)
    is_google = "googleapis.com" in url.lower()

    if is_google:
        # Google official: only ?key= needed
        auth_strategies = [{"headers": headers, "url": url}]
    else:
        api_key = headers.get("_raw_api_key", "")
        base_headers = {
            k: v for k, v in headers.items()
            if not k.startswith("_") and k.lower() not in ("authorization", "x-api-key", "api-key")
        }
        auth_strategies = []
        # Strategy 1: Bearer token
        auth_strategies.append({
            "headers": {**base_headers, "Authorization": f"Bearer {api_key}"} if api_key else base_headers,
            "url": url,
            "label": "Bearer",
        })
        if api_key:
            auth_strategies.append({
                "headers": {**base_headers, "Authorization": api_key},
                "url": url,
                "label": "Authorization",
            })
            auth_strategies.append({
                "headers": {**base_headers, "x-api-key": api_key},
                "url": url,
                "label": "x-api-key",
            })
            auth_strategies.append({
                "headers": {**base_headers, "api-key": api_key},
                "url": url,
                "label": "api-key",
            })
            # Strategy 3: ?key= URL param
            sep = "&" if "?" in url else "?"
            auth_strategies.append({
                "headers": base_headers,
                "url": f"{url}{sep}key={api_key}",
                "label": "?key=",
            })

    for attempt, strategy in enumerate(auth_strategies):
        req = urllib.request.Request(
            strategy["url"], data=data, method="POST",
            headers=strategy["headers"],
        )
        for ssl_attempt in range(2):
            _ssl_ctx = ssl.create_default_context()
            if ssl_attempt == 1:
                _ssl_ctx.check_hostname = False
                _ssl_ctx.verify_mode = ssl.CERT_NONE

            try:
                with urllib.request.urlopen(req, timeout=600, context=_ssl_ctx) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except ssl.SSLError:
                if ssl_attempt == 0:
                    continue  # retry with relaxed SSL
                last_error = RuntimeError(f"SSL error (mode: {strategy.get('label', 'default')})")
                attempt_errors.append(str(last_error))
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in (401, 403):
                    # Auth failed, try next strategy
                    last_error = RuntimeError(f"Auth failed (tried: {strategy.get('label', 'default')}) - {error_body[:200]}")
                    attempt_errors.append(str(last_error))
                    break  # break SSL loop, try next auth strategy
                elif exc.code == 429 and max_retries > 0:
                    # Rate limited, wait and retry same strategy
                    wait = 2 ** (3 - max_retries)
                    time.sleep(wait)
                    return _urlopen_with_retry(strategy["url"], data, strategy["headers"], max_retries - 1)
                else:
                    attempt_errors.append(
                        f"API error {exc.code} (tried: {strategy.get('label', 'default')}) - {error_body[:200]}"
                    )
                    raise RuntimeError(f"API error {exc.code} at {strategy['url']}: {error_body}") from exc

    if attempt_errors:
        details = " | ".join(attempt_errors[:4])
        raise RuntimeError(f"All auth strategies failed. Details: {details}")
    raise last_error or RuntimeError("All auth strategies failed")


def _call_gemini_official_rest(api_url: str, api_key: str, model: str, request_body: dict) -> dict:
    """Call Gemini official REST API (generateContent endpoint).

    URL building rules (matching the "一毛" plugin convention):
    - Full URL ending with :generateContent → use as-is
    - URL ending with /models → append /{model}:generateContent
    - URL containing /models/{name} → append :generateContent
    - Otherwise → append /v1beta/models/{model}:generateContent

    Auth: always use ?key= (works for both Google official and third-party proxies)
    """
    import re
    base_url = api_url.rstrip("/")
    api_key = re.sub(r"\s+", "", (api_key or "").strip().strip("\"'"))
    # Build the full URL: .../v1beta/models/{model}:generateContent
    if base_url.endswith(":generateContent"):
        url = base_url
    elif base_url.endswith("/models"):
        # URL ends with /models but no model name → append model name
        url = f"{base_url}/{model}:generateContent"
    elif "/models/" in base_url and not base_url.endswith("/models/"):
        # URL contains /models/ and has something after it (model name present)
        url = f"{base_url}:generateContent"
    else:
        url = f"{base_url}/v1beta/models/{model}:generateContent"

    data = json.dumps(request_body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    if "googleapis.com" in url.lower():
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}key={api_key}"
        return _urlopen_with_retry(url, data, headers)

    headers = {
        **headers,
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "api-key": api_key,
        "_raw_api_key": api_key,
    }
    return _urlopen_with_retry(url, data, headers)


def _call_openai_compatible(api_url: str, api_key: str, model: str, prompt: str,
                            size: str = "1024x1024", image_parts_base64: List[str] = None,
                            mode: str = "image", messages: list = None) -> dict:
    """
    Call third-party OpenAI-compatible API.
    mode="image" → POST /v1/images/generations
    mode="chat"  → POST /v1/chat/completions
    If api_url already contains the full endpoint path, use it directly.
    """
    import urllib.parse as _urlparse

    base_url = api_url.rstrip("/")

    # Detect if api_url already contains a full endpoint path
    if mode == "image":
        full_path_suffix = "/images/generations"
    else:
        full_path_suffix = "/chat/completions"

    if base_url.endswith(full_path_suffix):
        endpoint = base_url
    else:
        if mode == "image":
            endpoint = f"{base_url}/v1/images/generations"
        else:
            endpoint = f"{base_url}/v1/chat/completions"

    if mode == "image":
        body = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "url",
        }
    else:
        content = [{"type": "text", "text": prompt}]
        if image_parts_base64:
            for b64 in image_parts_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        body = {
            "model": model,
            "messages": messages or [{"role": "user", "content": content}],
            "temperature": 0.7,
        }

    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "api-key": api_key,
        "_raw_api_key": api_key,
    }
    return _urlopen_with_retry(endpoint, data, headers)


def _build_gemini_rest_body_text(prompt: str, parts_inline: List[dict],
                                 system_instruction: str = "",
                                 thinking_level: str = "default",
                                 seed: int = None,
                                 exclude_fields: set = None) -> dict:
    """Build request body for Gemini REST generateContent (text mode)."""
    exclude = exclude_fields or set()
    user_parts = [{"text": prompt}] + parts_inline
    contents = [{"role": "user", "parts": user_parts}]
    body: dict = {"contents": contents}
    generation_config: dict = {"responseModalities": ["TEXT"]}
    if thinking_level != "default" and "thinkingConfig" not in exclude:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
    if seed is not None and "seed" not in exclude:
        generation_config["seed"] = seed
    body["generationConfig"] = generation_config
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    return body


def _build_gemini_rest_body_image(prompt: str, parts_inline: List[dict],
                                  aspect_ratio: str = "1:1",
                                  image_size: str = "1K",
                                  thinking_level: str = "default",
                                  seed: int = None,
                                  exclude_fields: set = None) -> dict:
    """Build request body for Gemini REST generateContent (image mode)."""
    exclude = exclude_fields or set()
    user_parts = [{"text": prompt}] + parts_inline
    contents = [{"role": "user", "parts": user_parts}]
    body: dict = {"contents": contents}
    image_config: dict = {"aspectRatio": aspect_ratio, "imageSize": image_size}
    if seed is not None and "seed" not in exclude:
        image_config["seed"] = seed
    generation_config: dict = {
        "responseModalities": ["IMAGE", "TEXT"],
        "imageConfig": image_config,
    }
    if thinking_level != "default" and "thinkingConfig" not in exclude:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
    body["generationConfig"] = generation_config
    return body


def _extract_unknown_fields(error_message: str) -> set:
    """Extract unknown field names from a Gemini 400 error message.

    Parses messages like: Unknown name "seed" at 'generation_config.image_config'
    Also handles escaped quotes from proxy wrappers: Unknown name \\"seed\\"
    Returns field names like: {"seed"}
    """
    import re
    fields = set()
    # Match Unknown name "fieldName" with various quote escaping levels
    for m in re.finditer(r'Unknown name\s+[\\]*"(\w+)', error_message):
        fields.add(m.group(1))
    return fields


def _call_gemini_official_rest_with_field_fallback(
    api_url: str, api_key: str, model: str, build_body_fn, max_retries: int = 2
) -> dict:
    """Call Gemini official REST API, automatically retrying without unsupported fields.

    build_body_fn: callable(exclude_fields: set) -> dict
        Function that builds the request body, optionally excluding certain fields.
    """
    exclude_fields = set()
    last_error = None

    for attempt in range(max_retries + 1):
        body = build_body_fn(exclude_fields)
        try:
            return _call_gemini_official_rest(api_url, api_key, model, body)
        except RuntimeError as exc:
            last_error = exc
            error_msg = str(exc)
            # Only retry on 400 errors with unknown field hints
            if "400" in error_msg:
                unknown = _extract_unknown_fields(error_msg)
                if unknown and not unknown.issubset(exclude_fields):
                    exclude_fields.update(unknown)
                    import logging
                    logging.getLogger(__name__).warning(
                        "Gemini API rejected fields %s, retrying without them",
                        unknown,
                    )
                    continue
            raise

    raise last_error


def _parse_gemini_rest_image_response(data: dict):
    """Parse Gemini REST response, return (image_bytes, text)."""
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates in Gemini REST response.")
    parts = candidates[0].get("content", {}).get("parts", [])
    image_bytes = None
    text_parts = []
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            image_bytes = base64.b64decode(inline["data"])
        if part.get("text"):
            text_parts.append(part["text"])
    text = "".join(text_parts).strip()
    return image_bytes, text


def _parse_openai_image_response(data: dict):
    """Parse OpenAI-compatible /v1/images/generations response, return image_bytes."""
    items = data.get("data", [])
    if not items:
        raise ValueError("No image data in OpenAI-compatible response.")
    url_or_b64 = items[0].get("url") or items[0].get("b64_json", "")
    if url_or_b64.startswith("data:"):
        # data:image/png;base64,...
        b64_part = url_or_b64.split(",", 1)[-1]
        return base64.b64decode(b64_part)
    if url_or_b64.startswith("http"):
        # Download the image
        req = urllib.request.Request(url_or_b64)
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    # Assume raw base64
    return base64.b64decode(url_or_b64)


def _parse_openai_chat_response(data: dict) -> str:
    """Parse OpenAI-compatible /v1/chat/completions response, return text."""
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


TEXT_MODEL_OPTIONS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",

]

TEXT_MODEL_ALIASES = {
    "gemini-3-pro-preview": "gemini-3-pro-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
}

TEXT_MODEL_FALLBACKS = {
    "gemini-3-pro-preview": ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash"],
    "gemini-3-flash-preview": ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "gemini-2.5-pro": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "gemini-2.5-flash": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    "gemini-2.5-flash-lite": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
}


def _resolve_text_model(model: str) -> str:
    raw_model = (model or "").strip()
    if not raw_model:
        return TEXT_MODEL_OPTIONS[0]

    model_key = raw_model.lower().replace("_", "-").replace(" ", "-")
    if model_key in TEXT_MODEL_ALIASES:
        return TEXT_MODEL_ALIASES[model_key]

    supported_values = ", ".join(TEXT_MODEL_OPTIONS)
    raise ValueError(
        f"Unsupported text model '{model}'. Supported values: {supported_values}."
    )


def _resolve_text_model_candidates(model: str) -> List[str]:
    resolved = _resolve_text_model(model)
    fallback = TEXT_MODEL_FALLBACKS.get(resolved, [resolved])
    return _dedupe_keys(fallback)


IMAGE_MODEL_OPTIONS = [
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image",
]

IMAGE_SIZE_OPTIONS = [
    "0.5K",
    "1K",
    "2K",
    "4K",
]

IMAGE_THINKING_LEVEL_OPTIONS = [
    "default",
    "minimal",
    "high",
]

IMAGE_SIZE_ALIASES = {
    "0.5k": "512",
    "512": "512",
    "1k": "1K",
    "2k": "2K",
    "4k": "4K",
}

IMAGE_SIZE_512_SUPPORTED_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image",
}

IMAGE_THINKING_LEVEL_SUPPORTED_MODELS = {
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image",
}

IMAGE_MODEL_ALIASES = {
    "nano-banana-pro": ["gemini-3-pro-image-preview"],
    "nano-banana": [
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ],
    "nano-banana-2": [
        "gemini-3.1-flash-image-preview",
        "gemini-3.1-flash-image",
    ],
}


def _resolve_image_model_candidates(model: str) -> List[str]:
    raw_model = (model or "").strip()
    if not raw_model:
        raise ValueError("Image model is required.")

    model_key = raw_model.lower().replace("_", "-").replace(" ", "-")
    # Preferred path: direct top-level model names from IMAGE_MODEL_OPTIONS.
    for option in IMAGE_MODEL_OPTIONS:
        option_key = option.lower().replace("_", "-").replace(" ", "-")
        if model_key == option_key:
            return [option]

    # Backward-compatibility for older workflows that used alias values.
    if model_key in IMAGE_MODEL_ALIASES:
        return _dedupe_keys(IMAGE_MODEL_ALIASES[model_key])

    supported_values = ", ".join(IMAGE_MODEL_OPTIONS)
    raise ValueError(
        f"Unsupported image model '{model}'. Supported values: {supported_values}."
    )


def _normalize_image_size(image_size: str) -> str:
    raw_image_size = (image_size or "").strip()
    if not raw_image_size:
        return "1K"

    image_size_key = raw_image_size.lower().replace(" ", "")
    if image_size_key in IMAGE_SIZE_ALIASES:
        return IMAGE_SIZE_ALIASES[image_size_key]

    supported_values = ", ".join(IMAGE_SIZE_OPTIONS)
    raise ValueError(
        f"Unsupported image_size '{image_size}'. Supported values: {supported_values}. "
        "For 0.5K, the Gemini API expects the literal value '512'."
    )


def _resolve_image_model_candidates_for_size(model_candidates: List[str], image_size: str) -> List[str]:
    if image_size != "512":
        return model_candidates

    compatible_models = [
        candidate for candidate in model_candidates if candidate in IMAGE_SIZE_512_SUPPORTED_MODELS
    ]
    if compatible_models:
        return compatible_models

    tried = ", ".join(model_candidates)
    raise ValueError(
        "image_size 0.5K is currently only supported by Gemini 3.1 Flash Image models. "
        f"Current resolved model candidates: {tried}."
    )


def _resolve_image_model_candidates_for_thinking(
    model_candidates: List[str], thinking_level: str
) -> List[str]:
    if thinking_level == "default":
        return model_candidates

    compatible_models = [
        candidate
        for candidate in model_candidates
        if candidate in IMAGE_THINKING_LEVEL_SUPPORTED_MODELS
    ]
    if compatible_models:
        return compatible_models

    tried = ", ".join(model_candidates)
    raise ValueError(
        "thinking_level minimal/high is currently only supported by Gemini 3.1 Flash Image models. "
        "Gemini 3 Pro Image Preview uses its default thinking behavior and does not expose "
        "a thinking_level control in the Gemini API. "
        f"Current resolved model candidates: {tried}."
    )


def _run_with_model_fallback(model_candidates: List[str], request_fn):
    last_model_error = None
    for candidate in model_candidates:
        try:
            return request_fn(candidate)
        except Exception as exc:
            if _is_model_error(exc):
                last_model_error = exc
                continue
            raise

    tried = ", ".join(model_candidates)
    if last_model_error is not None:
        raise ValueError(
            f"None of the model candidates worked. Tried: {tried}. "
            f"Last model error: {last_model_error}"
        ) from last_model_error
    raise ValueError(f"No valid image model candidates. Tried: {tried}.")


def _tensor_to_part(image_tensor: torch.Tensor, media_resolution: Optional[str]) -> types.Part:
    array = np.clip(image_tensor[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    blob = types.Blob(mime_type="image/png", data=buffer.getvalue())
    kwargs = {"inline_data": blob}
    if media_resolution:
        kwargs["media_resolution"] = {"level": media_resolution}
    return types.Part(**kwargs)


def _bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    if isinstance(image_bytes, str):
        image_bytes = base64.b64decode(image_bytes)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    array = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


def _tensor_to_base64_jpeg(image_tensor: torch.Tensor, quality: int = 75) -> str:
    array = np.clip(image_tensor[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _gather_text_from_parts(parts: List[types.Part]) -> str:
    collected = []
    for part in parts:
        if getattr(part, "text", None):
            collected.append(part.text)
    return "".join(collected).strip()


AZIMUTH_MAP = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}

ELEVATION_MAP = {
    -30: "low-angle shot",
    0: "eye-level shot",
    30: "elevated shot",
    60: "high-angle shot",
}

DISTANCE_MAP = {
    0.6: "close-up",
    1.0: "medium shot",
    1.8: "wide shot",
}


def _snap_to_nearest(value: float, options: List[float]) -> float:
    return min(options, key=lambda x: abs(x - value))


def _normalize_azimuth(value: float) -> float:
    return float(value) % 360.0


def _build_camera_prompt(azimuth: float, elevation: float, distance: float) -> str:
    azimuth = _normalize_azimuth(azimuth)
    azimuth_snapped = _snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    elevation_snapped = _snap_to_nearest(float(elevation), list(ELEVATION_MAP.keys()))
    distance_snapped = _snap_to_nearest(float(distance), list(DISTANCE_MAP.keys()))
    return (
        f"{AZIMUTH_MAP[azimuth_snapped]} "
        f"{ELEVATION_MAP[elevation_snapped]} "
        f"{DISTANCE_MAP[distance_snapped]}"
    )


def _supports_field(model_cls, field_name: str) -> bool:
 

    try:
        model_fields = getattr(model_cls, "model_fields", None)
        if isinstance(model_fields, dict):
            return field_name in model_fields

        fields = getattr(model_cls, "__fields__", None)
        if isinstance(fields, dict):
            return field_name in fields

        dataclass_fields = getattr(model_cls, "__dataclass_fields__", None)
        if isinstance(dataclass_fields, dict):
            return field_name in dataclass_fields
    except Exception:
        return False

    return False


INT32_MAX = (2**31) - 1


def _random_seed_int32() -> int:
    # Gemini generation_config.seed is TYPE_INT32 (signed). Keep it in-range.
    return secrets.randbelow(INT32_MAX + 1)


def _normalize_seed_int32(seed: Optional[int], mode: str) -> int:
    parsed_seed = 0 if seed is None else int(seed)
    if mode == "random_if_negative" and parsed_seed < 0:
        return _random_seed_int32()
    if mode == "clamp":
        if parsed_seed < 0:
            return 0
        if parsed_seed > INT32_MAX:
            return INT32_MAX
        return parsed_seed
    return parsed_seed % (INT32_MAX + 1)


ALLOWED_ASPECTS = [
    ("1:1", 1.0),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:3", 4 / 3),
    ("4:5", 4 / 5),
    ("5:4", 5 / 4),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("21:9", 21 / 9),
    ("4:1", 4 / 1),
    ("1:4", 1 / 4),
    ("8:1", 8 / 1),
    ("1:8", 1 / 8),
]


def _auto_aspect_ratio(images: List[torch.Tensor]) -> str:
    for img in images:
        try:
            if img is None:
                continue
            h, w = int(img.shape[1]), int(img.shape[2])  # ComfyUI tensors: [B, H, W, C]
            if h <= 0 or w <= 0:
                continue
            ratio = w / h
            best = min(ALLOWED_ASPECTS, key=lambda x: abs(x[1] - ratio))
            return best[0]
        except Exception:
            continue
    return "1:1"




class AI_Gemini3_Img2T:

    @classmethod
    def INPUT_TYPES(cls):
        optional_images = {f"image_{idx}": ("IMAGE",) for idx in range(1, 11)}
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                #"api_url": ("STRING", {"default": "https://generativelanguage.googleapis.com/v1beta/openai/", "multiline": False, "tooltip": "留空=官方SDK | 支持服务商简写(apistudio/mmchat)或完整地址(自动拼接/v1/chat/completions)"}),
                "system_doc": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "Explain this image", "multiline": True}),
                "model": (TEXT_MODEL_OPTIONS, {"default": "gemini-3-flash-preview"}),
                "media_resolution": (
                    [
                        "auto",
                        "media_resolution_low",
                        "media_resolution_medium",
                        "media_resolution_high",
                    ],
                    {"default": "media_resolution_high"},
                ),
                "thinking_level": (
                    ["default", "low", "high"],
                    {"default": "default"},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": INT32_MAX}),
            },
            "optional": optional_images,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/AI_tool"

    def run(
        self,
        api_key: str,
        #api_url: str,
        system_doc: str,
        prompt: str,
        model: str,
        media_resolution: str,
        thinking_level: str,
        seed: int,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")

        api_url="https://generativelanguage.googleapis.com/v1beta/openai/"

        api_keys = _resolve_api_keys(api_key)
        normalized_api_url = _normalize_api_url(api_url)
        model_candidates = _resolve_text_model_candidates(model)
        resolved_model = model_candidates[0]
        api_mode = _detect_api_mode(normalized_api_url, node_type="text")

        # Collect reference images
        images = [
            image_1, image_2, image_3, image_4, image_5,
            image_6, image_7, image_8, image_9, image_10,
        ]

        parsed_seed = -1 if seed is None else int(seed)
        if parsed_seed < 0:
            resolved_seed = _random_seed_int32()
        elif parsed_seed > INT32_MAX:
            raise ValueError(f"Seed must be <= {INT32_MAX} (Gemini expects signed int32).")
        else:
            resolved_seed = parsed_seed

        # ---- Mode: OpenAI-compatible (default for text nodes with third-party URL) ----
        if api_mode in ("openai", "openai_default"):
            b64_images = []
            for img in images:
                if img is not None:
                    b64_images.append(_tensor_to_base64_jpeg(img))
            openai_key = api_keys[0] if api_keys else ""
            data = _run_with_model_fallback(
                model_candidates,
                lambda model_name: _call_openai_compatible(
                    normalized_api_url, openai_key, model_name, prompt,
                    image_parts_base64=b64_images if b64_images else None,
                    mode="chat",
                ),
            )
            text = _parse_openai_chat_response(data)
            return (text,)

        # ---- Mode: Gemini REST (official URL explicitly provided) ----
        if api_mode == "official":
            inline_parts = []
            for img in images:
                if img is not None:
                    b64 = _tensor_to_base64_jpeg(img)
                    inline_parts.append({
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": b64,
                        }
                    })
            rest_key = api_keys[0] if api_keys else ""
            data = _run_with_model_fallback(
                model_candidates,
                lambda model_name: _call_gemini_official_rest_with_field_fallback(
                    normalized_api_url, rest_key, model_name,
                    lambda exclude: _build_gemini_rest_body_text(
                        prompt, inline_parts,
                        system_instruction=(system_doc or "").strip(),
                        thinking_level=thinking_level,
                        seed=resolved_seed if parsed_seed >= 0 else None,
                        exclude_fields=exclude,
                    ),
                ),
            )
            _, text = _parse_gemini_rest_image_response(data)
            return (text,)

        # ---- Mode: SDK (default) ----
        resolution = None if media_resolution == "auto" else media_resolution
        parts: List[types.Part] = [types.Part.from_text(text=prompt)]

        for image in images:
            if image is not None:
                parts.append(_tensor_to_part(image, resolution))

        config_kwargs = {"response_modalities": ["TEXT"]}
        cleaned_system_doc = (system_doc or "").strip()
        if cleaned_system_doc:
            if not _supports_field(types.GenerateContentConfig, "system_instruction"):
                raise ValueError(
                    "The installed google-genai version does not support system_instruction. "
                    "Please update the dependency, for example: google-genai>=1.7.0,<2.0.0"
                )
            config_kwargs["system_instruction"] = cleaned_system_doc
        if thinking_level != "default":
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        if _supports_field(types.GenerateContentConfig, "seed"):
            config_kwargs["seed"] = resolved_seed

        def _request(client):
            return _run_with_model_fallback(
                model_candidates,
                lambda model_name: client.models.generate_content(
                    model=model_name,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(**config_kwargs),
                ),
            )

        response = _run_with_key_rotation(api_keys, _request, api_url=normalized_api_url)

        parts_out: List[types.Part] = []
        if getattr(response, "candidates", None):
            first_candidate = response.candidates[0]
            if first_candidate and first_candidate.content:
                parts_out = first_candidate.content.parts or []

        text = (response.text or _gather_text_from_parts(parts_out)).strip()
        return (text,)



class AI_Gemini3_ImageEdit:


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_url": ("STRING", {"default": "", "multiline": False, "tooltip": "留空=官方SDK | 支持服务商简写(apistudio/mmchat)或完整地址"}),
                "prompt": ("STRING", {"default": "Generate a cinematic landscape", "multiline": True}),
                "model": (IMAGE_MODEL_OPTIONS, {"default": "gemini-3-pro-image-preview"}),
                "aspect_ratio": (
                    [
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                        "4:1",
                        "1:4",
                        "8:1",
                        "1:8",
                    ],
                    {"default": "1:1"},
                ),
                "image_size": (
                    IMAGE_SIZE_OPTIONS,
                    {"default": "1K"},
                ),
                "thinking_level": (
                    IMAGE_THINKING_LEVEL_OPTIONS,
                    {"default": "default"},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": INT32_MAX}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
                "reference_image_6": ("IMAGE",),
                "reference_image_7": ("IMAGE",),
                "reference_image_8": ("IMAGE",),
                "reference_image_9": ("IMAGE",),
                "reference_image_10": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/AI_tool"

    def run(
        self,
        api_key: str,
        api_url: str,
        prompt: str,
        model: str,
        aspect_ratio: str,
        image_size: str,
        thinking_level: str,
        seed: int,
        reference_image=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_image_5=None,
        reference_image_6=None,
        reference_image_7=None,
        reference_image_8=None,
        reference_image_9=None,
        reference_image_10=None,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required.")

        api_keys = _resolve_api_keys(api_key)
        normalized_api_url = _normalize_api_url(api_url)
        model_candidates = _resolve_image_model_candidates(model)
        resolved_image_size = _normalize_image_size(image_size)
        model_candidates = _resolve_image_model_candidates_for_size(
            model_candidates, resolved_image_size
        )
        model_candidates = _resolve_image_model_candidates_for_thinking(
            model_candidates, thinking_level
        )

        reference_images = [
            reference_image,
            reference_image_2,
            reference_image_3,
            reference_image_4,
            reference_image_5,
            reference_image_6,
            reference_image_7,
            reference_image_8,
            reference_image_9,
            reference_image_10,
        ]

        chosen_aspect = aspect_ratio
        if aspect_ratio == "auto":
            chosen_aspect = _auto_aspect_ratio(reference_images)

        parsed_seed = -1 if seed is None else int(seed)
        if parsed_seed < 0:
            resolved_seed = _random_seed_int32()
        elif parsed_seed > INT32_MAX:
            raise ValueError(f"Seed must be <= {INT32_MAX} (Gemini expects signed int32).")
        else:
            resolved_seed = parsed_seed

        api_mode = _detect_api_mode(normalized_api_url, node_type="image")
        # Resolve actual model name for non-SDK modes (use first candidate)
        resolved_model = model_candidates[0] if model_candidates else model

        # ---- Mode: OpenAI-compatible (explicit /openai path or full OpenAI URL) ----
        if api_mode == "openai":
            b64_images = []
            for img in reference_images:
                if img is not None:
                    b64_images.append(_tensor_to_base64_jpeg(img))
            openai_key = api_keys[0] if api_keys else ""
            size_str = self._compute_openai_size(chosen_aspect, resolved_image_size)
            data = _call_openai_compatible(
                normalized_api_url, openai_key, resolved_model, prompt,
                size=size_str,
                image_parts_base64=b64_images if b64_images else None,
                mode="image",
            )
            image_bytes = _parse_openai_image_response(data)
            if image_bytes is None:
                raise ValueError("OpenAI-compatible API did not return an image.")
            image_tensor = _bytes_to_tensor(image_bytes)
            return (image_tensor, "")

        # ---- Mode: Gemini REST (official, gemini, or auto for image → always Gemini) ----
        if api_mode in ("official", "gemini"):
            inline_parts = []
            for img in reference_images:
                if img is not None:
                    b64 = _tensor_to_base64_jpeg(img)
                    inline_parts.append({
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": b64,
                        }
                    })
            rest_key = api_keys[0] if api_keys else ""
            data = _call_gemini_official_rest_with_field_fallback(
                normalized_api_url, rest_key, resolved_model,
                lambda exclude: _build_gemini_rest_body_image(
                    prompt, inline_parts,
                    aspect_ratio=chosen_aspect,
                    image_size=resolved_image_size,
                    thinking_level=thinking_level,
                    seed=resolved_seed if parsed_seed >= 0 else None,
                    exclude_fields=exclude,
                ),
            )
            image_bytes, text = _parse_gemini_rest_image_response(data)
            if image_bytes is None:
                raise ValueError("Gemini REST API did not return an image. Text response: %s" % text)
            image_tensor = _bytes_to_tensor(image_bytes)
            return (image_tensor, text)

        # ---- Mode: SDK (default) ----
        parts: List[types.Part] = [types.Part.from_text(text=prompt)]
        for image in reference_images:
            if image is not None:
                parts.append(_tensor_to_part(image, None))

        image_config_kwargs = {"aspect_ratio": chosen_aspect, "image_size": resolved_image_size}
        if _supports_field(types.ImageConfig, "seed"):
            image_config_kwargs["seed"] = resolved_seed
        image_config = types.ImageConfig(**image_config_kwargs)

        config_kwargs = {"response_modalities": ["IMAGE", "TEXT"], "image_config": image_config}
        if thinking_level != "default":
            if not _supports_field(types.GenerateContentConfig, "thinking_config"):
                raise ValueError(
                    "The installed google-genai version does not support thinking_config. "
                    "Please update the dependency, for example: google-genai>=1.7.0,<2.0.0"
                )
            if not _supports_field(types.ThinkingConfig, "thinking_level"):
                raise ValueError(
                    "The installed google-genai version does not support ThinkingConfig.thinking_level. "
                    "Please update the dependency, for example: google-genai>=1.7.0,<2.0.0"
                )
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )
        if _supports_field(types.GenerateContentConfig, "seed"):
            config_kwargs["seed"] = resolved_seed

        def _request(client):
            return _run_with_model_fallback(
                model_candidates,
                lambda resolved_model: client.models.generate_content(
                    model=resolved_model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(**config_kwargs),
                ),
            )

        response = _run_with_key_rotation(api_keys, _request, api_url=normalized_api_url)

        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        parts_out = candidate.content.parts if candidate and candidate.content else []
        text = (getattr(response, "text", "") or _gather_text_from_parts(parts_out)).strip()

        image_bytes = None
        for part in parts_out:
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                image_bytes = part.inline_data.data
                break

        if image_bytes is None:
            raise ValueError("Model did not return an image. Text response: %s" % text)

        image_tensor = _bytes_to_tensor(image_bytes)
        return (image_tensor, text)

    @staticmethod
    def _compute_openai_size(aspect_ratio: str, image_size: str) -> str:
        """Convert aspect_ratio + image_size to a WxH string for OpenAI-compatible APIs."""
        size_map = {"512": 512, "1K": 1024, "2K": 2048, "4K": 4096}
        long_side = size_map.get(image_size, 1024)
        ratio_map = {
            "1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2),
            "3:4": (3, 4), "4:3": (4, 3), "4:5": (4, 5),
            "5:4": (5, 4), "9:16": (9, 16), "16:9": (16, 9),
            "21:9": (21, 9), "4:1": (4, 1), "1:4": (1, 4),
            "8:1": (8, 1), "1:8": (1, 8),
        }
        rw, rh = ratio_map.get(aspect_ratio, (1, 1))
        # Scale so the long side matches
        if rw >= rh:
            w = long_side
            h = max(16, round(long_side * rh / rw / 16) * 16)
        else:
            h = long_side
            w = max(16, round(long_side * rw / rh / 16) * 16)
        return f"{w}x{h}"






