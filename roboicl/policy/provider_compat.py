"""Small Responses-API boundary used by the released RoboICL protocol."""

from __future__ import annotations

from copy import deepcopy
from urllib.parse import urlparse


RESPONSES = "responses"


def validate_api_mode(value: object) -> str:
    if value != RESPONSES:
        raise ValueError("The minimal release supports only api_mode=responses")
    return RESPONSES


def validate_endpoint(api_mode: str, endpoint: object) -> str:
    validate_api_mode(api_mode)
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("ASTRA_ENDPOINT must be a non-empty Responses API URL")
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("ASTRA_ENDPOINT must be an absolute HTTP(S) URL")
    if not parsed.path.rstrip("/").endswith("/responses"):
        raise ValueError("Responses API endpoint must end in /responses")
    return endpoint


def to_wire_payload(payload: dict, api_mode: str) -> dict:
    validate_api_mode(api_mode)
    return deepcopy(payload)


def from_wire_response(response: dict, api_mode: str) -> dict:
    validate_api_mode(api_mode)
    if not isinstance(response, dict):
        raise ValueError("Responses API returned a non-object payload")
    return response


def provider_cache_usage(usage: object) -> dict:
    """Normalize optional cache accounting without inventing missing values."""
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get("input_tokens")
    details = usage.get("input_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    cache_write_tokens = details.get("cache_write_tokens") if isinstance(details, dict) else None
    valid = (
        type(input_tokens) is int and input_tokens >= 0
        and type(cached_tokens) is int and cached_tokens >= 0
    )
    return {
        "input_tokens": input_tokens if type(input_tokens) is int else None,
        "cached_tokens": cached_tokens if type(cached_tokens) is int else None,
        "cache_write_tokens": cache_write_tokens if type(cache_write_tokens) is int else None,
        "cache_reporting_valid": valid,
    }
