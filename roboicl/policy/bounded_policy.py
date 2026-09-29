"""Fixed-window, fail-closed Astra development policy.

Budget counters are run-scoped, deliberately NOT reset by episode reset.
Token reservation is a local estimate, not a guarantee about provider billing.
"""
import base64
import json
import hashlib
import http.client
import math
import os
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from uuid import uuid4
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from .astra_policy import Model as LegacyModel, validate_actions, CAMERAS
from .observation_images import small_image
from .provider_compat import (
    RESPONSES, from_wire_response, to_wire_payload,
)


MAX_RETRY_AFTER_S = 60.0


class BudgetExceeded(RuntimeError):
    pass


class RequestTransportError(RuntimeError):
    """HTTP/network failure, not an account quota or local budget verdict."""
    def __init__(self, message, *, retryable=False, error_code=None, recovery=None,
                 retry_after_s=None, max_attempts=None):
        super().__init__(message)
        self.retryable = retryable
        self.error_code = error_code
        # A payload rejection is retryable only when the caller applies this
        # explicit transformation. Re-sending the rejected bytes is pointless.
        self.recovery = recovery
        self.retry_after_s = retry_after_s
        self.max_attempts = max_attempts


class ProviderResponseError(ValueError):
    """A terminal SSE error, optionally safe for bounded retransmission."""
    retryable = False

    def __init__(self, message, *, retryable=None, error_code=None,
                 recovery=None, max_attempts=None):
        super().__init__(message)
        if retryable is not None:
            self.retryable = retryable
        self.error_code = error_code
        self.recovery = recovery
        self.max_attempts = max_attempts


class IncompleteResponseError(ProviderResponseError):
    """The provider ended an SSE request with response.incomplete."""
    retryable = True


class FailedResponseError(ProviderResponseError):
    """The provider ended an SSE request with response.failed."""
    retryable = True


class TransientResponseError(ProviderResponseError):
    """The provider emitted a top-level transient SSE error."""
    retryable = True


class TruncatedResponseError(ProviderResponseError):
    """The SSE stream closed before a completed response was accepted."""
    retryable = True


def _provider_error_is_transient(details):
    if not isinstance(details, dict):
        return False
    fields = ' '.join(str(details.get(key, '')).lower()
                      for key in ('type', 'code', 'reason'))
    return any(marker in fields for marker in (
        'rate_limit', 'too_many_requests', 'server_error', 'overloaded',
        'temporarily_unavailable', 'service_unavailable', 'provider_capacity',
        'timeout', 'timed_out',
    ))


def _provider_error_is_deterministic(details):
    if not isinstance(details, dict):
        return False
    fields = ' '.join(str(details.get(key, '')).lower()
                      for key in ('type', 'code', 'reason'))
    return any(marker in fields for marker in (
        'content_filter', 'content_policy', 'invalid_request',
        'invalid_encrypted_content', 'max_output_tokens', 'max_tokens', 'unsupported_value',
        'insufficient_quota', 'billing_hard_limit',
    ))


def _provider_error_code(details, fallback):
    if not isinstance(details, dict):
        return fallback
    value = details.get('code') or details.get('type') or details.get('reason')
    return str(value)[:128] if value else fallback


def _http_error_details(error_text):
    """Extract bounded routing fields; never trust or replay provider prose."""
    try:
        parsed = json.loads(error_text)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    details = parsed.get('error', parsed)
    if not isinstance(details, dict):
        return None, None
    code = details.get('code') or details.get('type')
    param = details.get('param')
    return (str(code)[:128] if code is not None else None,
            str(param)[:256] if param is not None else None)


def _retry_after_seconds(headers):
    if not headers:
        return None
    value = headers.get('retry-after') or headers.get('Retry-After')
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    # Do not let an untrusted response bypass the run's bounded retry policy.
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S)


def _is_request_timeout(exc):
    """Recognize equivalent timeout failures from urllib and curl."""
    if isinstance(exc, TimeoutError):
        return True
    return (isinstance(exc, urllib.error.URLError)
            and isinstance(getattr(exc, 'reason', None), TimeoutError))


def _raise_provider_terminal_failure(kind, event, response=None, source='SSE'):
    """Classify a failed/incomplete Responses result for every HTTP encoding."""
    if response is None:
        nested = event.get('response') if isinstance(event, dict) else None
        response = nested if isinstance(nested, dict) else {}
    details = (event.get('error') or response.get('error')
               or response.get('incomplete_details')
               or event.get('incomplete_details') or event)
    recovery = None
    max_attempts = None
    if (kind == 'response.incomplete' and isinstance(details, dict)
            and details.get('reason') == 'content_filter'):
        # A filtered incomplete result never becomes an executable action.
        # Permit bounded retransmission of the same request; the dialogue's
        # configured transport limit still applies, including a limit of one.
        retryable = True
        max_attempts = 3
    elif _provider_error_is_transient(details):
        retryable = True
    elif _provider_error_is_deterministic(details):
        retryable = False
    else:
        # Some compatible gateways provide only the terminal status/event.
        # Give an unclassifiable result one retry, regardless of wire format.
        retryable = True
        max_attempts = 2
    error_code = _provider_error_code(details, kind.replace('.', '_'))
    if error_code == 'invalid_encrypted_content':
        retryable = True
        recovery = 'drop_reasoning_items'
        max_attempts = 2
    elif (isinstance(details, dict) and (
            error_code == 'invalid_comparison_response_id'
            or details.get('param') == 'prompt_cache_options.comparison_response_id')):
        # comparison_response_id is only a cache diagnostic hint. A stale or
        # unknown ID can be removed without changing the task dialogue.
        retryable = True
        recovery = 'drop_comparison_response_id'
        max_attempts = 2
    if isinstance(details, dict):
        details = {k: details[k] for k in
                   ('type', 'code', 'reason', 'message', 'param') if k in details}
    error_type = {
        'response.incomplete': IncompleteResponseError,
        'response.failed': FailedResponseError,
    }[kind]
    raise error_type(f'{source} terminal failure: {kind}; '
                     + json.dumps(details, ensure_ascii=False)[:2000],
                     retryable=retryable, error_code=error_code,
                     recovery=recovery, max_attempts=max_attempts)


def _validate_json_response(response):
    """Reject HTTP-200 JSON terminal failures before usage/action handling."""
    if not isinstance(response, dict):
        raise TypeError('Responses JSON body must be an object')
    status = response.get('status')
    if status in ('failed', 'incomplete'):
        _raise_provider_terminal_failure(
            f'response.{status}', response, response=response, source='JSON')
    nested = response.get('response')
    kind = response.get('type')
    if kind in ('response.failed', 'response.incomplete'):
        _raise_provider_terminal_failure(
            kind, response,
            response=nested if isinstance(nested, dict) else {}, source='JSON')
    if isinstance(nested, dict) and nested.get('status') in ('failed', 'incomplete'):
        _raise_provider_terminal_failure(
            f"response.{nested['status']}", response,
            response=nested, source='JSON')
    return response


def read_sse_response(reply, record, deadline):
    """Only a complete terminal response can become an executable action."""
    data, total = [], 0
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('SSE response deadline exceeded')
        line = reply.readline()
        total += len(line)
        if not line:
            raise TruncatedResponseError(
                'SSE ended without response.completed',
                error_code='provider_stream_truncated', max_attempts=2,
            )
        line = line.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif not line and data:
            raw, data = '\n'.join(data), []
            if raw == '[DONE]':
                raise TruncatedResponseError(
                    'SSE DONE without response.completed',
                    error_code='provider_stream_truncated', max_attempts=2,
                )
            event = json.loads(raw)
            kind = event.get('type')
            if kind in ('response.created', 'response.completed', 'response.failed', 'response.incomplete', 'error'):
                record('sse_event', event_type=kind, response_id=event.get('response', {}).get('id'))
            if kind == 'response.completed':
                result = event['response']
                if result.get('status') != 'completed':
                    raise ValueError('Contradictory SSE completion status')
                return result
            if kind in ('response.failed', 'response.incomplete', 'error'):
                response = event.get('response', {})
                if kind in ('response.failed', 'response.incomplete'):
                    _raise_provider_terminal_failure(kind, event, response=response)
                details = (event.get('error') or response.get('error')
                           or response.get('incomplete_details') or event)
                retryable = _provider_error_is_transient(details)
                recovery = None
                max_attempts = None
                error_code = _provider_error_code(details, kind.replace('.', '_'))
                if error_code == 'invalid_encrypted_content':
                    retryable = True
                    recovery = 'drop_reasoning_items'
                    max_attempts = 2
                elif (isinstance(details, dict) and (
                        error_code == 'invalid_comparison_response_id'
                        or details.get('param') == 'prompt_cache_options.comparison_response_id')):
                    # comparison_response_id is only a cache diagnostic hint.
                    # A stale/unknown ID can be removed without changing the
                    # task dialogue or replaying any robot action.
                    retryable = True
                    recovery = 'drop_comparison_response_id'
                    max_attempts = 2
                if isinstance(details, dict):
                    details = {k: details[k] for k in ('type', 'code', 'reason', 'message', 'param') if k in details}
                error_type = TransientResponseError if retryable else ProviderResponseError
                raise error_type(f'SSE terminal failure: {kind}; '
                                 + json.dumps(details, ensure_ascii=False)[:2000],
                                 retryable=retryable, error_code=error_code,
                                 recovery=recovery, max_attempts=max_attempts)


class Model(LegacyModel):
    def _restore_offline_reference_checks(self):
        """Restore a persisted audit after the RPC server resets the model.

        The WebSocket harness calls ``reset()`` at the beginning of every
        episode.  ``LegacyModel.reset`` intentionally clears episode state,
        including ``train_reference_checks``; for a portable TRAIN bundle the
        original HDF5 may be absent, so the checked verification sidecar must
        be restored after that reset instead of triggering a source reread.
        """
        from .train_reference_bundle import is_reference_bundle
        if not getattr(self, 'demo', None) or not is_reference_bundle(self.demo):
            return
        data_root = Path(os.environ.get('ROBOICL_DATA_ROOT', '.')).expanduser()
        sources = []
        for item in self.demo.get('episodes', []):
            source = Path(item.get('source_file', '')).expanduser()
            sources.append(source if source.is_absolute() else data_root / source)
        if not sources or not any(not source.is_file() for source in sources):
            return
        demo_path = os.environ.get('ASTRA_DEMO_MANIFEST')
        if not demo_path:
            return
        verification_path = Path(demo_path).with_name('verification.json')
        if not verification_path.is_file():
            return
        candidate = json.loads(verification_path.read_text(encoding='utf-8'))
        if (candidate.get('passed') is True
                and candidate.get('source_sha256') == self.demo.get('sha256')):
            self.train_reference_checks = candidate

    def reset(self):
        super().reset()
        self._restore_offline_reference_checks()

    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_chunk = 8
        self.max_output = 1536
        self.effort = os.environ.get('ASTRA_REASONING_EFFORT', 'medium')
        self.request_count = 0
        self.reported_tokens = 0
        self.reserved_tokens = 0
        self.pending_reserved_tokens = 0
        self.started = time.monotonic()
        self.budget_poisoned = False
        self.max_requests = 4
        self.max_total_tokens = 60000
        self.max_seconds = 600
        self.demo = None
        demo_path = os.environ.get('ASTRA_DEMO_MANIFEST')
        if demo_path:
            self.demo = json.loads(Path(demo_path).read_text(encoding='utf-8'))
            from roboicl.config import supported_task
            if not supported_task(self.demo.get('task')):
                raise ValueError('This smoke demonstration requires a supported task')
            from .train_reference_bundle import is_reference_bundle, validate_bundle
            if is_reference_bundle(self.demo):
                if (not hasattr(self, 'dialogue_config')
                        or self.dialogue_config.demo_message_format != 'tool_calls'):
                    raise ValueError('TRAIN bundles require the dialogue harness with tool-call demonstrations')
                validate_bundle(self.demo)
                # Prefer a persisted source audit only for portable bundles
                # whose original HDF5 is unavailable.  When every source is
                # local, dialogue_policy re-reads and verifies the HDF5.
                self._restore_offline_reference_checks()
            elif len(self.demo['examples']) != 5:
                raise ValueError('Exactly five demo examples required')
            if self.demo.get('variant') == 'train_action_chunks_world_ee_delta_v1' and not hasattr(self, 'dialogue_config'):
                raise ValueError('Action tensor demonstrations require the dialogue harness; bounded mode would omit actions')
    small_image = staticmethod(small_image)

    def _tools(self):
        tool = next(t for t in super()._tools() if t['name'] == 'act')
        tool['parameters']['properties']['memory']['maxLength'] = 800
        action = tool['parameters']['properties']['actions']['items']['properties']
        action['mode']['enum'] = ['ee_delta']
        for arm in ('left', 'right'):
            action[arm]['minItems'] = action[arm]['maxItems'] = 6
        return [tool]

    def build_payload(self):
        raise RuntimeError(
            'The legacy bounded prompt entrypoint was removed; use '
            'roboicl.policy.dialogue_policy.Model'
        )

    def payload_metrics(self, payload):
        # Count EVERYTHING textual, including schema/instructions, not base64.
        images = []
        def scrub(value):
            if isinstance(value, dict):
                if value.get('type') == 'input_image':
                    images.append(value)
                    return {'type': 'input_image', 'detail': value['detail']}
                return {k: scrub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [scrub(v) for v in value]
            return value
        text_bytes = len(json.dumps(scrub(payload), ensure_ascii=False).encode('utf-8'))
        wire_bytes = len(json.dumps(payload).encode('utf-8'))
        if len(images) > 20 or text_bytes > 16000 or wire_bytes > 2000000:
            raise BudgetExceeded('Fixed payload limit exceeded; no request sent')
        for item in images:
            if item['detail'] != 'low':
                raise BudgetExceeded('Smoke only allows low-detail images')
            with Image.open(BytesIO(base64.b64decode(item['image_url'].split(',', 1)[1]))) as pic:
                if max(pic.size) > 384:
                    raise BudgetExceeded('Image dimension exceeds smoke bound')
        return {'text_bytes': text_bytes, 'image_count': len(images), 'wire_bytes': wire_bytes,
                'reserved_tokens': text_bytes + len(images) * 1024 + self.max_output,
                'reservation_note': 'Conservative local estimate, not provider tokenization or monetary cap'}

    def _time_limit_reached(self):
        return self.max_seconds is not None and time.monotonic() - self.started >= self.max_seconds

    def _request(self, payload):
        payload = dict(payload)
        # RESPONSES is also the compatibility default for narrowly constructed
        # offline fixtures that bypass __init__.
        api_mode = getattr(self, 'api_mode', RESPONSES)
        if os.environ.get('ASTRA_STREAM', '0') == '1':
            payload['stream'] = True
        metrics = self.payload_metrics(payload)
        reserve = metrics['reserved_tokens']
        if (self.budget_poisoned or (self.max_requests is not None and self.request_count >= self.max_requests) or
            self._time_limit_reached() or
            (self.max_total_tokens is not None and self.reported_tokens + self.pending_reserved_tokens + reserve > self.max_total_tokens)):
            raise BudgetExceeded('Run-scoped smoke budget exhausted; no request sent')
        self.request_count += 1
        self.reserved_tokens += reserve
        self.pending_reserved_tokens += reserve
        self._log('budget.jsonl', {'event': 'reserved', 'request': self.request_count, **metrics,
                                  'cumulative_reserved_tokens': self.reserved_tokens})
        mode = os.environ.get('ASTRA_NETWORK_MODE', 'system')
        if mode not in ('system', 'direct'):
            self.budget_poisoned = True
            raise ValueError('ASTRA_NETWORK_MODE must be system or direct')
        wire_payload = to_wire_payload(payload, api_mode)
        body = json.dumps(wire_payload).encode()
        client_id = str(uuid4())
        authorization = f'Bearer {self.api_key}'
        request = urllib.request.Request(self.endpoint, data=body,
                  headers={'Authorization': authorization, 'Content-Type': 'application/json',
                           'X-Client-Request-Id': client_id})
        remaining = getattr(self, 'request_timeout_s', 120)
        if self.max_seconds is not None:
            run_remaining = self.max_seconds - (time.monotonic() - self.started)
            remaining = run_remaining if remaining is None else min(remaining, run_remaining)
        remaining = max(1, remaining) if remaining is not None else None
        started = time.monotonic()
        diagnostic = {'request': self.request_count, 'client_request_id': client_id,
                      'network_mode': mode, 'http_backend': 'python',
                      'api_mode': api_mode, 'stream': wire_payload.get('stream', False),
                      'started_at': datetime.now(timezone.utc).isoformat(),
                      'timeout_s': remaining, 'request_bytes': len(body),
                      'payload_sha256': hashlib.sha256(body).hexdigest(), 'retry': False}
        self._log('requests.jsonl', {**diagnostic, 'payload': payload})
        def record(event, **fields):
            self._log('http_transport.jsonl', {**diagnostic, 'event': event,
                      'elapsed_s': time.monotonic()-started, **fields})
        try:
            # No global proxy mutation; direct is explicit and per request only.
            open_request = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                            if mode == 'direct' else urllib.request.urlopen)
            with open_request(request, timeout=remaining) as reply:
                headers = getattr(reply, 'headers', {})
                record('headers', http_status=getattr(reply, 'status', None),
                       response_headers={k: headers.get(k) for k in
                           ('x-request-id', 'x-oneapi-request-id', 'cf-ray', 'server', 'date')
                           if headers.get(k)})
                if 'text/event-stream' in headers.get('Content-Type', ''):
                    response = read_sse_response(
                        reply, record, None if remaining is None else started+remaining)
                else:
                    response = _validate_json_response(
                        from_wire_response(json.load(reply), api_mode))
            record('completed', response_id=response.get('id'), response_status=response.get('status'))
        except urllib.error.HTTPError as exc:
            self.budget_poisoned = True
            # Bound and redact error text; never log Authorization or arbitrary headers.
            try:
                error_text = exc.read(2048).decode('utf-8', errors='replace').replace(self.api_key, '[REDACTED]')
            except Exception:
                error_text = '[error body unavailable]'
            provider_code, provider_param = _http_error_details(error_text)
            retryable = exc.code in (408, 409, 429) or 500 <= exc.code <= 599
            max_attempts = 2 if exc.code == 409 else None
            if _provider_error_is_deterministic({'code': provider_code}):
                retryable = False
            recovery = None
            if (exc.code == 400 and provider_code == 'invalid_encrypted_content'
                    and any(isinstance(item, dict) and item.get('type') == 'reasoning'
                            for item in payload.get('input', []))):
                # The server rejected an opaque provider-issued reasoning item.
                # Retrying identical bytes cannot repair it; DialogueModel may
                # make one bounded retry without opaque reasoning history.
                retryable = True
                recovery = 'drop_reasoning_items'
                max_attempts = 2
            options = payload.get('prompt_cache_options')
            comparison_rejected = (
                provider_code == 'invalid_comparison_response_id'
                or provider_param == 'prompt_cache_options.comparison_response_id'
            )
            if (exc.code == 400 and comparison_rejected
                    and isinstance(options, dict)
                    and options.get('comparison_response_id')):
                retryable = True
                recovery = 'drop_comparison_response_id'
                max_attempts = 2
            retry_after_s = _retry_after_seconds(exc.headers)
            record('http_error', exception_type=type(exc).__name__, http_status=exc.code,
                   error_body=error_text, request_id=exc.headers.get('x-request-id') if exc.headers else None,
                   provider_error_code=provider_code, provider_error_param=provider_param,
                   retryable=retryable, recovery=recovery, retry_after_s=retry_after_s)
            self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(exc).__name__,
                                      'retryable': retryable, 'recovery': recovery,
                                      'max_attempts': max_attempts})
            from .eval_failure import safe_error_code
            code = safe_error_code(f'HTTP {exc.code} {error_text}')
            disposition = recovery or ('retryable' if retryable else 'no retry')
            raise RequestTransportError(
                f'HTTP {exc.code}; error_code={code}; client_request_id={client_id}; {disposition}',
                retryable=retryable, error_code=code, recovery=recovery,
                retry_after_s=retry_after_s,
                max_attempts=max_attempts,
            ) from None
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            self.budget_poisoned = True
            timed_out = _is_request_timeout(exc)
            error_code = 'timeout' if timed_out else 'connection_error'
            record('request_timeout' if timed_out else 'connection_error',
                   exception_type=type(exc).__name__,
                   exception_message=str(exc).replace(self.api_key, '[REDACTED]')[:1000],
                   traceback=traceback.format_exc().replace(self.api_key, '[REDACTED]')[-6000:],
                   outcome='unknown; request may have reached provider')
            self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(exc).__name__,
                                      'retryable': True, 'error_code': error_code})
            raise RequestTransportError(f'{type(exc).__name__}; error_code={error_code}; '
                                        f'client_request_id={client_id}; outcome unknown',
                                        retryable=True, error_code=error_code) from None
        except (ValueError, TypeError, AttributeError) as exc:
            self.budget_poisoned = True
            retryable = bool(getattr(exc, 'retryable', False))
            record('invalid_response', exception_type=type(exc).__name__,
                   exception_message=str(exc).replace(self.api_key, '[REDACTED]')[:2500],
                   retryable=retryable)
            from .eval_failure import safe_error_code
            # Provider error codes are useful for internal retry routing, but
            # only the allowlisted/normalized code may cross the RPC boundary.
            code = safe_error_code(exc)
            recovery = getattr(exc, 'recovery', None)
            max_attempts = getattr(exc, 'max_attempts', None)
            self._log('budget.jsonl', {'event': 'attempt_failed', 'type': type(exc).__name__,
                                      'retryable': retryable, 'error_code': code,
                                      'recovery': recovery, 'max_attempts': max_attempts})
            disposition = 'retryable' if retryable else 'no retry'
            raise RequestTransportError(
                f'Invalid API response; error_code={code}; client_request_id={client_id}; {disposition}',
                retryable=retryable, error_code=code, recovery=recovery,
                max_attempts=max_attempts,
            ) from None
        try:
            usage = response.get('usage') or {}
            values = [usage.get('input_tokens'), usage.get('output_tokens')]
            if any(type(n) is not int or n < 0 for n in values):
                raise ValueError('Missing/invalid provider usage')
            self.reported_tokens += sum(values)
            # A completed request is settled at provider-reported usage. Keep the
            # cumulative estimate for audit, but do not charge unused reservations.
            # Unknown/failed requests never reach this line and remain fail-closed.
            self.pending_reserved_tokens -= reserve
            self._log('budget.jsonl', {'event': 'response', 'request': self.request_count,
                      'usage': usage, 'cumulative_reported_tokens': self.reported_tokens,
                      'pending_reserved_tokens': self.pending_reserved_tokens,
                      'released_reservation_tokens': reserve})
            if response.get('error') or (self.max_total_tokens is not None and self.reported_tokens > self.max_total_tokens):
                raise BudgetExceeded('Provider error or reported usage exceeds smoke budget')
            return response
        except Exception as exc:
            self.budget_poisoned = True
            self._log('budget.jsonl', {'event': 'stopped', 'type': type(exc).__name__,
                                      'classification': 'usage_validation_or_budget', 'retry': False})
            raise BudgetExceeded('Usage validation/budget check failed; no action dispatched') from None

    def get_action(self):
        raise RuntimeError(
            'The legacy bounded prompt entrypoint was removed; use '
            'roboicl.policy.dialogue_policy.Model'
        )
