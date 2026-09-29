"""TRAIN/LIVE trajectory gaps and deterministic, episode-local LIVE key chunks.

The anchor schedule is private policy bookkeeping. Only observed step intervals
and recorded content are rendered; no budget or future anchor reaches the model.
The caller retains its complete observation/action archive for audit/retrieval.
"""
from copy import deepcopy
import json


def omitted_interval(source, start, end, *, reason='unshown', summary=None,
                     episode_id=None):
    """Return the same explicit gap marker for TRAIN and compressed LIVE.

    Intervals are half-open action intervals: action steps [start, end) connect
    observations at start and end. Episode boundaries may use null bounds.
    A supplied summary is a report of intent/metadata, never a success label.
    """
    if (start is None) != (end is None):
        raise ValueError('both gap bounds must be provided or both omitted')
    if start is not None and (type(start) is not int or type(end) is not int
                              or not 0 <= start <= end):
        raise ValueError('gap bounds must be nonnegative ordered integers')
    payload = {
        'source': str(source),
        'reason': str(reason),
        'omitted_action_interval': [start, end],
        'interval_convention': '[start,end)',
        'notice': ('Intermediate actions and observations are not shown. '
                   'Do not infer continuity or task success across this gap.'),
    }
    if episode_id is not None:
        payload['episode_id'] = deepcopy(episode_id)
    if summary is not None:
        payload['summary'] = deepcopy(summary)
        payload['summary_scope'] = ('Reported action intent or execution metadata; '
                                    'not verified visual outcome or task success.')
    return {'role': 'user', 'content': [{
        'type': 'input_text',
        'text': '<TRAJECTORY_GAP>' + json.dumps(
            payload, ensure_ascii=False, separators=(',', ':')) + '</TRAJECTORY_GAP>',
    }]}


class LiveTrajectoryMemory:
    """Keep full chunks at a few uniform anchors; replace all other chunks.

    ``capture`` accepts only committed execution, with complete response items
    and matching receipt(s). Observation user messages are added by ``render``.
    Anchor membership uses start < anchor <= end, so the first observed endpoint
    reaching an anchor retains its preceding chunk. Anchor zero retains the
    first nonempty chunk starting at zero. A zero-step rejection never consumes
    an anchor. Rejected/unmatched current-turn items belong in ``tail``.
    """

    def __init__(self, count=5, fallback_interval=210, retain_latest=False):
        if type(count) is not int or count < 1:
            raise ValueError('anchor count must be a positive integer')
        if type(fallback_interval) is not int or fallback_interval < 1:
            raise ValueError('fallback interval must be a positive integer')
        if type(retain_latest) is not bool:
            raise ValueError('retain_latest must be a boolean')
        if retain_latest and count < 2:
            raise ValueError('retain_latest requires at least two retained chunks')
        self.count = count
        self.fallback_interval = fallback_interval
        self.retain_latest = retain_latest
        self._records = []
        self._anchors = ()
        self.configure()

    def configure(self, maximum=None):
        """Choose an internal anchor schedule before recording this episode."""
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise ValueError('maximum must be a positive integer or None')
        anchors = tuple(sorted(set(
            i * maximum // self.count if maximum is not None
            else i * self.fallback_interval
            for i in range(self.count - int(self.retain_latest)))))
        if self._records and anchors != self._anchors:
            raise ValueError('cannot change anchor schedule after capture')
        self._anchors = anchors

    @property
    def anchors(self):
        return self._anchors

    @property
    def stats(self):
        """Bounded audit metadata, not model-facing prompt content."""
        fixed = [r for r in self._records if r['retained']]
        latest = self._records[-1] if self.retain_latest and self._records else None
        retained = [r for r in self._records if r['retained'] or r is latest]
        return {
            'anchor_steps': list(self._anchors),
            'captured_chunks': len(self._records),
            'retained_chunks': len(retained),
            'omitted_chunks': len(self._records) - len(retained),
            'captured_through_step': self._records[-1]['end'] if self._records else 0,
            'retained_intervals': [[r['start'], r['end']] for r in retained],
            'fixed_chunks': len(fixed),
            'latest_interval': [latest['start'], latest['end']] if latest else None,
            'retain_latest': self.retain_latest,
            'max_retained_chunks': self.count,
        }

    def snapshot(self):
        return self.stats

    def capture(self, start, end, items, summary=None):
        """Record actual execution, returning whether it crosses a fixed anchor.

        Repeating an identical capture is idempotent. Inputs and renders are
        copied so callers cannot mutate the saved archive or closed gaps.
        """
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError('capture bounds must be nonnegative ordered integers')
        if start == end:
            return False
        if not isinstance(items, (list, tuple)):
            raise ValueError('items must be a list or tuple of response/receipt items')
        saved_items = deepcopy(list(items))
        saved_summary = deepcopy(summary)
        for record in self._records:
            if (record['start'], record['end']) == (start, end):
                if record['items'] != saved_items or record['summary'] != saved_summary:
                    raise ValueError('a committed interval cannot be replaced')
                return record['retained']
        if self._records and start < self._records[-1]['end']:
            raise ValueError('captured intervals must be chronological and nonoverlapping')
        retained = any(start < anchor <= end or (anchor == 0 and start == 0)
                       for anchor in self._anchors)
        self._records.append({
            'start': start, 'end': end, 'retained': retained,
            'items': saved_items, 'summary': saved_summary,
        })
        return retained

    def render(self, initial_observation, current_step, observation_fn, tail=None):
        """Build a request snapshot without changing any saved records."""
        return self.render_with_frontier(initial_observation, current_step,
                                         observation_fn, tail)[0]

    def render_with_frontier(self, initial_observation, current_step, observation_fn, tail=None):
        """Return request items and the length of their immutable LIVE prefix.

        The initial observation and current RGB are always present. Each key
        chunk is a start observation, original response/receipt items, and end
        observation. Adjacent key chunks share their boundary image exactly
        once. Consecutive omitted intervals share one gap, with each executed
        chunk's actual interval and unmodified intent/metadata kept in order.
        The open gap grows until another fixed chunk closes it. With
        ``retain_latest``, the latest completed chunk also remains full, but
        cannot close the gap permanently unless it crosses a fixed anchor.
        Explicit cache boundaries may only cover the initial observation and
        the prefix through the latest fixed chunk's ending observation.
        ``tail`` follows the current observation for retries.
        """
        if type(current_step) is not int or current_step < 0:
            raise ValueError('current_step must be a nonnegative integer')
        eligible = [record for record in self._records if record['end'] <= current_step]
        if (self.retain_latest and self._records and current_step > 0
                and (not eligible or eligible[-1]['end'] != current_step)):
            raise ValueError('retain_latest requires current observation at the latest committed chunk endpoint')
        latest = eligible[-1] if self.retain_latest and eligible else None
        result = [deepcopy(initial_observation)]
        stable_length = 1
        last_shown_step = 0
        previous_end = 0
        pending_gap = []

        def omit(start, end, summary=None, *, unrecorded=False):
            entry = {'action_interval': [start, end]}
            if unrecorded:
                entry['reason'] = 'unrecorded_interval'
            else:
                entry['execution'] = deepcopy(summary)
            pending_gap.append(entry)

        def flush_gap():
            if pending_gap:
                result.append(omitted_interval(
                    'LIVE', pending_gap[0]['action_interval'][0],
                    pending_gap[-1]['action_interval'][1],
                    reason='compressed_history', summary={'chunks': pending_gap}))
                pending_gap.clear()

        def show(step):
            nonlocal last_shown_step
            if last_shown_step != step:
                result.append(deepcopy(observation_fn(step)))
                last_shown_step = step

        for record in eligible:
            start, end = record['start'], record['end']
            if previous_end < start:
                omit(previous_end, start, unrecorded=True)
            if record['retained'] or record is latest:
                flush_gap()
                show(start)
                result.extend(deepcopy(record['items']))
                show(end)
                if record['retained']:
                    stable_length = len(result)
            else:
                omit(start, end, record['summary'])
            previous_end = end
        if previous_end < current_step:
            omit(previous_end, current_step, unrecorded=True)
        flush_gap()
        show(current_step)
        result.extend(deepcopy(list(tail or [])))
        return result, stable_length
