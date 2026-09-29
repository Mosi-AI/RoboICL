"""Persistent text-only cache boundaries through completed fixed anchors."""


def observation_end(step):
    """A stable TRAIN/LIVE view delimiter, not an episode-completion signal."""
    if type(step) is not int or step < 0:
        raise ValueError('observation delimiter requires a nonnegative step')
    return {'type': 'input_text', 'text': f'<OBSERVATION_END step="{step}"/>'}


def annotate_anchored_breakpoints(inputs, fixed_length, stable_live_length):
    """Annotate a request COPY only within its immutable LIVE prefix.

    ``stable_live_length`` comes from trajectory rendering and ends after the
    latest fixed anchor chunk (or the initial observation). Complete retained
    observations and closed gaps within that prefix get text-only boundaries.
    A growing gap, rolling latest chunk, current nonfixed RGB and correction
    tail remain unmarked until a fixed anchor closes the history before them.
    """
    if (type(fixed_length) is not int or fixed_length < 0
            or type(stable_live_length) is not int or stable_live_length < 0
            or fixed_length + stable_live_length > len(inputs)):
        raise ValueError('cache frontier must be within the rendered request')
    positions = []
    for index in range(fixed_length, fixed_length + stable_live_length):
        item = inputs[index]
        content = item.get('content')
        if item.get('role') != 'user' or not isinstance(content, list) or not content:
            continue
        if content[-1].get('type') != 'input_text':
            raise ValueError('Anchored observation cache boundary must follow all RGB on a text footer')
        content[-1]['prompt_cache_breakpoint'] = {'mode': 'explicit'}
        positions.append({'input_index': index, 'content_index': len(content)-1})
    return positions
