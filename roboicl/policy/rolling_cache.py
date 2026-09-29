"""Persistent explicit cache frontiers for rolling RGB history (no prompt edits)."""


def annotate_history_breakpoints(inputs, archive, visual, trim_count, every, fixed_length):
    """Annotate a pruned request copy, retaining all established read boundaries.

    Cache the last supported text block before the first retained image. That
    text survives when the image is replaced on the next pruning event. Keep
    all previous frontiers: removing older markers caused DMX to fall back to
    the fixed prefix in the integration probe. Lookup boundaries are not the
    same as the provider's four cache-write slots. Never rotate old markers.
    Derivation from archive positions is deterministic across retries/builds.
    """
    positions = []
    for count in range(every, trim_count + 1, every):
        if count <= 0 or count >= len(visual):
            continue
        frontier = visual[count]
        image_index = next(j for j, part in enumerate(archive[frontier]['content'])
                           if part.get('type') == 'input_image')
        position = None
        for i in range(frontier, fixed_length - 1, -1):
            item = inputs[i]
            content = item.get('content')
            if item.get('role') not in ('user', 'developer') or not isinstance(content, list):
                continue
            limit = image_index if i == frontier else len(content)
            for j in range(limit - 1, -1, -1):
                if content[j].get('type') == 'input_text':
                    position = (i, j)
                    break
            if position is not None:
                break
        if position is not None and position not in positions:
            positions.append(position)
    for i, j in positions:
        inputs[i]['content'][j]['prompt_cache_breakpoint'] = {'mode': 'explicit'}
    return [{'input_index': i, 'content_index': j} for i, j in positions]
