"""Conservative admission bounds for paired-frame anchored image history.

Count TRAIN images after assembling the actual prefix. LIVE needs at most two
observations per retained action chunk plus the current observation, unless one
chunk slot reserves the latest completed chunk whose endpoint is already current.
The initial observation is the first key chunk's start, never an extra image.
The bound assumes no endpoint sharing between later key chunks and does not
predict early completion or depend on the number of executed steps.
"""


class ImageBudgetExceeded(ValueError):
    """A valid image configuration cannot fit the configured hard image cap."""


def _observation_image_count(cameras, image_layout, *, name='cameras'):
    if not isinstance(cameras, (tuple, list)) or not cameras:
        raise ValueError(f'{name} must be a nonempty tuple or list of camera names')
    if any(not isinstance(camera, str) or not camera.strip() for camera in cameras):
        raise ValueError(f'{name} must contain nonempty camera names')
    if len(set(cameras)) != len(cameras):
        raise ValueError(f'{name} must not contain duplicate camera names')
    if image_layout not in ('triptych', 'separate'):
        raise ValueError('image_layout must be triptych or separate')
    if image_layout == 'triptych' and len(cameras) != 3:
        raise ValueError('triptych image layout requires three cameras')
    return 1 if image_layout == 'triptych' else len(cameras)


def reference_image_count(reference, *, cameras, frame0_cameras=(), image_layout='triptych',
                          include_result_observations=True):
    """Count the tool-call TRAIN prefix from manifest metadata, without I/O.

    Bundles are counted per episode, ignoring their redundant flattened view.
    Every sampled chunk has its start image(s), and optionally its endpoint.
    A terminal observation is additional unless that final endpoint is already
    shown. Headers, episode boundaries and calibration text add no images.
    """
    if reference is None:
        return 0
    if not isinstance(reference, dict):
        raise ValueError('reference must be a manifest dictionary or None')
    if type(include_result_observations) is not bool:
        raise ValueError('include_result_observations must be a boolean')
    normal = _observation_image_count(cameras, image_layout)
    if not isinstance(frame0_cameras, (tuple, list)):
        raise ValueError('frame0_cameras must be a tuple or list of camera names')
    initial = (_observation_image_count(frame0_cameras, image_layout, name='frame0_cameras')
               if frame0_cameras else normal)
    if 'episodes' in reference:
        episodes = reference['episodes']
        if not isinstance(episodes, (list, tuple)):
            raise ValueError('reference episodes must be a list or tuple')
        return sum(reference_image_count(
            episode, cameras=cameras, frame0_cameras=frame0_cameras,
            image_layout=image_layout, include_result_observations=include_result_observations,
        ) for episode in episodes)
    examples = reference.get('examples')
    if not isinstance(examples, (list, tuple)):
        raise ValueError('reference examples must be a list or tuple')
    count, last_result = 0, None
    for example in examples:
        if not isinstance(example, dict) or type(example.get('frame')) is not int or example['frame'] < 0:
            raise ValueError('reference examples require nonnegative integer frames')
        count += initial if example['frame'] == 0 else normal
        if include_result_observations:
            last_result = example.get('result_frame')
            if type(last_result) is not int or last_result <= example['frame']:
                raise ValueError('reference example requires a later integer result_frame')
            count += normal
    terminal = reference.get('terminal_observation')
    if terminal is not None:
        if not isinstance(terminal, dict) or type(terminal.get('frame')) is not int or terminal['frame'] < 0:
            raise ValueError('terminal_observation requires a nonnegative integer frame')
        if not include_result_observations or terminal['frame'] != last_result:
            count += normal
    return count


def anchored_image_budget(train_images, anchor_count, live_cameras, image_layout, *, retain_latest=False):
    """Return a whole-episode request-image upper bound from static settings.

    Image counts refer to input_image content parts, not triptych source panels.
    This bound requires the LIVE renderer to retain at most ``anchor_count``
    complete key chunks and to omit images from all other completed chunks.
    ``retain_latest`` reserves one of those slots for the latest completed chunk,
    sharing its endpoint with current RGB and reducing the bound from 2K+1 to 2K.
    """
    if type(train_images) is not int or train_images < 0:
        raise ValueError('train_images must be a nonnegative integer')
    if type(anchor_count) is not int or anchor_count < 1:
        raise ValueError('anchor_count must be a positive integer')
    if type(retain_latest) is not bool:
        raise ValueError('retain_latest must be a boolean')
    if retain_latest and anchor_count < 2:
        raise ValueError('retain_latest requires at least two retained chunks')
    images_per_observation = _observation_image_count(live_cameras, image_layout, name='live_cameras')
    # The reserved latest chunk's endpoint is current, so no extra current frame.
    live_max_images = (2 * anchor_count + (0 if retain_latest else 1)) * images_per_observation
    return {
        'train_images': train_images,
        'live_max_images': live_max_images,
        'max_total_images': train_images + live_max_images,
    }


def assert_image_budget(train_images, anchor_count, live_cameras, image_layout, *, max_images,
                        retain_latest=False):
    """Admit a configuration only when its full-episode bound fits the cap.

    Return the bound and ``image_limit`` for logging. Reject before a rollout/API request rather
    than silently dropping TRAIN examples or LIVE key chunks to fit the cap.
    """
    if type(max_images) is not int or max_images < 1:
        raise ValueError('max_images must be a positive integer')
    budget = anchored_image_budget(train_images, anchor_count, live_cameras, image_layout,
                                   retain_latest=retain_latest)
    budget['image_limit'] = max_images
    if budget['max_total_images'] > max_images:
        raise ImageBudgetExceeded(
            'Anchored image budget exceeds cap: '
            f"TRAIN {budget['train_images']} + LIVE worst-case {budget['live_max_images']} "
            f"= {budget['max_total_images']} images > max_images={max_images}. "
            'Preserve the configured key chunks; reduce TRAIN shots or change the image layout.'
        )
    return budget
