"""Checked, opt-in TRAIN episodes with action chunks and one final image.

The final observation has no fabricated action or execution receipt. Episode
boundaries and source-specific calibrations are preserved by the dialogue layer.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np


BUNDLE_VARIANT = 'train_reference_bundle_v2'
CHILD_VARIANT = 'train_action_chunks_world_ee_delta_v1'
COMMON_FIELDS = ('task', 'instruction', 'action_space', 'delta_frame', 'frequency', 'prompt_image_encoding')


def is_reference_bundle(demo):
    return isinstance(demo, dict) and demo.get('variant') == BUNDLE_VARIANT


def bundle_digest(demo):
    """Content address the complete bundle, excluding only its own digest field."""
    content = {key: value for key, value in demo.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def portable_bundle_digest(demo):
    """Hash bundle content independently of the data-root-relative source locator."""
    content = deepcopy(demo)
    content.pop('sha256', None)
    for episode in content.get('episodes', []):
        episode['source_file'] = 'sha256:' + episode['sha256']
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_bundle(demo):
    """Reject ambiguous episode boundaries, stale flattened views, or bad labels."""
    from .observation_images import image_encoding
    from .astra_policy import CAMERAS

    if not is_reference_bundle(demo):
        raise ValueError('Expected train_reference_bundle_v2')
    if not _sha256(demo.get('sha256')) or demo['sha256'] != bundle_digest(demo):
        raise ValueError('TRAIN bundle content hash mismatch')
    episodes = demo.get('episodes')
    if not isinstance(episodes, list) or len(episodes) < 1:
        raise ValueError('TRAIN bundle requires at least one episode')
    from roboicl.config import supported_task
    if (not supported_task(demo.get('task'))
            or not isinstance(demo.get('instruction'), str) or not demo['instruction'].strip()
            or demo.get('action_space') != 'ee_delta' or demo.get('delta_frame') != 'world'
            or demo.get('frequency') != 25):
        raise ValueError('TRAIN bundle requires a shared supported task and 25Hz world ee_delta contract')
    detail = demo.get('prompt_image_encoding', {}).get('detail')
    if demo.get('prompt_image_encoding') != image_encoding('native_jpeg', detail, 'triptych'):
        raise ValueError('TRAIN bundle requires the native triptych image contract')
    hashes, paths, flattened = set(), set(), []
    chunk_count = None
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get('variant') != CHILD_VARIANT:
            raise ValueError('TRAIN bundle children must be action-chunk manifests')
        if any(episode.get(key) != demo[key] for key in COMMON_FIELDS):
            raise ValueError('TRAIN bundle episodes have different common contracts')
        source = episode.get('source_file')
        if not isinstance(source, str) or not source or not _sha256(episode.get('sha256')):
            raise ValueError('TRAIN bundle episode requires a source path and SHA256')
        path = source.casefold()
        if path in paths or episode['sha256'] in hashes:
            raise ValueError('TRAIN bundle episodes must have distinct sources')
        paths.add(path)
        hashes.add(episode['sha256'])
        total = episode.get('total_frames')
        examples = episode.get('examples')
        if (type(total) is not int or total < 2 or not isinstance(examples, list)
                or not 1 <= len(examples) <= 47):
            raise ValueError('Each TRAIN bundle episode requires 1..47 full action chunks')
        if chunk_count is not None and len(examples) != chunk_count:
            raise ValueError('TRAIN bundle episodes must have equal sampled observation counts')
        chunk_count = len(examples)
        previous_end = 0
        horizon = len(examples[0].get('action_tensor', []))
        if not 1 <= horizon <= 64:
            raise ValueError('TRAIN horizon must be 1..64')
        for index, example in enumerate(examples):
            start, end = example.get('frame'), example.get('result_frame')
            tensor = example.get('action_tensor')
            if (type(start) is not int or type(end) is not int or start < previous_end
                    or (index == 0 and start != 0) or end != start + horizon or end >= total
                    or example.get('action_step_interval_inclusive') != [start, end - 1]
                    or not isinstance(tensor, list) or len(tensor) != horizon):
                raise ValueError('TRAIN bundle chunks require frame 0, ordered equal-length action intervals and resulting observations')
            values = np.asarray(tensor, dtype=float)
            if (values.shape != (horizon, 14) or not np.isfinite(values).all()
                    or any(type(value) not in (float, int) for row in tensor for value in row)
                    or np.any(values[:, [6, 13]] < 0) or np.any(values[:, [6, 13]] > 1)):
                raise ValueError('TRAIN bundle action tensors must be finite [horizon,14] with valid grippers')
            previous_end = end
        terminal = episode.get('terminal_observation')
        if (not isinstance(terminal, dict) or type(terminal.get('frame')) is not int
                or terminal['frame'] != total - 1 or terminal['frame'] < previous_end
                or 'action_tensor' in terminal or 'result_frame' in terminal):
            raise ValueError('TRAIN bundle requires the true final observation without a fabricated action')
        for observation in [*examples, terminal]:
            images = observation.get('prompt_images')
            if (not isinstance(observation.get('state'), dict) or not isinstance(images, dict)
                    or set(images) != set(CAMERAS) or observation.get('head_image') != images['cam_head']
                    or any(not isinstance(url, str) or not url.startswith('data:image/jpeg;base64,')
                           for url in images.values())):
                raise ValueError('TRAIN bundle observations require state and all three matching source camera images')
        for example in examples:
            result_images = example.get('result_prompt_images')
            if not isinstance(example.get('result_state'), dict):
                raise ValueError('TRAIN bundle chunks require embedded result state')
            # Older verified bundles predate result-camera embedding.  They are
            # still valid when the selected dialogue profile does not request
            # TRAIN result observations; newer bundles may include the images.
            if result_images is not None and (
                    not isinstance(result_images, dict)
                    or set(result_images) != set(CAMERAS)
                    or any(not isinstance(url, str) or not url.startswith('data:image/jpeg;base64,')
                           for url in result_images.values())):
                raise ValueError('TRAIN result camera images are malformed')
        flattened.extend(examples)
    if demo.get('examples') != flattened:
        raise ValueError('TRAIN bundle flattened examples differ from episode examples')
    return episodes


def verify_reference_bundle(demo, profile=None):
    """Re-read every source and every selected RGB triple; never trust receipts."""
    import h5py
    from XPolicyLab.utils.process_data import decode_image_bit
    from .observation_images import encode_camera_rgb
    from .train_reference_checks import source_cameras, source_path, source_state, verify_training_manifest
    from .train_reference_checks import ARMS, CAMERAS, digest

    episodes = validate_bundle(demo)
    reports = []
    for episode in episodes:
        report = verify_training_manifest(episode, profile)
        origin = report['profile']['train_world_origin']
        with h5py.File(source_path(episode['source_file'], episode.get('task')), 'r') as data:
            total = episode['total_frames']
            for arm in ARMS:
                for group in ('state', 'action'):
                    for key in ('arm_joint_states', 'ee_poses', 'ee_joint_states'):
                        if len(data[f'{group}/{arm}_{key}']) != total:
                            raise ValueError('TRAIN bundle total_frames differs from source labels')
            instruction = data['instruction'][()]
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            if instruction != episode['instruction']:
                raise ValueError('TRAIN bundle instruction differs from source')
            observations = [*episode['examples'], episode['terminal_observation']]
            image_checks = []
            for observation in observations:
                frame = observation['frame']
                cameras = source_cameras(data, frame, origin)
                if observation['state'] != source_state(data, frame):
                    raise ValueError('TRAIN bundle observation state differs from source')
                image_hashes = {}
                for camera in CAMERAS:
                    colors = data[f'vision/{camera}/colors']
                    if len(colors) != total:
                        raise ValueError('TRAIN bundle camera length differs from source frames')
                    rgb = decode_image_bit(colors[frame])
                    if list(np.shape(rgb)) != cameras[camera]['shape'] or list(np.shape(rgb)) != [480, 640, 3]:
                        raise ValueError('TRAIN bundle RGB shape differs from calibrated native tile')
                    expected = encode_camera_rgb(rgb, 'native_jpeg')
                    if observation['prompt_images'][camera] != expected:
                        raise ValueError(f'TRAIN bundle image differs from source: frame={frame}, camera={camera}')
                    image_hashes[camera] = hashlib.sha256(expected.encode('ascii')).hexdigest()
                image_checks.append({'frame': frame, 'passed': True, 'source_image_sha256': image_hashes})
            for example in episode['examples']:
                frame = example['result_frame']
                if example['result_state'] != source_state(data, frame):
                    raise ValueError('TRAIN bundle result state differs from source')
                result_images = example.get('result_prompt_images')
                if result_images is not None:
                    for camera in CAMERAS:
                        rgb = decode_image_bit(data[f'vision/{camera}/colors'][frame])
                        expected = encode_camera_rgb(rgb, 'native_jpeg')
                        if result_images[camera] != expected:
                            raise ValueError(f'TRAIN bundle result image differs from source: frame={frame}, camera={camera}')
            terminal = observations[-1]
            cameras = source_cameras(data, terminal['frame'], origin)
            report['terminal_observation'] = {'frame': terminal['frame'], 'passed': True,
                'state': deepcopy(terminal['state']), 'cameras': cameras,
                'inputs_sha256': digest({'state': terminal['state'], 'cameras': cameras})}
            report['selected_image_checks'] = image_checks
        reports.append(report)
    return {'schema': 'robodojo.train_bundle_integrity.v1', 'passed': True, 'bundle': True,
            'source_sha256': demo['sha256'], 'episodes': reports,
            'camera_scope': 'All three source RGB images and calibrations checked at every selected observation; no kinematic or projection audit.',
            'convention_provenance': 'Explicit geometry profiles; historical writer provenance is not certified by HDF5 field names.'}
