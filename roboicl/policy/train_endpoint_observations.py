"""Materialize sampled TRAIN endpoints from their checked source recording.

Endpoint pixels are never synthesized from an action or copied from its start.
The original manifest remains unchanged; the fixed prompt owns these images.
"""
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from .astra_policy import CAMERAS
from .observation_images import encode_camera_rgb
from .train_reference_checks import file_hash, source_path, source_state


def result_observations(demo, *, cameras=('cam_head',), image_profile='thumbnail_jpeg'):
    """Read all requested endpoints with one source hash check and one HDF5 open.

Even optional precomputed ``result_prompt_images`` are checked against the raw
recording before use. A missing or stale source fails before an API request.
    """
    from XPolicyLab.utils.process_data import decode_image_bit

    cameras = tuple(dict.fromkeys(cameras))
    if not cameras or any(camera not in CAMERAS for camera in cameras):
        raise ValueError('TRAIN endpoint RGB requires supported cameras')
    embedded = all(
        isinstance(example.get('result_prompt_images'), dict)
        and all(camera in example['result_prompt_images'] for camera in cameras)
        for example in demo.get('examples', [])
    )
    if embedded:
        if image_profile != 'native_jpeg':
            raise ValueError('Embedded TRAIN endpoint RGB requires native_jpeg')
        return {
            example['result_frame']: {
                'frame': example['result_frame'],
                # Match source_state()/LIVE serialization regardless of the
                # insertion order used by a portable embedded bundle.
                'state': {
                    key: deepcopy(example['result_state'][key])
                    for key in (
                        'left_arm_joint_state', 'left_ee_pose', 'left_ee_joint_state',
                        'right_arm_joint_state', 'right_ee_pose', 'right_ee_joint_state',
                    )
                    if key in example['result_state']
                },
                'prompt_images': {camera: example['result_prompt_images'][camera] for camera in cameras},
            }
            for example in demo['examples']
        }

    source_file = demo.get('source_file')
    if not isinstance(source_file, str) or not source_file:
        raise ValueError('TRAIN endpoint RGB requires a verifiable source_file')
    source = source_path(source_file)
    if not source.is_file():
        raise ValueError('TRAIN endpoint RGB source_file is missing')
    if file_hash(source) != demo.get('sha256'):
        raise ValueError('TRAIN endpoint RGB source hash mismatch')
    endpoints = {}
    with h5py.File(source, 'r') as data:
        total = len(data['state/left_ee_poses'])
        if total < 2 or demo.get('total_frames', total) != total:
            raise ValueError('TRAIN endpoint RGB total_frames differs from source')
        for camera in cameras:
            if len(data[f'vision/{camera}/colors']) != total:
                raise ValueError('TRAIN endpoint RGB camera length differs from source frames')
        for example in demo['examples']:
            start, end = example.get('frame'), example.get('result_frame')
            tensor = example.get('action_tensor', [])
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end < total or end - start != len(tensor)
                    or example.get('action_step_interval_inclusive') != [start, end - 1]):
                raise ValueError('TRAIN endpoint RGB requires an aligned action/result interval')
            state = source_state(data, end)
            if example.get('result_state') != state:
                raise ValueError('TRAIN endpoint observation state differs from source')
            images = {}
            for camera in cameras:
                rgb = decode_image_bit(data[f'vision/{camera}/colors'][end])
                recorded_shape = data[f'vision/{camera}/shape']
                recorded_shape = recorded_shape[:] if recorded_shape.shape == (3,) else recorded_shape[end]
                if list(np.shape(rgb)) != list(recorded_shape) or list(np.shape(rgb)) != [480, 640, 3]:
                    raise ValueError('TRAIN endpoint RGB shape differs from native source camera')
                images[camera] = encode_camera_rgb(rgb, image_profile)
            supplied = example.get('result_prompt_images')
            if supplied is not None and any(supplied.get(camera) != images[camera] for camera in cameras):
                raise ValueError('TRAIN endpoint image differs from source')
            endpoints[end] = {'frame': end, 'state': deepcopy(state), 'prompt_images': images}
    return endpoints
