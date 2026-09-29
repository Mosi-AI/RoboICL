"""TRAIN source integrity and camera calibration only; no kinematic audits."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import h5py
import numpy as np
from .astra_policy import CAMERAS
from .head_calibration_capture import camera_arrays
from roboicl.paths import data_root
ARMS = ('left','right')


def source_path(value, task=None):
    """Resolve a TRAIN source, relocating stale absolute paths by task.

    Historical bundles may contain an absolute path from another server.  A
    local canonical copy under ``runtime-data/<task>/train`` is equivalent
    only after its locked SHA256 is checked by the caller; never silently use a
    different file.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        return data_root() / path
    if path.is_file() or not task:
        return path
    candidate = data_root() / 'runtime-data' / str(task) / 'train' / path.name
    return candidate if candidate.is_file() else path

def finite(value, shape):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Invalid calibration array')
    return value

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def train_calibration_profile(root=None):
    return {'frame':'environment_origin','train_world_origin':[0.,0.,0.],
            'train_camera_convention':'camera_to_world_usd'}

def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def source_state(data, step, group='state'):
    return {f'{arm}_{key}': data[f'{group}/{arm}_{source}'][step].tolist()
            for arm in ARMS for key, source in (('arm_joint_state', 'arm_joint_states'),
                ('ee_pose', 'ee_poses'), ('ee_joint_state', 'ee_joint_states'))
            if f'{group}/{arm}_{source}' in data}

def source_cameras(data, step, origin):
    cameras = {}
    for name in CAMERAS:
        prefix = f'vision/{name}/'
        raw_k, raw_t = data[prefix + 'intrinsic_matrix'], data[prefix + 'extrinsic_matrix']
        # Intrinsics can be static or recorded per frame. Extrinsics must be per-frame.
        k = raw_k[:] if raw_k.shape == (3, 3) else raw_k[step]
        if raw_t.ndim != 3 or raw_t.shape[1:] != (4, 4):
            raise ValueError('TRAIN requires per-frame camera extrinsics')
        transform = raw_t[step].copy()
        transform[:3, 3] -= finite(origin, (3,))
        shape = data[prefix + 'shape']
        shape = shape[:] if shape.shape == (3,) else shape[step]
        camera = {'intrinsic_matrix': k.tolist(), 'camera_to_environment': transform.tolist(), 'shape': shape.tolist()}
        camera_arrays(camera)
        cameras[name] = camera
    return cameras

def verify_training_manifest(demo, profile=None):
    from .train_reference_bundle import is_reference_bundle, verify_reference_bundle
    if is_reference_bundle(demo):
        return verify_reference_bundle(demo, profile)
    from .reference_data import tensor_at
    supplied = profile or demo.get('geometry_profile') or train_calibration_profile()
    profile = {k:deepcopy(supplied[k]) for k in ('frame','train_world_origin','train_camera_convention')}
    if profile['train_camera_convention'] != 'camera_to_world_usd':
        raise ValueError('Explicit TRAIN camera convention required')
    origin = finite(profile['train_world_origin'], (3,))
    source = source_path(demo['source_file'], demo.get('task'))
    actual_hash = file_hash(source)
    if actual_hash != demo['sha256']:
        raise ValueError('TRAIN source hash mismatch')
    if demo.get('action_space') != 'ee_delta' or demo.get('delta_frame') != 'world':
        raise ValueError('TRAIN requires world ee_delta labels')
    chunks = []
    with h5py.File(source,'r') as data:
        frequency = float(data['additional_info/frequency'][()])
        if frequency != 25 or demo.get('frequency') != frequency:
            raise ValueError('TRAIN control frequency mismatch')
        reference = source_cameras(data,0,origin)['cam_head']
        previous_end = -1
        for example in demo['examples']:
            start,end = example['frame'],example['result_frame']
            if (type(start) is not int or type(end) is not int or start < 0 or start < previous_end
                or end <= start or example['action_step_interval_inclusive'] != [start,end-1]
                or end-start != len(example['action_tensor'])):
                raise ValueError('Invalid TRAIN chunk interval')
            if example['state'] != source_state(data,start) or example['result_state'] != source_state(data,end):
                raise ValueError('TRAIN example state differs from source')
            if tensor_at(data,start,end-start) != example['action_tensor']:
                raise ValueError('TRAIN tensor differs from source actions')
            chunks.append({'passed':True,'step_interval':[start,end]})
            previous_end=end
    if not chunks:raise ValueError('No TRAIN chunks to verify')
    return {'schema':'robodojo.train_source_integrity.v1','passed':True,'source_sha256':actual_hash,
            'profile':profile,'head_reference':reference,'source_frequency_hz':frequency,
            'camera_scope':'Calibration extraction and source integrity only; no geometry audit',
            'convention_provenance':'Explicit TRAIN camera convention and origin', 'chunks':chunks}

def training_summary(report):
    keys=('schema','passed','source_sha256','camera_scope','convention_provenance')
    out={k:deepcopy(report[k]) for k in keys}
    if report.get('bundle'):
        out.update(bundle=True,episodes=[training_summary(e) for e in report['episodes']])
    else:
        out['chunks']=deepcopy(report['chunks'])
        if 'terminal_observation' in report:
            terminal = report['terminal_observation']
            out['terminal_observation'] = {key: deepcopy(terminal[key])
                for key in ('frame', 'passed', 'inputs_sha256') if key in terminal}
        if 'selected_image_checks' in report:
            out['selected_image_checks']=deepcopy(report['selected_image_checks'])
    return out
