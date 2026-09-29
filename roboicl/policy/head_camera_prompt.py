"""Expose checked head calibration without exporting robot assets or scene data."""
import base64
from io import BytesIO
import json

import numpy as np
from PIL import Image

from .head_calibration_capture import camera_arrays


def image_size(image_url):
    with Image.open(BytesIO(base64.b64decode(image_url.split(',', 1)[1], validate=True))) as image:
        return image.size


def calibration_conventions():
    """State the shared geometry contract once, before source-specific matrices."""
    return {'type': 'input_text', 'text': (
        'Head calibration is fixed per episode. TRAIN and TEST_RUNTIME matrices are separate; '
        'a TRAIN call-id prefix identifies its episode when calibrations differ. '
        'K=intrinsic_matrix for the supplied head tile after resizing, before layout offset. '
        'T=camera_to_environment maps USD camera axes (+X right,+Y up,-Z forward) to the EE '
        'environment frame in meters. Project p: u=inverse(T)@[p,1], '
        'v=[u.x,-u.y,-u.z], h=K@v, uv_head=h.xy/h.z for v.z>0. '
        'Pixels use continuous top-left coordinates; resizing scales them without a pixel-center correction. '
        'Head only; not wrist-camera calibration.')}



def calibration_content(camera, image_url, source, provenance, *, image_profile='thumbnail_jpeg', image_layout='separate'):
    """Export only the checked source and effective head K/T; provenance stays in local audits."""
    if source not in ('TRAIN', 'TEST_RUNTIME'):
        raise ValueError('Head calibration requires an explicit TRAIN or TEST_RUNTIME source')
    k, transform, shape = camera_arrays(camera)
    width, height = int(shape[1]), int(shape[0])
    size = image_size(image_url)
    if image_profile == 'thumbnail_jpeg':
        with Image.new('RGB', (width, height)) as expected:
            expected.thumbnail((384, 384))
            expected_size = expected.size
    elif image_profile == 'native_jpeg':
        expected_size = (width, height)
    else:
        raise ValueError('Unknown camera image profile')
    if size != expected_size:
        raise ValueError('Head calibration/image size mismatch for the selected full-frame image profile')
    scale = np.diag([size[0] / width, size[1] / height, 1.])
    value = {
        'source': source, 'camera': 'cam_head', 'reference_step': 0,
        'intrinsic_matrix': (scale @ k).tolist(),
        'camera_to_environment': transform.tolist(),
    }
    if image_layout == 'triptych':
        if image_profile != 'native_jpeg' or size != (640, 480):
            raise ValueError('Triptych calibration requires a native 640x480 head tile')
        # The shared layout describes the +640 pixel translation once. Keep K
        # in head-tile coordinates, with no second, shifted matrix in each block.
    elif image_layout != 'separate':
        raise ValueError('Unknown camera image layout')
    return {'type': 'input_text', 'text': json.dumps({'head_camera_calibration': value},
            ensure_ascii=False, allow_nan=False, separators=(',', ':'))}
