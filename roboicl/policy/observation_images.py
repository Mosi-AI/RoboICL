"""One RGB-to-prompt encoding path for TRAIN and live camera observations."""
import base64
from copy import deepcopy
from functools import lru_cache
from io import BytesIO

from PIL import Image

from .astra_policy import image_url


CAMERA_IMAGE_ENCODING = {
    'schema': 'robodojo.camera_image_encoding.v1',
    'color_mode': 'RGB', 'archive_format': 'JPEG', 'archive_quality': 90,
    'resize': 'PIL.Image.thumbnail', 'max_size_wh': [384, 384],
    'output_format': 'JPEG', 'output_quality': 80,
    'url_prefix': 'data:image/jpeg;base64,', 'detail': 'low',
}


TRIPTYCH_CAMERA_ORDER = ('cam_left_wrist', 'cam_head', 'cam_right_wrist')
TRIPTYCH_PANEL_SIZE_WH = (640, 480)


def image_layout_content(image_layout='separate'):
    """Describe a shared layout once in the fixed prompt, not at every frame."""
    image_layout_metadata(image_layout)
    if image_layout == 'separate':
        return []
    return [{'type': 'input_text', 'text': (
        'Images are native 1920x480 triptychs: left wrist | head | right wrist, '
        'each 640x480. Head pixels map as uv_triptych = uv_head + [640,0].')}]



def image_layout_metadata(image_layout='separate'):
    """Describe pixel coordinates without modifying any source camera calibration."""
    if image_layout == 'separate':
        return {'layout': 'separate'}
    if image_layout != 'triptych':
        raise ValueError('Unknown camera image layout')
    width, height = TRIPTYCH_PANEL_SIZE_WH
    return {
        'layout': 'triptych',
        'camera_order': list(TRIPTYCH_CAMERA_ORDER),
        'size_wh': [3 * width, height],
        'resize': 'none',
        'pixel_regions': [
            {'camera': camera, 'xyxy_exclusive': [index * width, 0, (index + 1) * width, height],
             'source_to_image_offset_xy': [index * width, 0]}
            for index, camera in enumerate(TRIPTYCH_CAMERA_ORDER)
        ],
    }


def image_encoding(profile='thumbnail_jpeg', detail='low', image_layout='separate'):
    """Return the source-to-prompt contract without changing the legacy default."""
    if profile not in ('thumbnail_jpeg', 'native_jpeg'):
        raise ValueError('Unknown camera image profile')
    if detail not in ('low', 'high', 'original', 'auto'):
        raise ValueError('Unknown camera image detail')
    contract = deepcopy(CAMERA_IMAGE_ENCODING)
    contract['detail'] = detail
    if profile == 'native_jpeg':
        contract.update(resize='none', max_size_wh=[640, 640], output_quality=90)
    layout = image_layout_metadata(image_layout)
    if image_layout == 'triptych':
        if profile != 'native_jpeg':
            raise ValueError('Triptych layout requires the native_jpeg image profile')
        contract.update(max_size_wh=layout['size_wh'], exact_size_wh=layout['size_wh'],
                        output_chroma_subsampling=0, image_layout=layout,
                        source_camera_size_wh=list(TRIPTYCH_PANEL_SIZE_WH))
    return contract


def _validate_image(url, contract):
    if not isinstance(url, str) or not url.startswith(contract['url_prefix']):
        raise ValueError('Camera input requires a JPEG base64 data URL')
    try:
        raw = base64.b64decode(url.split(',', 1)[1], validate=True)
        with Image.open(BytesIO(raw)) as picture:
            if (picture.format != 'JPEG' or picture.mode != 'RGB'
                    or any(size > limit for size, limit in zip(picture.size, contract['max_size_wh']))
                    or ('exact_size_wh' in contract and list(picture.size) != contract['exact_size_wh'])):
                raise ValueError('Camera input must be RGB JPEG within the image profile dimensions')
            picture.load()
    except (OSError, ValueError) as error:
        raise ValueError(f'Invalid camera image: {error}') from error


def validate_prompt_image(url, profile='thumbnail_jpeg', detail='low', image_layout='separate'):
    """Validate a final prompt image, including an already composed triptych."""
    _validate_image(url, image_encoding(profile, detail, image_layout))


def small_image(url):
    raw = base64.b64decode(url.split(',', 1)[1])
    with Image.open(BytesIO(raw)) as source:
        rgb = source.convert('RGB')
        rgb.thumbnail((384, 384))
        stream = BytesIO()
        rgb.save(stream, format='JPEG', quality=80)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def prompt_image(archive_url, profile='thumbnail_jpeg'):
    """Use one archived JPEG90; native images are validated and returned unchanged."""
    contract = image_encoding(profile)
    if profile == 'thumbnail_jpeg':
        return small_image(archive_url)
    _validate_image(archive_url, contract)
    return archive_url


def encode_camera_rgb(rgb, profile='thumbnail_jpeg'):
    return prompt_image(image_url(rgb), profile)


def _triptych_image(images, cameras):
    cameras = tuple(cameras)
    if len(cameras) != 3 or set(cameras) != set(TRIPTYCH_CAMERA_ORDER):
        raise ValueError('Triptych layout requires all three cameras exactly once')
    missing = [camera for camera in TRIPTYCH_CAMERA_ORDER if camera not in images]
    if missing:
        raise ValueError(f'Triptych layout is missing camera images: {missing}')
    return _compose_triptych(tuple(images[camera] for camera in TRIPTYCH_CAMERA_ORDER))


@lru_cache(maxsize=128)
def _compose_triptych(urls):
    """Reuse immutable source URLs, with a bounded cache for repeated TRAIN frames."""
    source_contract = image_encoding('native_jpeg')
    source_contract['exact_size_wh'] = list(TRIPTYCH_PANEL_SIZE_WH)
    width, height = TRIPTYCH_PANEL_SIZE_WH
    panorama = Image.new('RGB', (3 * width, height))
    for index, url in enumerate(urls):
        _validate_image(url, source_contract)
        raw = base64.b64decode(url.split(',', 1)[1], validate=True)
        with Image.open(BytesIO(raw)) as picture:
            panorama.paste(picture, (index * width, 0))
    stream = BytesIO()
    # Full-size panels and 4:4:4 chroma preserve small features without padding.
    panorama.save(stream, format='JPEG', quality=90, subsampling=0)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def camera_input_parts(step, images, cameras, *, profile='thumbnail_jpeg', detail='low',
                       image_layout='separate'):
    """Share TRAIN/LIVE assembly; separate JPEGs remain byte-for-byte unchanged."""
    if image_layout == 'triptych':
        contract = image_encoding(profile, detail, image_layout)
        url = _triptych_image(images, cameras)
        _validate_image(url, contract)
        return [
            {'type': 'input_image', 'detail': detail, 'image_url': url},
        ]
    image_layout_metadata(image_layout)
    contract = image_encoding(profile, detail)
    content = []
    for camera in cameras:
        url = images[camera]
        _validate_image(url, contract)
        content.extend([{'type': 'input_text', 'text': f'step={step}, camera={camera}'},
                        {'type': 'input_image', 'detail': detail, 'image_url': url}])
    return content
