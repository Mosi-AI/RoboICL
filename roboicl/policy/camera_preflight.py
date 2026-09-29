"""Opt-in render diagnostic. No model request, robot action, or score."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image


def run(env):
    folder = Path(os.environ['ASTRA_CAMERA_PREFLIGHT'])
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(10):
        obs = env.get_obs()
        for camera, data in obs['vision'].items():
            pixels = np.asarray(data['color'])
            rows.append({'frame': frame, 'camera': camera, 'route': 'observation',
                         'shape': list(pixels.shape), 'mean': float(pixels.mean()),
                         'max': int(pixels.max())})
            if frame in (0, 9):
                Image.fromarray(pixels).save(folder/f'{camera}_{frame}.png')
        for index, view in enumerate(env.capture_manager.tiled_cameras):
            for device in ('cuda', 'cpu'):
                raw = view._annotators['rgba'].get_data(device=device)
                pixels = raw.numpy() if hasattr(raw, 'numpy') else np.asarray(raw)
                rows.append({'frame': frame, 'camera': index, 'route': device,
                             'product': str(view._render_product.path),
                             'prim_paths': view.prim_paths,
                             'shape': list(pixels.shape), 'mean': float(pixels.mean()) if pixels.size else None,
                             'max': int(pixels.max()) if pixels.size else None})
    (folder/'capture_probe.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
