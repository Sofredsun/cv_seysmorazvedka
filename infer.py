import argparse
import time
from pathlib import Path

import numpy as np
import torch

import seismic_lib as sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_tta', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cube = sl.load_cube(args.task)
    mask = sl.hole_mask(args.task, cube.shape)
    model, cfg, norm = sl.load_model(args.ckpt, device)
    print('config', cfg, 'norm', norm, flush=True)

    t0 = time.time()
    restored = sl.restore_cube(args.task, cube, model, cfg, norm, device=device, tta=not args.no_tta)
    print(f'inference {time.time() - t0:.0f}s', flush=True)

    # Проверки
    assert restored.shape == cube.shape, (restored.shape, cube.shape)
    assert restored.dtype == cube.dtype, restored.dtype
    assert np.isfinite(restored).all(), 'NaN/inf в результате'
    assert np.array_equal(restored[~mask], cube[~mask]), 'известные значения изменены'
    filled = restored[mask]
    known = cube[~mask][::97]
    print(f'OK: shape {restored.shape}, hole voxels {mask.sum()}, '
          f'zeros left in hole {(filled == 0).sum()}', flush=True)
    print(f'known: mean {known.mean():.4f} std {known.std():.4f} | '
          f'filled: mean {filled.mean():.4f} std {filled.std():.4f}', flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, restored)
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
