import argparse
import time
from pathlib import Path

import numpy as np
import torch

import seismic_lib as sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--ckpt')
    source.add_argument('--ensemble', nargs='+', metavar='CKPT')
    ap.add_argument('--blend-wide', nargs=2, metavar='CKPT',
                    help='mix the original ensemble 50/50 with a two-model wide ensemble')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_tta', action='store_true')
    ap.add_argument('--bs', type=int, default=8)
    args = ap.parse_args()
    if args.ensemble and (args.task != 'inpainting' or len(args.ensemble) < 2):
        ap.error('--ensemble requires inpainting and at least two checkpoints')
    if args.blend_wide and not args.ensemble:
        ap.error('--blend-wide requires --ensemble')
    if args.bs < 1:
        ap.error('--bs must be positive')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cube = sl.load_cube(args.task)
    mask = sl.hole_mask(args.task, cube.shape)
    if args.ensemble:
        model, cfg, norm = sl.load_inpainting_ensemble(args.ensemble, device)
    else:
        model, cfg, norm = sl.load_model(args.ckpt, device)
    print('config', cfg, 'norm', norm, flush=True)
    if args.blend_wide:
        wide, wide_cfg, wide_norm = sl.load_inpainting_ensemble(args.blend_wide, device)
        if cfg['k'] != 1 or cfg['context_z'] not in (None, 192) or cfg['dip_guided']:
            ap.error('--blend-wide requires a standard single-slice baseline ensemble')
        if not wide_cfg['wide_tunnel'] or wide_norm != norm:
            ap.error('--blend-wide requires wide-tunnel checkpoints with matching normalization')

    t0 = time.time()
    restored = sl.restore_cube(args.task, cube, model, cfg, norm, device=device,
                               bs=args.bs, tta=not args.no_tta)
    if args.blend_wide:
        original_fill = restored[mask]
        del restored
        print('Baseline ensemble finished; reconstructing with both wide models', flush=True)
        restored = sl.restore_cube(args.task, cube, wide, wide_cfg, norm, device=device,
                                   bs=args.bs, tta=not args.no_tta)
        restored[mask] = (original_fill + restored[mask]) * 0.5
        del original_fill
    print(f'inference {time.time() - t0:.0f}s', flush=True)

    # Проверки
    assert restored.shape == cube.shape, (restored.shape, cube.shape)
    assert restored.dtype == cube.dtype, restored.dtype
    assert np.isfinite(restored).all(), 'NaN/inf в результате'
    assert np.array_equal(restored[~mask], cube[~mask]), 'известные значения изменены'
    filled = restored[mask]
    if args.blend_wide:
        assert np.all(filled != 0), 'в пропуске остались нули'
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
