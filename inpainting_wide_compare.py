import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import seismic_lib as sl
from compare import RAcc, model_pred
from inpainting_wide import HOLE, PATCH
from train_unet import SPLITS


class MatchedWindows(Dataset):
    def __init__(self, cube, real_mask, norm, rows, count, seed, include_trace=False):
        self.cube, self.real_mask, self.norm, self.rows = cube, real_mask[0], norm, rows
        self.include_trace = include_trace
        self.standard = sl.InpaintPatches(cube, real_mask, norm, rows, n_samples=count,
                                          sources=('a0',), augment=False, seed=seed, fixed=True)

    def __len__(self):
        return len(self.standard)

    def __getitem__(self, idx):
        x, y, mask, (row, trace, depth) = self.standard.sample(idx, with_location=True)
        h, w = PATCH
        hh, hw = HOLE
        j = int(np.clip(trace - (h - hh) // 2, 0, self.cube.shape[1] - h))
        z = int(np.clip(depth - (w - hw) // 2, 0, self.cube.shape[2] - w))
        neighbors = row + np.arange(-2, 3)
        neighbors = np.where(neighbors < self.rows.start, 2 * self.rows.start - neighbors, neighbors)
        neighbors = np.where(neighbors >= self.rows.stop, 2 * (self.rows.stop - 1) - neighbors, neighbors)
        box = self.cube[neighbors, j:j + h, z:z + w]
        score_mask = np.zeros(PATCH, dtype=bool)
        score_mask[trace - j:trace - j + hh, depth - z:depth - z + hw] = True
        input_mask = score_mask | self.real_mask[j:j + h, z:z + w]
        wide_x, _, _ = sl._pack(self.norm.fwd(box), input_mask.astype(np.float32))
        sample = x, y[mask.bool()].reshape(1, hh, hw), mask.bool(), wide_x, score_mask[None]
        return (*sample, trace) if self.include_trace else sample


def predict_groups(baseline, wide, norm, x, y, mask, wx, wmask):
    coarse = model_pred(baseline, x, 'inpainting', 1, True)[mask].reshape_as(y)
    predictions = [((member(wx) + member(wx.flip(2)).flip(2)) / 2)[wmask].reshape_as(y)
                   for member in wide.models]
    first = norm.fwd(norm.inv(predictions[0]).clamp(0, 1))
    average = norm.fwd(norm.inv(torch.stack(predictions).mean(0)).clamp(0, 1))
    return coarse, first, average


def invert_polarity(wide_input):
    return torch.cat((-wide_input[:, :-1], wide_input[:, -1:]), dim=1)


@torch.inference_mode()
def evaluate(loader, baseline, wide, norm, device):
    names = ('baseline', 'wide_first', 'wide_ensemble', 'blend_first', 'blend_two_seeds')
    stats = {name: RAcc() for name in names}
    for x, y, mask, wx, wmask in loader:
        x, y, mask, wx, wmask = (v.to(device) for v in (x, y, mask, wx, wmask))
        coarse, first, average = predict_groups(baseline, wide, norm, x, y, mask, wx, wmask)
        valid = torch.ones_like(y)
        for name, pred in zip(names, (coarse, first, average, (coarse + first) / 2, (coarse + average) / 2)):
            stats[name].add(pred, y, valid)
    return {name: dict(zip(('pearson_mean_tta', 'pearson_global_tta'), acc.result()))
            for name, acc in stats.items()}


def blend_depth_bands(coarse, wide, kernel, low_weight, high_weight):
    delta = wide - coarse
    radius = kernel.shape[-1] // 2
    low = F.conv2d(F.pad(delta, (radius, radius, 0, 0), mode='reflect'), kernel)
    return coarse + low_weight * low + high_weight * (delta - low)


@torch.inference_mode()
def evaluate_frequency_blends(loader, baseline, wide, norm, device):
    profiles = {'fixed_50_50': (0.5, 0.5), 'wide_more_low': (0.75, 0.25),
                'wide_more_high': (0.25, 0.75)}
    regions = ('all', 'left', 'right', 'center')
    stats = {name: {region: RAcc() for region in regions} for name in profiles}
    grid = torch.arange(-8, 9, dtype=torch.float32, device=device)
    kernel = (-0.5 * (grid / 2).square()).softmax(0).view(1, 1, 1, -1)
    for x, y, mask, wx, wmask, trace in loader:
        x, y, mask, wx, wmask, trace = (v.to(device) for v in (x, y, mask, wx, wmask, trace))
        coarse, _, average = predict_groups(baseline, wide, norm, x, y, mask, wx, wmask)
        left = trace < 368
        center = np.s_[..., 25:-25, 25:-25]
        for name, (low, high) in profiles.items():
            pred = blend_depth_bands(coarse, average, kernel, low, high)
            valid = torch.ones_like(y)
            scores = stats[name]
            scores['all'].add(pred, y, valid)
            scores['center'].add(pred[center], y[center], valid[center])
            if left.any():
                scores['left'].add(pred[left], y[left], valid[left])
            if (~left).any():
                scores['right'].add(pred[~left], y[~left], valid[~left])
    return {name: {'wide_low_weight': profiles[name][0], 'wide_high_weight': profiles[name][1],
                   **{f'pearson_{region}': acc.result()[1] for region, acc in scores.items() if acc.rs}}
            for name, scores in stats.items()}


@torch.inference_mode()
def evaluate_polarity_blends(loader, baseline, wide, norm, device):
    stats = {name: (RAcc(), RAcc()) for name in ('fixed_50_50', 'polarity_tta')}
    center = np.s_[..., 25:-25, 25:-25]
    for x, y, mask, wx, wmask in loader:
        x, y, mask, wx, wmask = (v.to(device) for v in (x, y, mask, wx, wmask))
        coarse = model_pred(baseline, x, 'inpainting', 1, True)[mask].reshape_as(y)

        def lateral_tta(input_window):
            p = (wide(input_window) + wide(input_window.flip(2)).flip(2)) / 2
            return p[wmask].reshape_as(y)

        ordinary = lateral_tta(wx)
        polarity = (ordinary - lateral_tta(invert_polarity(wx))) / 2
        for name, raw in (('fixed_50_50', ordinary), ('polarity_tta', polarity)):
            clipped = norm.fwd(norm.inv(raw).clamp(0, 1))
            prediction = (coarse + clipped) / 2
            global_acc, center_acc = stats[name]
            global_acc.add(prediction, y, torch.ones_like(y))
            center_acc.add(prediction[center], y[center], torch.ones_like(y[center]))
    return {name: {'pearson_mean': full.result()[0], 'pearson_global': full.result()[1],
                   'pearson_center': inner.result()[1]}
            for name, (full, inner) in stats.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', nargs=3, required=True, metavar='CKPT')
    parser.add_argument('--wide', nargs=2, required=True, metavar='CKPT')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--count', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bs', type=int, default=2)
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument('--frequency-probe', action='store_true',
                        help='compare fixed wide weights for smooth and detailed depth bands')
    probes.add_argument('--polarity-probe', action='store_true',
                        help='compare the blend with and without polarity TTA on its wide models')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if min(args.count, args.bs) < 1:
        parser.error('--count and --bs must be positive')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline, cfg, norm = sl.load_inpainting_ensemble(args.baseline, device)
    wide, wide_cfg, wide_norm = sl.load_inpainting_ensemble(args.wide, device)
    if cfg['k'] != 1 or cfg['context_z'] not in (None, 192) or cfg['dip_guided']:
        parser.error('--baseline requires single-slice models with standard input windows')
    if not wide_cfg['wide_tunnel'] or wide_norm != norm:
        parser.error('--wide requires wide-tunnel checkpoints with the same normalization as --baseline')
    cube = np.load(sl.DATA_DIR / 'inpainting_cube_missed_data.npy', mmap_mode='r')
    mask = sl.hole_mask('inpainting', (1, *cube.shape[1:]))
    rows = SPLITS['inpainting'][1 if args.split == 'val' else 2]
    seed = (1000 if args.split == 'val' else 2000) + args.seed
    dataset = MatchedWindows(cube, mask, norm, rows, args.count, seed, include_trace=args.frequency_probe)
    if args.frequency_probe:
        evaluator = evaluate_frequency_blends
    elif args.polarity_probe:
        evaluator = evaluate_polarity_blends
    else:
        evaluator = evaluate
    results = evaluator(DataLoader(dataset, batch_size=args.bs), baseline, wide, norm, device)
    report = {'split': args.split, 'count': args.count, 'seed': args.seed,
              'baseline_checkpoints': args.baseline, 'wide_checkpoints': args.wide, 'results': results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
