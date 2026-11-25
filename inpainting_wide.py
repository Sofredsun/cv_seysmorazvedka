import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import seismic_lib as sl


PATCH = (384, 256)
HOLE = (213, 103)


class WideTunnelPatches(Dataset):
    def __init__(self, cube, real_mask, norm, rows, size, seed, training,
                 extra_augmentation=False, paired=False, clean_depth_fraction=0.0,
                 crossline_fraction=0.5, fixed_traces=(100, 640)):
        self.cube, self.norm, self.rows = cube, norm, rows
        self.real = real_mask[0]
        self.size, self.seed, self.training = size, seed, training
        self.extra_augmentation = extra_augmentation
        self.paired = paired
        self.clean_depth_fraction = clean_depth_fraction
        self.crossline_fraction = crossline_fraction
        self.fixed_traces = fixed_traces
        self.rng = np.random.default_rng(seed)
        self.offsets = np.arange(-2, 4 if paired else 3)
        bad = self.real.any(axis=1)
        centers = np.arange(2, len(bad) - self.offsets[-1])
        self.clean_crosslines = centers[~bad[centers[:, None] + self.offsets].any(axis=1)]
        if training and clean_depth_fraction:
            origins = np.arange(len(bad) - PATCH[0] + 1)
            overlap = np.convolve(bad.astype(int), np.ones(PATCH[0], dtype=int), mode='valid')
            # Известные участки по краям имеют ширину 368 и 370 трасс, чуть меньше ширины окна
            # Если на краю окна замаскировано не более 24 трасс, по обе стороны искусственного пропуска остаются видимые данные
            self.clean_origins = origins[overlap <= 24]
            self.tunnel_depth = int(np.flatnonzero(self.real.any(axis=0))[0])
            if not self.clean_origins.size:
                raise ValueError('No inline window with at most 24 masked traces')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rng = self.rng if self.training else np.random.default_rng(self.seed * 100003 + idx)
        h, w = PATCH
        hh, hw = HOLE
        if self.training and self.clean_depth_fraction and rng.random() < self.clean_depth_fraction:
            i = rng.integers(self.rows.start + 2, self.rows.stop - self.offsets[-1])
            j = rng.choice(self.clean_origins)
            a = (h - hh) // 2 + rng.integers(-24, 25)
            b = (w - hw) // 2
            z = self.tunnel_depth - b
            stack = self.cube[i + self.offsets, j:j + h, z:z + w]
            actual = self.real[j:j + h, z:z + w]
        elif self.training and self.extra_augmentation and rng.random() < self.crossline_fraction:
            i = rng.integers(self.rows.start, self.rows.stop - h + 1)
            j = rng.choice(self.clean_crosslines)
            z = rng.integers(0, self.cube.shape[2] - w + 1)
            a = (h - hh) // 2 + rng.integers(-24, 25)
            b = (w - hw) // 2 + rng.integers(-24, 25)
            stack = self.cube[i:i + h, j + self.offsets, z:z + w].transpose(1, 0, 2)
            actual = np.zeros(PATCH, dtype=bool)
        else:
            if self.training:
                i = rng.integers(self.rows.start + 2, self.rows.stop - self.offsets[-1])
                while True:
                    j = rng.integers(0, self.cube.shape[1] - h + 1)
                    z = rng.integers(0, self.cube.shape[2] - w + 1)
                    a = (h - hh) // 2 + rng.integers(-24, 25)
                    b = (w - hw) // 2 + rng.integers(-24, 25)
                    if not self.real[j + a:j + a + hh, z + b:z + b + hw].any():
                        break
            else:
                i = self.rows.start + 2 + (idx // len(self.fixed_traces)) % (len(self.rows) - 4)
                start = self.fixed_traces[idx % len(self.fixed_traces)]
                a, b = (h - hh) // 2, (w - hw) // 2
                j, z = start - a, 179 - b
            stack = self.cube[i + self.offsets, j:j + h, z:z + w]
            actual = self.real[j:j + h, z:z + w]

        target_mask = np.zeros(PATCH, dtype=bool)
        target_mask[a:a + hh, b:b + hw] = True
        stack = self.norm.fwd(np.asarray(stack, dtype=np.float32))
        if self.training:
            if rng.random() < 0.5:
                stack, actual, target_mask = stack[:, ::-1], actual[::-1], target_mask[::-1]
            if rng.random() < 0.5:
                stack = stack[::-1]
            if self.extra_augmentation:
                stack *= rng.choice((-1, 1)) * np.exp(rng.uniform(-0.2, 0.2))

        input_mask = (actual | target_mask).astype(np.float32)
        supervision = torch.from_numpy(target_mask[None].astype(np.float32))
        if self.paired:
            first_x, first_y, _ = sl._pack(stack[:-1], input_mask)
            second_x, second_y, _ = sl._pack(stack[1:], input_mask)
            return torch.stack((first_x, second_x)), torch.stack((first_y, second_y)), supervision
        x, y, _ = sl._pack(stack, input_mask)
        return x, y, supervision


class WideTunnelLoss(nn.Module):
    def forward(self, pred, target, mask):
        l1 = ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1)
        correlation_loss = sl.HoleLoss.correlation_loss(pred, target, mask)
        return l1 + correlation_loss, {'r': float(1 - correlation_loss.detach())}

