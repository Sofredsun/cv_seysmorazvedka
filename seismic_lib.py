from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent / 'data'


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Данные и маски
def load_cube(task: str) -> np.ndarray:
    """task: 'inpainting' | 'outpainting'. Возвращает float32 массив в памяти."""
    return np.load(DATA_DIR / f'{task}_cube_missed_data.npy')


def hole_mask(task: str, shape: tuple[int, int, int]) -> np.ndarray:
    """Булева маска реального пропуска (True = данных нет), по описанию задания"""
    m = np.zeros(shape, dtype=bool)
    if task == 'inpainting':
        m[:, 368:581, 179:282] = True
    elif task == 'outpainting':
        m[:, 0:24, :] = True
        m[:, shape[1] - 24:, :] = True
    else:
        raise ValueError(task)
    return m


@dataclass
class Norm:
    mean: float
    std: float

    def fwd(self, x):
        return (x - self.mean) / self.std

    def inv(self, x):
        return x * self.std + self.mean


def fit_norm(cube: np.ndarray, mask: np.ndarray) -> Norm:
    known = cube[::2][~mask[::2]]
    return Norm(float(known.mean()), float(known.std()))


def neighbor_offsets(k: int, stride: int) -> np.ndarray:
    """Смещения соседних срезов, центр - в середине: k=5, stride=2 -> [-4,-2,0,2,4]."""
    if k % 2 == 0:
        raise ValueError('k должно быть нечётным')
    return (np.arange(k) - k // 2) * stride


# Метрики
def pearson_r(pred, target, eps: float = 1e-8) -> float:
    """Коэффициент корреляции Пирсона по всем переданным значениям."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    p = p - p.mean()
    t = t - t.mean()
    return float((p * t).sum() / (np.sqrt((p * p).sum() * (t * t).sum()) + eps))


def pearson_r_torch(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """R по маске отдельно для каждого элемента батча -> тензор (B,)."""
    m = mask.flatten(1)
    n = m.sum(1).clamp_min(1)
    p = pred.flatten(1)
    t = target.flatten(1)
    pm = (p * m).sum(1) / n
    tm = (t * m).sum(1) / n
    pc = (p - pm[:, None]) * m
    tc = (t - tm[:, None]) * m
    return (pc * tc).sum(1) / (torch.sqrt((pc ** 2).sum(1) * (tc ** 2).sum(1)) + eps)


# Self-supervised датасеты: патч из известной части + искусственный пропуск
def _pack(stack: np.ndarray, m: np.ndarray):
    """
    stack: (k, H, W) нормированные срезы, центральный - stack[k//2]; m: (H, W) маска.
    Возвращает x = [k срезов с нулями в пропуске, маска], y = центральный срез, маска.
    """
    stack = np.ascontiguousarray(stack, dtype=np.float32)
    k = stack.shape[0]
    x = np.concatenate([stack * (1 - m)[None], m[None]], 0)
    return (torch.from_numpy(x).float(),
            torch.from_numpy(stack[k // 2][None].copy()).float(),
            torch.from_numpy(m[None].astype(np.float32)).float())


class _PatchBase(Dataset):
    def __init__(self, cube, real_mask, norm, rows0, n_samples, sources, augment, seed, fixed,
                 k, stride):
        self.cube, self.real_mask, self.norm = cube, real_mask, norm
        self.rows0 = np.asarray(rows0)
        self.n, self.sources, self.augment = n_samples, sources, augment
        self.seed, self.fixed = seed, fixed
        self.offs = neighbor_offsets(k, stride)
        bad1 = real_mask.any(axis=(0, 2))
        self.bad1 = bad1
        self.clean1 = np.where(~bad1)[0]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def _rng(self, idx):
        return np.random.default_rng(self.seed * 100003 + idx) if self.fixed else self.rng

    def _idx0(self, i):
        """Соседи по axis0 (с отражением у границ куба)."""
        n0 = self.cube.shape[0]
        idx = i + self.offs
        idx = np.where(idx < 0, -idx, idx)
        return np.where(idx >= n0, 2 * (n0 - 1) - idx, idx)

    def _idx1(self, j):
        """Соседи по axis1 только по чистым трассам: при попадании в пропуск/за край - зеркально."""
        idx = j + self.offs
        bad = (idx < 0) | (idx >= len(self.bad1))
        bad |= self.bad1[np.clip(idx, 0, len(self.bad1) - 1)]
        idx = np.where(bad, j - self.offs, idx)
        return idx

    def _augment(self, stack, rng):
        if self.augment:
            if rng.random() < 0.5:
                stack = stack[:, ::-1]
            if rng.random() < 0.5:
                stack = stack[::-1]
            stack = stack * rng.uniform(0.85, 1.15)
        return stack


class InpaintPatches(_PatchBase):
    """
    Плоскость (lateral, z) с прямоугольной дырой hole_hw посреди контекста.

    Источники плоскостей (обе латеральные оси геологически равноправны):
      'a0' - срезы cube[i, :, :]  (как на инференсе), i из rows0, соседи по axis0;
      'a1' - срезы cube[:, j, :]  (аугментация), латераль = axis0 внутри rows0, соседи по axis1.
    Патч берётся только там, где нет реального пропуска.
    """

    def __init__(self, cube, real_mask, norm: Norm, rows0, patch_hw=(352, 192),
                 hole_hw=(213, 103), n_samples=4000, sources=('a0', 'a1'),
                 augment=True, seed=0, fixed=False, k=1, stride=1):
        super().__init__(cube, real_mask, norm, rows0, n_samples, sources, augment, seed, fixed, k, stride)
        self.ph, self.pw = patch_hw
        self.hh, self.hw = hole_hw
        # Допустимые старты окна по axis1, при которых окно целиком в чистой зоне
        cs = np.concatenate([[0], np.cumsum(self.bad1)])
        starts = np.arange(len(self.bad1) - self.ph + 1)
        self.j0_ok = starts[cs[starts + self.ph] - cs[starts] == 0]

    def _sample_stack(self, rng):
        src = self.sources[rng.integers(len(self.sources))]
        z0 = rng.integers(0, self.cube.shape[2] - self.pw + 1)
        zs = slice(z0, z0 + self.pw)
        if src == 'a0':
            i = rng.choice(self.rows0)
            j0 = rng.choice(self.j0_ok)
            return self.cube[self._idx0(i), j0:j0 + self.ph, zs]
        j = rng.choice(self.clean1)
        lo, hi = self.rows0.min(), self.rows0.max() + 1
        i0 = rng.integers(lo, hi - self.ph + 1)
        return self.cube[i0:i0 + self.ph, self._idx1(j), zs].transpose(1, 0, 2)

    def __getitem__(self, idx):
        rng = self._rng(idx)
        stack = self.norm.fwd(np.array(self._sample_stack(rng), dtype=np.float32))
        stack = self._augment(stack, rng)
        # Дыра: около центра со случайным смещением
        cy = (self.ph - self.hh) // 2 + rng.integers(-24, 25)
        cx = (self.pw - self.hw) // 2 + rng.integers(-16, 17)
        cy = int(np.clip(cy, 8, self.ph - self.hh - 8))
        cx = int(np.clip(cx, 8, self.pw - self.hw - 8))
        m = np.zeros(stack.shape[1:], dtype=np.float32)
        m[cy:cy + self.hh, cx:cx + self.hw] = 1
        return _pack(stack, m)


class OutpaintPatches(_PatchBase):
    """
    Окно (lateral=width, z), у которого последние edge трасс справа скрыты.

    'a0' - срезы cube[i, j0:j0+width, z0:z0+depth] (как на инференсе), соседи по axis0;
    'a1' - аугментация: латераль = axis0, cube[i0:i0+width, j, z], соседи по axis1.
    Левый край обрабатывается на инференсе зеркальным флипом.
    """

    def __init__(self, cube, real_mask, norm: Norm, rows0, width=128, edge=24,
                 depth=256, n_samples=4000, sources=('a0', 'a1'), augment=True,
                 seed=0, fixed=False, k=1, stride=1):
        super().__init__(cube, real_mask, norm, rows0, n_samples, sources, augment, seed, fixed, k, stride)
        self.w, self.e, self.d = width, edge, depth
        self.j_lo, self.j_hi = self.clean1.min(), self.clean1.max() + 1

    def __getitem__(self, idx):
        rng = self._rng(idx)
        z0 = rng.integers(0, self.cube.shape[2] - self.d + 1)
        zs = slice(z0, z0 + self.d)
        src = self.sources[rng.integers(len(self.sources))]
        if src == 'a0':
            i = rng.choice(self.rows0)
            j0 = rng.integers(self.j_lo, self.j_hi - self.w + 1)
            win = self.cube[self._idx0(i), j0:j0 + self.w, zs]
        else:
            j = rng.choice(self.clean1)
            lo, hi = self.rows0.min(), self.rows0.max() + 1
            i0 = rng.integers(lo, hi - self.w + 1)
            win = self.cube[i0:i0 + self.w, self._idx1(j), zs].transpose(1, 0, 2)
        stack = self.norm.fwd(np.array(win, dtype=np.float32))
        stack = self._augment(stack, rng)
        m = np.zeros(stack.shape[1:], dtype=np.float32)
        m[-self.e:, :] = 1
        return _pack(stack, m)


# Модель: U-Net, регрессия 1 канала
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    """Билинейный апсемплинг x2 + свертка 3x3: без шахматных артефактов ConvTranspose."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))


class UNet(nn.Module):
    """
    Вход: [k срезов с нулями в пропуске, маска], выход: центральный срез целиком.
    Итог композитится: известное берется из входа, пропуск - из выхода модели.

    up_mode: 'transpose' (ConvTranspose2d, первая версия) | 'upsample' (bilinear + conv).
    """

    def __init__(self, in_ch=2, base=32, depth=4, up_mode='transpose'):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.downs.append(DoubleConv(c, ch))
            c = ch
        self.bottleneck = DoubleConv(chs[-2], chs[-1])
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2) if up_mode == 'transpose'
                            else UpConv(ch * 2, ch))
            self.decs.append(DoubleConv(ch * 2, ch))
        self.head = nn.Conv2d(chs[0], 1, 1)
        self.mult = 2 ** depth

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % self.mult, (-W) % self.mult
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, s in zip(self.ups, self.decs, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)[..., :H, :W]


def model_config(ck: dict) -> dict:
    """Конфиг модели из чекпоинта (старые чекпоинты - без k/stride/up_mode)."""
    return {'base': ck.get('base', 32), 'k': ck.get('k', 1), 'stride': ck.get('stride', 1),
            'up_mode': ck.get('up_mode', 'transpose')}


def load_model(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    cfg = model_config(ck)
    model = UNet(in_ch=cfg['k'] + 1, base=cfg['base'], up_mode=cfg['up_mode']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, cfg, Norm(*ck['norm'])


def composite(inp_known: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return inp_known * (1 - mask) + pred * mask


class HoleLoss(nn.Module):
    """
    L1 по пропуску + w_g * L1 разницы градиентов (по глубине и латерали) по пропуску.

    Коэффициент корреляции в loss не используется -
    R считается только как метрика (без градиента).
    """

    def __init__(self, w_g=0.5):
        super().__init__()
        self.w_g = w_g

    def forward(self, pred, tgt, m):
        n = m.sum().clamp_min(1)
        l1 = ((pred - tgt).abs() * m).sum() / n
        gz = ((torch.diff(pred, dim=-1) - torch.diff(tgt, dim=-1)).abs() * m[..., 1:]).sum() / n
        gx = ((torch.diff(pred, dim=-2) - torch.diff(tgt, dim=-2)).abs() * m[..., 1:, :]).sum() / n
        with torch.no_grad():
            r = pearson_r_torch(pred, tgt, m).mean()
        return l1 + self.w_g * (gz + gx), {'l1': l1.item(), 'r': r.item()}


# Инференс на реальные пропуски
@torch.no_grad()
def _predict_windows(model, stacks, m, device, bs, tta_lateral):
    """stacks: (N, k, H, W) нормированные, m: (H, W) маска. Возвращает (N, H, W) предсказание."""

    out = np.empty((stacks.shape[0],) + stacks.shape[2:], dtype=np.float32)
    mt = torch.from_numpy(m.astype(np.float32)).to(device)
    for s in range(0, stacks.shape[0], bs):
        xs = torch.from_numpy(np.ascontiguousarray(stacks[s:s + bs])).to(device)
        x = torch.cat([xs * (1 - mt), mt.expand(xs.shape[0], 1, *mt.shape)], 1)
        preds = [model(x)]
        # TTA: флип порядка соседних срезов (имеет смысл для k > 1)
        if xs.shape[1] > 1:
            xf = torch.cat([xs.flip(1) * (1 - mt), mt.expand(xs.shape[0], 1, *mt.shape)], 1)
            preds.append(model(xf))
        # TTA: флип латерали - только если маска симметрична при флипе (inpainting)
        if tta_lateral:
            preds.append(model(x.flip(2)).flip(2))
        out[s:s + bs] = torch.stack(preds).mean(0)[:, 0].cpu().numpy()
    return out


def _stacks_axis0(cube_n, offs, js, zs):
    """Для каждого i: стопка соседей по axis0 (с отражением у границ) в окне (js, zs)."""
    n0 = cube_n.shape[0]
    idx = np.arange(n0)[:, None] + offs[None]
    idx = np.where(idx < 0, -idx, idx)
    idx = np.where(idx >= n0, 2 * (n0 - 1) - idx, idx)
    win = cube_n[:, js, zs]
    return win[idx]


def restore_cube(task, cube, model, cfg, norm: Norm, device='cuda', bs=8, tta=True,
                 patch_hw=(352, 192), width=128) -> np.ndarray:
    """Заполняет реальные пропуски куба. Известные значения не меняются (бит-в-бит)."""
    mask = hole_mask(task, cube.shape)
    offs = neighbor_offsets(cfg['k'], cfg['stride'])
    cube_n = norm.fwd(cube.astype(np.float32))
    restored = cube.copy()
    if task == 'inpainting':
        js_hole, zs_hole = np.where(mask[0].any(1))[0], np.where(mask[0].any(0))[0]
        h, w = len(js_hole), len(zs_hole)
        j0 = js_hole[0] - (patch_hw[0] - h) // 2
        z0 = zs_hole[0] - (patch_hw[1] - w) // 2
        js, zs = slice(j0, j0 + patch_hw[0]), slice(z0, z0 + patch_hw[1])
        stacks = _stacks_axis0(cube_n, offs, js, zs)
        m = mask[0, js, zs]
        pred = _predict_windows(model, stacks, m, device, bs, tta_lateral=tta)
        win = restored[:, js, zs]
        win[:, m] = norm.inv(pred[:, m])
    elif task == 'outpainting':
        e = int(mask[0, :, 0].sum() // 2)
        n1 = cube.shape[1]
        # Правый край: окно [n1-width, n1), скрыты последние e трасс
        js = slice(n1 - width, n1)
        stacks = _stacks_axis0(cube_n, offs, js, slice(None))
        m = mask[0, js, :]
        pred = _predict_windows(model, stacks, m, device, bs, tta_lateral=False)
        win = restored[:, js, :]
        win[:, m] = norm.inv(pred[:, m])
        # Левый край: окно [0, width), зеркально по axis1 -> пропуск справа
        js = slice(0, width)
        stacks = _stacks_axis0(cube_n, offs, js, slice(None))[:, :, ::-1]
        m = mask[0, js, :][::-1]
        pred = _predict_windows(model, stacks, m, device, bs, tta_lateral=False)[:, ::-1]
        win = restored[:, js, :]
        win[:, m[::-1]] = norm.inv(pred[:, m[::-1]])
    else:
        raise ValueError(task)
    return restored


# Бейзлайны (без обучения)
def baseline_inpaint_linear(plane: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Линейная интерполяция по латерали (ось 0 плоскости) между краями дыры на каждой глубине."""
    out = plane.copy()
    rows = np.where(m.any(1))[0]
    cols = np.where(m.any(0))[0]
    r0, r1 = rows.min() - 1, rows.max() + 1
    t = (np.arange(r0 + 1, r1) - r0) / (r1 - r0)
    for c in cols:
        out[r0 + 1:r1, c] = plane[r0, c] * (1 - t) + plane[r1, c] * t
    return out


def baseline_outpaint_nearest(plane: np.ndarray, edge: int) -> np.ndarray:
    """Копия последней известной трассы в скрытую полосу (правый край)."""
    out = plane.copy()
    out[-edge:] = plane[-edge - 1]
    return out
