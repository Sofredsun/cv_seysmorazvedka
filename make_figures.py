"""Картинки для README: кривые обучения, тестовые патчи, восстановленный куб.

python make_figures.py --task outpainting \
    --ckpt runs/outpainting_unet_up_k5/best.pt --hist runs/outpainting_unet_up_k5/hist.json \
    --hist_full runs/outpainting_unet_final/hist.json --restored results/outpainting_restored.npy
python make_figures.py --task inpainting \
    --ckpt runs/inpainting_unet/best.pt --hist runs/inpainting_unet/hist.json \
    --hist_full runs/inpainting_unet_final/hist.json --restored results/inpainting_restored.npy

Тестовые патчи - те же 400, что в compare.py (test-блок, seed 42); модель для них - отобранная по val
(не --full: та видела test-строки). Восстановленный куб - результат финальной модели (infer.py).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import torch

import seismic_lib as sl
from compare import baseline_pred, model_pred
from train_unet import make_datasets

# Срезы для показа реального пропуска: inline (axis0), z-срез, срез вдоль axis1 внутри пропуска
VIEWS = {
    'outpainting': {'i0': 230, 'z': 400, 'j_hole': 470, 'edge_w': 96, 'test_idx': [7, 150, 333]},
    # test_idx - окна с R U-Net на 15%, 50% и 85% квантилях по test-блоку; zoom - (j0, j1, z0, z1) вокруг пропуска
    'inpainting': {'i0': 290, 'z': 230, 'j_hole': 474, 'test_idx': [157, 312, 355],
                   'zoom': (300, 650, 130, 330), 'zoom_i0': [60, 290, 520]},
}


def _box(a, m2, color='r', lw=1):
    """Контур прямоугольного пропуска m2 (lateral, z) на картинке img.T."""
    rows, cols = np.where(m2.any(1))[0], np.where(m2.any(0))[0]
    a.add_patch(Rectangle((rows[0] - .5, cols[0] - .5), len(rows), len(cols),
                          fill=False, edgecolor=color, lw=lw))


def _clim(a):
    lo, hi = np.percentile(a, [1, 99])
    return float(lo), float(hi)


def fig_training(hist, hist_full, out):
    h = hist['hist']
    ep = np.arange(1, len(h['train_loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, h['train_loss'], label='train')
    ax[0].plot(ep, h['val_loss'], label='val')
    ax[1].plot(ep, h['train_r'], label='train R')
    ax[1].plot(ep, h['val_r'], label='val R_mean')
    if 'val_r_global' in h:
        ax[1].plot(ep, h['val_r_global'], label='val R_global')
    b = hist['best_epoch']
    ax[1].axvline(b, color='gray', ls=':', label=f'лучшая эпоха {b}')
    if hist_full is not None:
        hf = hist_full['hist']
        ax[0].plot(ep, hf['train_loss'], '--', color='C4', alpha=.7, label='train (--full)')
        ax[1].plot(ep, hf['train_r'], '--', color='C4', alpha=.7, label='train R (--full)')
    ax[0].set_title('loss (L1 + градиенты) по пропуску')
    ax[1].set_title('R по пропуску')
    for a in ax:
        a.set_xlabel('эпоха')
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


@torch.no_grad()
def fig_test_patches(task, cube, mask, norm, ckpt, idxs, out, device):
    model, cfg, _ = sl.load_model(ckpt, device)
    te = make_datasets(task, cube, mask, norm, 1, 400, 42, k=cfg['k'], stride=cfg['stride'])[2]
    fig, ax = plt.subplots(len(idxs), 4, figsize=(12, 3.6 * len(idxs)))
    for row, i in enumerate(idxs):
        x, y, m = (t[None] for t in te[i])
        p = model_pred(model, x.to(device), task, cfg['k'], tta=True).cpu()
        p = sl.composite(y, p, m)
        b = baseline_pred(task, y, m)
        r_p = sl.pearson_r_torch(p, y, m).item()
        r_b = sl.pearson_r_torch(b, y, m).item()
        inp = (y * (1 - m))[0, 0].numpy()
        lo, hi = _clim(y[0, 0].numpy())
        panels = [(inp, 'вход (пропуск = 0)'), (y[0, 0].numpy(), 'истина'),
                  (b[0, 0].numpy(), f'бейзлайн, R={r_b:.2f}'), (p[0, 0].numpy(), f'U-Net, R={r_p:.2f}')]
        m2 = m[0, 0].numpy() > 0.5
        for c, (img, title) in enumerate(panels):
            a = ax[row, c]
            a.imshow(img.T, cmap='gray', vmin=lo, vmax=hi, aspect='auto')
            if task == 'inpainting':
                _box(a, m2)
            else:
                a.axvline(img.shape[0] - int(m2[:, 0].sum()) - .5, color='r', lw=1)
            a.set_title(title, fontsize=10)
            a.set_xticks([])
            if c:
                a.set_yticks([])
        ax[row, 0].set_ylabel(f'патч {i}: z')
    fig.suptitle('test-блок: внутри красной рамки - синтетический пропуск 213 × 103' if task == 'inpainting'
                 else 'test-блок: справа от красной линии - синтетический пропуск (24 трассы)')
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_restored(task, cube, restored, mask, v, out):
    i0, z, j = v['i0'], v['z'], v['j_hole']
    n1 = cube.shape[1]
    lo, hi = _clim(cube[i0][~mask[i0]])
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1, 1]})
    for row, (c, name) in enumerate([(cube, 'вход'), (restored, 'восстановлено')]):
        ax[row, 0].imshow(c[i0].T, cmap='gray', vmin=lo, vmax=hi, aspect='auto')
        ax[row, 0].set_title(f'{name}: срез axis0={i0} (axis1 × z)')
        ax[row, 1].imshow(c[:, :, z], cmap='gray', vmin=lo, vmax=hi, aspect='auto')
        ax[row, 1].set_title(f'{name}: срез z={z} (axis0 × axis1)')
        ax[row, 2].imshow(c[:, j, :].T, cmap='gray', vmin=lo, vmax=hi, aspect='auto')
        ax[row, 2].set_title(f'{name}: axis1={j}, в пропуске (axis0 × z)')
        if task == 'inpainting':
            js, zs = np.where(mask[0].any(1))[0], np.where(mask[0].any(0))[0]
            _box(ax[row, 0], mask[i0], lw=.8)
            for jj in (js[0], js[-1] + 1):
                ax[row, 1].axvline(jj - .5, color='r', lw=.8)
            for zz in (zs[0], zs[-1] + 1):
                ax[row, 2].axhline(zz - .5, color='r', lw=.8)
        else:
            e = int(mask[0, :, 0].sum() // 2)
            for a in ax[row, :2]:
                a.axvline(e - .5, color='r', lw=.8)
                a.axvline(n1 - e - .5, color='r', lw=.8)
    fig.suptitle('реальный пропуск [:, 368:581, 179:282] - красные линии/рамка' if task == 'inpainting'
                 else 'реальный пропуск: красные линии - границы полос [0:24] и [457:481]')
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def fig_edges(cube, restored, mask, v, out):
    i0, w = v['i0'], v['edge_w']
    n1 = cube.shape[1]
    e = int(mask[0, :, 0].sum() // 2)
    lo, hi = _clim(cube[i0][~mask[i0]])
    fig, ax = plt.subplots(1, 4, figsize=(14, 6))
    parts = [(cube[i0, :w], 'вход, левый край', e), (restored[i0, :w], 'восстановлено, левый край', e),
             (cube[i0, n1 - w:], 'вход, правый край', w - e), (restored[i0, n1 - w:], 'восстановлено, правый край', w - e)]
    for a, (img, title, line) in zip(ax, parts):
        a.imshow(img.T, cmap='gray', vmin=lo, vmax=hi, aspect='auto')
        a.axvline(line - .5, color='r', lw=1)
        a.set_title(title, fontsize=10)
    fig.suptitle(f'срез axis0={i0}, крайние {w} трасс (axis1 × z)')
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_zoom(cube, restored, mask, v, out):
    """Inpainting: окрестность пропуска на нескольких срезах axis0, вход и восстановление."""
    j0, j1, z0, z1 = v['zoom']
    js, zs = np.where(mask[0].any(1))[0], np.where(mask[0].any(0))[0]
    ii = v['zoom_i0']
    fig, ax = plt.subplots(2, len(ii), figsize=(5 * len(ii), 8))
    for c, i0 in enumerate(ii):
        lo, hi = _clim(cube[i0][~mask[i0]])
        for r, (src, name) in enumerate([(cube, 'вход'), (restored, 'восстановлено')]):
            a = ax[r, c]
            a.imshow(src[i0, j0:j1, z0:z1].T, cmap='gray', vmin=lo, vmax=hi, aspect='auto',
                     extent=(j0 - .5, j1 - .5, z1 - .5, z0 - .5))
            a.add_patch(Rectangle((js[0] - .5, zs[0] - .5), len(js), len(zs), fill=False, edgecolor='r', lw=1))
            a.set_title(f'{name}: axis0={i0}', fontsize=10)
            a.set_xlabel('axis1')
        ax[0, c].set_ylabel('z')
    fig.suptitle('окрестность пропуска крупно (axis1 × z)')
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=list(VIEWS))
    ap.add_argument('--ckpt', required=True, help='модель, отобранная по val (для test-патчей)')
    ap.add_argument('--hist', required=True, help='hist.json этой модели')
    ap.add_argument('--hist_full', default=None, help='hist.json финального --full прогона')
    ap.add_argument('--restored', required=True, help='восстановленный куб (infer.py)')
    ap.add_argument('--out', default='figures')
    args = ap.parse_args()

    sl.seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t, v = args.task, VIEWS[args.task]

    cube = sl.load_cube(t)
    mask = sl.hole_mask(t, cube.shape)
    norm = sl.fit_norm(cube, mask)
    restored = np.load(args.restored, mmap_mode='r')

    hist = json.load(open(args.hist))
    hist_full = json.load(open(args.hist_full)) if args.hist_full else None
    fig_training(hist, hist_full, out / f'{t}_training.png')
    fig_test_patches(t, cube, mask, norm, args.ckpt, v['test_idx'], out / f'{t}_test_patches.png', device)
    fig_restored(t, cube, restored, mask, v, out / f'{t}_restored_views.png')
    if t == 'inpainting':
        fig_zoom(cube, restored, mask, v, out / f'{t}_zoom.png')
    else:
        fig_edges(cube, restored, mask, v, out / f'{t}_edges.png')
    print('saved:', *sorted(p.name for p in out.glob(f'{t}_*.png')))


if __name__ == '__main__':
    main()
