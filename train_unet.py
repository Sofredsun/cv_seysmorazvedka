import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import seismic_lib as sl

AMP = False

SPLITS = {
    'inpainting': (range(0, 400), range(416, 496), range(500, 580)),
    'outpainting': (range(0, 320), range(336, 396), range(400, 460)),
}


def make_datasets(task, cube, mask, norm, n_train, n_val, seed, k=1, stride=1, full=False):
    """
    full=True: train по всем строкам axis0 (для финальной модели); val/test тогда
    пересекаются с train и служат только для контроля, а не для честной оценки.
    """
    tr, va, te = SPLITS[task]
    if full:
        tr = range(0, cube.shape[0])
    cls = sl.InpaintPatches if task == 'inpainting' else sl.OutpaintPatches
    mk = lambda rows, n, src, aug, s, fixed: cls(
        cube, mask, norm, rows, n_samples=n, sources=src, augment=aug, seed=s, fixed=fixed,
        k=k, stride=stride)
    return (mk(tr, n_train, ('a0', 'a1'), True, seed, False),
            mk(va, n_val, ('a0',), False, 1000 + seed, True),
            mk(te, n_val, ('a0',), False, 2000 + seed, True))


@torch.no_grad()
def evaluate(model, loader, crit, device):
    """
    r_mean - среднее R по патчам (R считается внутри каждой дыры отдельно);
    r_global - один R по всем вокселам всех дыр сразу, как R по восстановленной области куба.
    r_global копится потоково в float64 (суммы p, t, p^2, t^2, pt), без хранения предсказаний.
    """
    model.eval()
    tot, rs = 0.0, []
    acc = torch.zeros(6, dtype=torch.float64, device=device)
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
            p = model(x)
        p = p.float()
        loss, _ = crit(p, y, m)
        tot += loss.item() * x.size(0)
        rs.append(sl.pearson_r_torch(p, y, m).cpu())
        sel = m > 0.5
        pv, tv = p[sel].double(), y[sel].double()
        acc += torch.stack([pv.new_tensor(pv.numel()), pv.sum(), tv.sum(),
                            (pv * pv).sum(), (tv * tv).sum(), (pv * tv).sum()])
    model.train()
    n, sp, st, spp, stt, spt = acc.tolist()
    cov = spt - sp * st / n
    r_global = cov / (((spp - sp * sp / n) * (stt - st * st / n)) ** 0.5 + 1e-12)
    return tot / len(loader.dataset), torch.cat(rs).mean().item(), r_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--n_train', type=int, default=3000)
    ap.add_argument('--n_val', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--up', default='transpose', choices=['transpose', 'upsample'])
    ap.add_argument('--k', type=int, default=1, help='число срезов во входе (нечётное), 1 = 2D')
    ap.add_argument('--stride', type=int, default=1, help='шаг между соседними срезами')
    ap.add_argument('--full', action='store_true', help='обучать на всех данных, сохранить последнюю эпоху')
    ap.add_argument('--tag', default='', help='суффикс папки прогона')
    ap.add_argument('--out', default='runs')
    args = ap.parse_args()

    sl.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cube = sl.load_cube(args.task)
    mask = sl.hole_mask(args.task, cube.shape)
    norm = sl.fit_norm(cube, mask)
    tr_ds, va_ds, te_ds = make_datasets(args.task, cube, mask, norm, args.n_train, args.n_val, args.seed,
                                        k=args.k, stride=args.stride, full=args.full)
    tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=False)
    va = DataLoader(va_ds, batch_size=args.bs)
    te = DataLoader(te_ds, batch_size=args.bs)

    model = sl.UNet(in_ch=args.k + 1, base=args.base, up_mode=args.up).to(device)
    crit = sl.HoleLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr))
    scaler = torch.amp.GradScaler(enabled=AMP)

    out = Path(args.out) / f'{args.task}_unet{args.tag}'
    meta = {'base': args.base, 'k': args.k, 'stride': args.stride, 'up_mode': args.up,
            'norm': [norm.mean, norm.std], 'full': args.full}
    out.mkdir(parents=True, exist_ok=True)
    # В --full val пересекается с train: это лишь контроль сходимости, не оценка -> другие ключи/подпись
    vk = 'ctrl' if args.full else 'val'
    hist = {'train_loss': [], 'train_r': [], f'{vk}_loss': [], f'{vk}_r': [], f'{vk}_r_global': []}
    if args.full:
        print('--full: строки val входят в train, "ctrl r" - НЕ оценка качества '
              '(для отчёта брать test R из прогона без --full)', flush=True)
    best = -1.0
    print(f'params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, norm {norm}', flush=True)
    for ep in range(1, args.epochs + 1):
        t0, tl, trs = time.time(), 0.0, []
        for x, y, m in tr:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
                p = model(x)
            loss, info = crit(p.float(), y, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * x.size(0)
            trs.append(info['r'])
        vl, vr, vrg = evaluate(model, va, crit, device)
        hist['train_loss'].append(tl / len(tr_ds)); hist['train_r'].append(float(np.mean(trs)))
        hist[f'{vk}_loss'].append(vl); hist[f'{vk}_r'].append(vr); hist[f'{vk}_r_global'].append(vrg)
        if not args.full and vr > best:  # В --full выбор по val некорректен -> best.pt не пишем
            best = vr
            torch.save({'state_dict': model.state_dict(), 'epoch': ep, **meta}, out / 'best.pt')
        print(f'ep {ep:3d} train {hist["train_loss"][-1]:.4f} r {hist["train_r"][-1]:.4f} | '
              f'{vk} {vl:.4f} r {vr:.4f} rg {vrg:.4f} | {time.time() - t0:.0f}s', flush=True)

    if args.full:  # val пересекается с train -> выбор по val некорректен, берем последнюю эпоху
        torch.save({'state_dict': model.state_dict(), 'epoch': args.epochs, **meta}, out / 'final.pt')
        print(f'saved final.pt (epoch {args.epochs})', flush=True)
        json.dump({'hist': hist, 'args': vars(args),
                   'note': 'full: ctrl_* computed on rows included in train; not a quality estimate'},
                  open(out / 'hist.json', 'w'), indent=1)
        return
    ck = torch.load(out / 'best.pt', map_location=device)
    model.load_state_dict(ck['state_dict'])
    tl_, tr_, trg = evaluate(model, te, crit, device)
    print(f'TEST (best ep {ck["epoch"]}) loss {tl_:.4f} R {tr_:.4f} R_global {trg:.4f}', flush=True)
    json.dump({'hist': hist, 'test_r': tr_, 'test_r_global': trg, 'best_epoch': ck['epoch'],
               'args': vars(args)},
              open(out / 'hist.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
