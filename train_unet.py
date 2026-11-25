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
EDGE_TEST_SPLIT = (range(90, 490), range(500, 580), range(0, 80))
EDGE_TEST_TRACES = (86, 650)


def curriculum_widths(epochs, target_width):
    phase = max(1, epochs // 8)
    warmup = [width for width in (64, 128, 176) for _ in range(phase)]
    return warmup + [target_width] * (epochs - len(warmup))


def make_datasets(task, cube, mask, norm, n_train, n_val, seed, k=1, stride=1, full=False,
                  context_z=None, dip_guided=False, depth_focus=0.0,
                  wide_tunnel=False, wide_augment=False, wide_paired=False, inpaint_paired=False,
                  wide_clean_depth_fraction=0.0, wide_crossline_fraction=0.5, edge_test_split=False):
    """
    full=True: train по всем строкам axis0 (для финальной модели); val/test тогда
    пересекаются с train и служат только для контроля, а не для честной оценки.
    """
    tr, va, te = EDGE_TEST_SPLIT if edge_test_split else SPLITS[task]
    if full:
        tr = range(0, cube.shape[0])
    if wide_tunnel:
        from inpainting_wide import WideTunnelPatches
        fixed = {'fixed_traces': EDGE_TEST_TRACES} if edge_test_split else {}
        if edge_test_split:
            n_val = len(EDGE_TEST_TRACES) * (len(va) - 4)
        return (WideTunnelPatches(cube, mask, norm, tr, n_train, seed, True, wide_augment, wide_paired,
                                  clean_depth_fraction=wide_clean_depth_fraction,
                                  crossline_fraction=wide_crossline_fraction),
                WideTunnelPatches(cube, mask, norm, va, n_val, 1000 + seed, False, **fixed),
                WideTunnelPatches(cube, mask, norm, te, n_val, 2000 + seed, False, **fixed))
    cls = sl.InpaintPatches if task == 'inpainting' else sl.OutpaintPatches
    def options(augment):
        if task == 'inpainting':
            return dict(context_z=context_z, dip_guided=dip_guided,
                        depth_focus=depth_focus if augment else 0.0,
                        paired=inpaint_paired and augment)
        return {}
    mk = lambda rows, n, src, aug, s, fixed: cls(
        cube, mask, norm, rows, n_samples=n, sources=src, augment=aug, seed=s, fixed=fixed,
        k=k, stride=stride, **options(aug))
    train_sources = ('a0',) if inpaint_paired else ('a0', 'a1')
    return (mk(tr, n_train, train_sources, True, seed, False),
            mk(va, n_val, ('a0',), False, 1000 + seed, True),
            mk(te, n_val, ('a0',), False, 2000 + seed, True))


@torch.no_grad()
def evaluate(model, loader, crit, device, ranks=None, lateral_tta=False):
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
            if lateral_tta:
                p = (p + model(x.flip(2)).flip(2)) / 2
        p = p.float()
        if ranks is not None:
            ranks.add(p, y, m)
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
    ap.add_argument('--context-z', type=int, help='inpainting window depth, default 192')
    ap.add_argument('--dip-guided', action='store_true', help='add a directional fill input channel')
    ap.add_argument('--wide-tunnel', action='store_true',
                    help='five-level 2.5D inpainting with input and target masks')
    ap.add_argument('--edge-test-split', action='store_true',
                    help='train wide-tunnel from scratch with test at the low-y edge and fixed-depth evaluation')
    ap.add_argument('--wide-augment', action='store_true',
                    help='enable crosslines, polarity flips and gain for --wide-tunnel')
    ap.add_argument('--wide-clean-depth-fraction', type=float, default=0.0,
                    help='fraction of wide training patches at tunnel depth with minimal real-mask overlap')
    ap.add_argument('--wide-crossline-fraction', type=float, default=0.5,
                    help='crossline sampling probability after the clean-depth branch with --wide-augment')
    ap.add_argument('--wide-neighbor-weight', type=float,
                    help='train on paired wide-tunnel patches with a slice-difference loss; 0 is a paired control')
    ap.add_argument('--neighbor-weight', type=float,
                    help='train on adjacent pairs using the standard U-Net; 0 is a paired control')
    ap.add_argument('--rank-weight', type=float, default=0, help='weight of the pairwise rank loss')
    ap.add_argument('--corr-weight', type=float, default=0, help='masked Pearson loss weight')
    ap.add_argument('--center-corr-weight', type=float, default=0, help='Pearson loss weight inside the gap center')
    ap.add_argument('--depth-focus', type=float, default=0,
                    help='fraction of training gaps sampled near the true inpainting depth')
    ap.add_argument('--width-curriculum', action='store_true',
                    help='increase the training gap width from 64 to 213 traces')
    ap.add_argument('--joint-ensemble', nargs='+', metavar='CKPT',
                    help='fine-tune members jointly using the loss on their mean prediction')
    ap.add_argument('--init', help='initialize inpainting weights from an existing checkpoint')
    args = ap.parse_args()
    rank_eval = (args.context_z is not None or args.dip_guided
                 or args.rank_weight != 0 or args.init is not None
                 or args.corr_weight != 0 or args.center_corr_weight != 0 or args.depth_focus != 0
                 or args.width_curriculum
                 or args.wide_tunnel or args.neighbor_weight is not None or args.joint_ensemble is not None)
    if rank_eval and args.task != 'inpainting':
        ap.error('experimental options are for inpainting only')
    if args.joint_ensemble and (len(args.joint_ensemble) < 2 or args.k != 1 or args.stride != 1
                                or args.init
                                or args.wide_tunnel or args.dip_guided or args.neighbor_weight is not None):
        ap.error('--joint-ensemble requires at least two standard single-slice checkpoints')
    if args.neighbor_weight is not None:
        if not np.isfinite(args.neighbor_weight) or args.neighbor_weight < 0:
            ap.error('--neighbor-weight must be finite and nonnegative')
        if args.wide_tunnel or args.dip_guided:
            ap.error('--neighbor-weight requires the standard inpainting U-Net')
    if args.wide_augment and not args.wide_tunnel:
        ap.error('--wide-augment requires --wide-tunnel')
    if args.edge_test_split and (not args.wide_tunnel or args.init or args.full or args.joint_ensemble):
        ap.error('--edge-test-split requires --wide-tunnel and training from scratch on the training block')
    if not 0 <= args.wide_clean_depth_fraction <= 1:
        ap.error('--wide-clean-depth-fraction must be between 0 and 1')
    if args.wide_clean_depth_fraction and not args.wide_tunnel:
        ap.error('--wide-clean-depth-fraction requires --wide-tunnel')
    if not 0 <= args.wide_crossline_fraction <= 1:
        ap.error('--wide-crossline-fraction must be between 0 and 1')
    if args.wide_crossline_fraction != 0.5 and not args.wide_augment:
        ap.error('--wide-crossline-fraction requires --wide-augment')
    if args.wide_neighbor_weight is not None and (
            not args.wide_tunnel or not np.isfinite(args.wide_neighbor_weight) or args.wide_neighbor_weight < 0):
        ap.error('--wide-neighbor-weight requires --wide-tunnel and a finite nonnegative weight')
    if args.wide_tunnel and (args.k != 5 or args.stride != 1 or args.context_z is not None
                             or args.depth_focus or args.width_curriculum
                             or args.dip_guided
                             or args.rank_weight or args.corr_weight or args.center_corr_weight):
        ap.error('--wide-tunnel requires --k 5, --stride 1 and its own sampling and loss')
    if args.width_curriculum and args.epochs < 8:
        ap.error('--width-curriculum requires standard inpainting and at least 8 epochs')
    if not 0 <= args.depth_focus <= 1:
        ap.error('--depth-focus must be between 0 and 1')
    if args.context_z is not None and not 192 <= args.context_z <= 462:
        ap.error('--context-z must be between 192 and 462')
    if not np.isfinite(args.rank_weight) or args.rank_weight < 0:
        ap.error('--rank-weight must be finite and nonnegative')
    if any(not np.isfinite(w) or w < 0 for w in (args.corr_weight, args.center_corr_weight)):
        ap.error('correlation weights must be finite and nonnegative')
    if rank_eval and not args.tag:
        ap.error('use --tag to save the inpainting experiment in a separate directory')
    if rank_eval:
        from inpainting import Spearman

    sl.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cube = sl.load_cube(args.task)
    mask = sl.hole_mask(args.task, cube.shape)
    norm_rows = slice(EDGE_TEST_SPLIT[0].start, EDGE_TEST_SPLIT[0].stop) if args.edge_test_split else slice(None)
    norm = sl.fit_norm(cube[norm_rows], mask[norm_rows])
    tr_ds, va_ds, te_ds = make_datasets(args.task, cube, mask, norm, args.n_train, args.n_val, args.seed,
                                        k=args.k, stride=args.stride, full=args.full,
                                        context_z=args.context_z, dip_guided=args.dip_guided,
                                        depth_focus=args.depth_focus, wide_tunnel=args.wide_tunnel,
                                        wide_augment=args.wide_augment,
                                        wide_clean_depth_fraction=args.wide_clean_depth_fraction,
                                        wide_crossline_fraction=args.wide_crossline_fraction,
                                        edge_test_split=args.edge_test_split,
                                        wide_paired=args.wide_neighbor_weight is not None,
                                        inpaint_paired=args.neighbor_weight is not None)
    tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=False)
    va = DataLoader(va_ds, batch_size=args.bs)
    te = DataLoader(te_ds, batch_size=args.bs)

    if args.joint_ensemble:
        model, joint_members, baseline_norm = sl.load_trainable_inpainting_ensemble(args.joint_ensemble, device)
        if baseline_norm != norm:
            ap.error('ensemble normalization must match the current cube')
    else:
        model = sl.UNet(in_ch=args.k + 1 + args.dip_guided, base=args.base, up_mode=args.up,
                        depth=5 if args.wide_tunnel else 4).to(device)
    if args.init:
        ck = torch.load(args.init, map_location='cpu')
        cfg = sl.model_config(ck)
        if (cfg['k'], cfg['stride'], cfg['base'], cfg['up_mode']) != (
                args.k, args.stride, args.base, args.up):
            ap.error('--init requires matching k, stride, base and upsampling')
        if cfg['dip_guided'] and not args.dip_guided:
            ap.error('a dip-guided checkpoint requires --dip-guided')
        if cfg['wide_tunnel'] != args.wide_tunnel or cfg['joint_members']:
            ap.error('this checkpoint uses a different architecture')
        state = ck['state_dict']
        if args.dip_guided and not cfg['dip_guided']:
            key = 'downs.0.block.0.weight'
            state[key] = torch.nn.functional.pad(state[key], (0, 0, 0, 0, 0, 1))
        model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(tr)
    if args.wide_tunnel:
        from inpainting_wide import WideTunnelLoss
        crit = WideTunnelLoss()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    else:
        crit = sl.HoleLoss(w_corr=args.corr_weight, w_center_corr=args.center_corr_weight)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=AMP)

    out = Path(args.out) / f'{args.task}_unet{args.tag}'
    meta = {'base': args.base, 'k': args.k, 'stride': args.stride, 'up_mode': args.up,
            'norm': [norm.mean, norm.std], 'full': args.full}
    if rank_eval:
        meta.update(context_z=args.context_z, dip_guided=args.dip_guided)
    if args.wide_tunnel:
        meta['wide_tunnel'] = True
    if args.joint_ensemble:
        meta['joint_members'] = joint_members
    select_pearson = args.wide_tunnel or args.neighbor_weight is not None or bool(args.joint_ensemble)
    selection = {'selection_metric': 'val_r'} if select_pearson and not args.full else {}
    if args.edge_test_split:
        selection.update(
            y_split={name: [rows.start, rows.stop]
                     for name, rows in zip(('train', 'val', 'test'), EDGE_TEST_SPLIT)},
            fixed_gap_x_starts=EDGE_TEST_TRACES,
            fixed_gap_z=[179, 282],
            lateral_tta=True,
        )
    out.mkdir(parents=True, exist_ok=True)
    # В --full val пересекается с train: это лишь контроль сходимости, не оценка -> другие ключи/подпись
    vk = 'ctrl' if args.full else 'val'
    hist = {'train_loss': [], 'train_r': [], f'{vk}_loss': [], f'{vk}_r': [], f'{vk}_r_global': []}
    if args.width_curriculum:
        target_width = tr_ds.hh
        widths = curriculum_widths(args.epochs, target_width)
        hist['train_gap_width'] = []
    if args.full:
        print('--full: строки val входят в train, "ctrl r" - НЕ оценка качества '
              '(для отчёта брать test R из прогона без --full)', flush=True)
    best = -float('inf')
    if rank_eval:
        hist[f'{vk}_spearman_global'] = []
    print(f'params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, norm {norm}', flush=True)
    if selection:
        print('Checkpoint selection: val r (mean Pearson over validation gaps)', flush=True)
    neighbor_weight = (args.wide_neighbor_weight if args.wide_tunnel else args.neighbor_weight)
    paired = neighbor_weight is not None
    model.train()
    for ep in range(1, args.epochs + 1):
        width_log, eligible = '', True
        if args.width_curriculum:
            tr_ds.hh = widths[ep - 1]
            hist['train_gap_width'].append(tr_ds.hh)
            width_log = f' width {tr_ds.hh}'
            eligible = tr_ds.hh == target_width
        t0, tl, trs = time.time(), 0.0, []
        for x, y, m in tr:
            x, y, m = x.to(device), y.to(device), m.to(device)
            batch_size = x.size(0)
            if paired:
                paired_targets, pair_mask = y, m
                x, y = x.flatten(0, 1), y.flatten(0, 1)
                m = m[:, None].expand_as(paired_targets).flatten(0, 1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
                p = model(x)
            loss, info = crit(p.float(), y, m)
            if neighbor_weight:
                neighbor = sl.slice_difference_loss(p.float().reshape_as(paired_targets),
                                                    paired_targets, pair_mask)
                loss = loss + neighbor_weight * neighbor
            if args.rank_weight:
                pv, tv = p[m > 0.5], y[m > 0.5]
                a, b = torch.randint(pv.numel(), (2, 8192), device=device)
                sign = (tv[a] - tv[b]).sign()
                pair_loss = torch.nn.functional.softplus(-sign * (pv[a] - pv[b]))
                loss = loss + args.rank_weight * (pair_loss * sign.abs()).sum() / sign.abs().sum().clamp_min(1)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * batch_size
            trs.append(info['r'])
        ranks = Spearman() if rank_eval else None
        vl, vr, vrg = evaluate(model, va, crit, device, ranks, lateral_tta=args.edge_test_split)
        hist['train_loss'].append(tl / len(tr_ds)); hist['train_r'].append(float(np.mean(trs)))
        hist[f'{vk}_loss'].append(vl); hist[f'{vk}_r'].append(vr); hist[f'{vk}_r_global'].append(vrg)
        score, rank_log = vr, ''
        if ranks is not None:
            rho = ranks.result()['spearman_global']
            hist[f'{vk}_spearman_global'].append(rho)
            rank_log = f' rho {rho:.4f}'
            if not select_pearson:
                score = rho
            (out / 'hist.json').write_text(json.dumps({'hist': hist, 'args': vars(args), **selection}, indent=1))
        if not args.full and eligible and score > best:  # В --full выбор по val некорректен -> best.pt не пишем
            best = score
            torch.save({'state_dict': model.state_dict(), 'epoch': ep, **meta, **selection}, out / 'best.pt')
        print(f'ep {ep:3d}{width_log} train {hist["train_loss"][-1]:.4f} r {hist["train_r"][-1]:.4f} | '
              f'{vk} {vl:.4f} r {vr:.4f} rg {vrg:.4f}{rank_log} | {time.time() - t0:.0f}s', flush=True)

    if args.full:  # val пересекается с train -> выбор по val некорректен, берем последнюю эпоху
        torch.save({'state_dict': model.state_dict(), 'epoch': args.epochs, **meta}, out / 'final.pt')
        print(f'saved final.pt (epoch {args.epochs})', flush=True)
        json.dump({'hist': hist, 'args': vars(args),
                   'note': 'full: ctrl_* computed on rows included in train; not a quality estimate'},
                  open(out / 'hist.json', 'w'), indent=1)
        return
    if rank_eval and not args.edge_test_split:
        return  # Оценка модели отдельно с помощью compare.py
    ck = torch.load(out / 'best.pt', map_location=device)
    model.load_state_dict(ck['state_dict'])
    tl_, tr_, trg = evaluate(model, te, crit, device, lateral_tta=args.edge_test_split)
    print(f'TEST (best ep {ck["epoch"]}) loss {tl_:.4f} R {tr_:.4f} R_global {trg:.4f}', flush=True)
    json.dump({'hist': hist, 'test_r': tr_, 'test_r_global': trg, 'best_epoch': ck['epoch'],
               'args': vars(args), **selection},
              open(out / 'hist.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
