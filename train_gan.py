import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import seismic_lib as sl
from train_unet import make_datasets, evaluate


class PatchD(nn.Module):
    """PatchGAN: на вход [вход с пропуском, маска, кандидат], на выход карта логитов."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, c = [], in_ch
        for i, ch in enumerate([base, base * 2, base * 4, base * 8]):
            layers += [sn(nn.Conv2d(c, ch, 4, stride=2 if i < 3 else 1, padding=1)), nn.LeakyReLU(0.2, True)]
            c = ch
        layers += [sn(nn.Conv2d(c, 1, 4, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x, cand):
        return self.net(torch.cat([x, cand], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr_g', type=float, default=1e-4)
    ap.add_argument('--lr_d', type=float, default=2e-4)
    ap.add_argument('--lam_adv', type=float, default=0.01)
    ap.add_argument('--n_train', type=int, default=3000)
    ap.add_argument('--n_val', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--init', default=None, help='веса U-Net (по умолчанию runs/<task>_unet/best.pt)')
    ap.add_argument('--out', default='runs')
    ap.add_argument('--tag', default='', help='суффикс папки прогона')
    args = ap.parse_args()

    sl.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cube = sl.load_cube(args.task)
    mask = sl.hole_mask(args.task, cube.shape)
    norm = sl.fit_norm(cube, mask)

    # Конфиг генератора (k, stride, up_mode, base) берется из чекпоинта U-Net
    init = args.init or f'{args.out}/{args.task}_unet/best.pt'
    ck = torch.load(init, map_location=device)
    cfg = sl.model_config(ck)
    n_sl = cfg['k']
    ctr = slice(n_sl // 2, n_sl // 2 + 1)  # центральный срез стопки во входе x
    G = sl.UNet(in_ch=n_sl + 1, base=cfg['base'], up_mode=cfg['up_mode']).to(device)
    G.load_state_dict(ck['state_dict'])
    D = PatchD(in_ch=n_sl + 2).to(device)  # x (k срезов + маска) + кандидат

    tr_ds, va_ds, te_ds = make_datasets(args.task, cube, mask, norm, args.n_train, args.n_val, args.seed,
                                        k=cfg['k'], stride=cfg['stride'])
    tr = DataLoader(tr_ds, batch_size=args.bs)
    va = DataLoader(va_ds, batch_size=args.bs)
    te = DataLoader(te_ds, batch_size=args.bs)
    crit = sl.HoleLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    out = Path(args.out) / f'{args.task}_gan{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    hist = {k: [] for k in ['train_loss', 'train_r', 'adv_g', 'loss_d', 'val_loss', 'val_r', 'val_r_global']}
    _, r0, r0g = evaluate(G, va, crit, device)
    print(f'init from {init}: val R {r0:.4f} R_global {r0g:.4f}', flush=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        acc = {k: [] for k in ['loss', 'r', 'adv', 'd']}
        for x, y, m in tr:
            x, y, m = x.to(device), y.to(device), m.to(device)
            fake = sl.composite(x[:, ctr], G(x), m)
            real = y
            # D: hinge loss
            opt_d.zero_grad(set_to_none=True)
            ld = F.relu(1 - D(x, real)).mean() + F.relu(1 + D(x, fake.detach())).mean()
            ld.backward()
            opt_d.step()
            # G: реконструкция + adversarial
            opt_g.zero_grad(set_to_none=True)
            pred = G(x)
            rec, info = crit(pred, y, m)
            adv = -D(x, sl.composite(x[:, ctr], pred, m)).mean()
            loss = rec + args.lam_adv * adv
            loss.backward()
            opt_g.step()
            acc['loss'].append(rec.item()); acc['r'].append(info['r'])
            acc['adv'].append(adv.item()); acc['d'].append(ld.item())
        vl, vr, vrg = evaluate(G, va, crit, device)
        for k, a in [('train_loss', 'loss'), ('train_r', 'r'), ('adv_g', 'adv'), ('loss_d', 'd')]:
            hist[k].append(float(np.mean(acc[a])))
        hist['val_loss'].append(vl); hist['val_r'].append(vr); hist['val_r_global'].append(vrg)
        if vr > best:
            best = vr
            torch.save({'state_dict': G.state_dict(), **cfg, 'norm': [norm.mean, norm.std],
                        'epoch': ep, 'init': str(init), 'D': D.state_dict()}, out / 'best.pt')
        print(f'ep {ep:3d} rec {hist["train_loss"][-1]:.4f} r {hist["train_r"][-1]:.4f} '
              f'advG {hist["adv_g"][-1]:.3f} D {hist["loss_d"][-1]:.3f} | val {vl:.4f} r {vr:.4f} rg {vrg:.4f} | '
              f'{time.time() - t0:.0f}s', flush=True)

    ck2 = torch.load(out / 'best.pt', map_location=device)
    G.load_state_dict(ck2['state_dict'])
    _, test_r, test_rg = evaluate(G, te, crit, device)
    print(f'TEST (best ep {ck2["epoch"]}) R {test_r:.4f} R_global {test_rg:.4f}', flush=True)
    json.dump({'hist': hist, 'test_r': test_r, 'test_r_global': test_rg, 'best_epoch': ck2['epoch'], 'args': vars(args)},
              open(out / 'hist.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
