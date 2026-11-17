import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import seismic_lib as sl
from train_unet import make_datasets

REF_BOXES = {
    'inpainting': {'hole': [np.s_[:, 368:581, 179:282]],
                   'ref': [np.s_[:, 100:313, 179:282], np.s_[:, 650:863, 179:282]]},
    'outpainting': {'hole': [np.s_[:, 0:24, :], np.s_[:, 457:481, :]],
                    'ref': [np.s_[:, 24:48, :], np.s_[:, 433:457, :]]},
}


class RAcc:
    """Накопитель R_mean (по патчам) и R_global (по всем вокселам пропусков)."""

    def __init__(self):
        self.rs, self.s = [], np.zeros(6)

    def add(self, p, y, m):
        self.rs.append(sl.pearson_r_torch(p, y, m).cpu())
        sel = m > 0.5
        pv, tv = p[sel].double(), y[sel].double()
        self.s += torch.stack([pv.new_tensor(pv.numel()), pv.sum(), tv.sum(), (pv * pv).sum(),
                               (tv * tv).sum(), (pv * tv).sum()]).cpu().numpy()

    def result(self):
        n, sp, st, spp, stt, spt = self.s
        rg = (spt - sp * st / n) / (np.sqrt((spp - sp * sp / n) * (stt - st * st / n)) + 1e-12)
        return float(torch.cat(self.rs).mean()), float(rg)


@torch.no_grad()
def model_pred(model, x, task, k, tta):
    preds = [model(x)]
    if tta:
        if k > 1:
            preds.append(model(torch.cat([x[:, :k].flip(1), x[:, k:]], 1)))
        if task == 'inpainting':
            preds.append(model(x.flip(2)).flip(2))
    return torch.stack(preds).mean(0)


def baseline_pred(task, y, m):
    out = torch.empty_like(y)
    for b in range(y.shape[0]):
        mm = m[b, 0].numpy() > 0.5
        plane = y[b, 0].numpy() * ~mm
        if task == 'inpainting':
            out[b, 0] = torch.from_numpy(sl.baseline_inpaint_linear(plane, mm))
        else:
            out[b, 0] = torch.from_numpy(sl.baseline_outpaint_nearest(plane, int(mm[:, 0].sum())))
    return out


def baseline_restore(task, cube, mask):
    r = cube.copy()
    if task == 'inpainting':
        m0 = mask[0]
        for i in range(cube.shape[0]):
            r[i] = sl.baseline_inpaint_linear(cube[i], m0)
    else:
        e = int(mask[0, :, 0].sum() // 2)
        n1 = cube.shape[1]
        r[:, n1 - e:, :] = cube[:, n1 - e - 1:n1 - e, :]
        r[:, :e, :] = cube[:, e:e + 1, :]
    return r


def shape_stats(task, restored):
    """
    adjR - средний R между соседними срезами axis0 внутри области: не зависит от амплитуды,
    падает, если срезы заполнены несогласованно (полосы).
    """
    def adj_std(boxes):
        rs, vals = [], []
        for b in boxes:
            p = restored[b].reshape(restored[b].shape[0], -1).astype(np.float64)
            p -= p.mean(1, keepdims=True)
            nrm = np.sqrt((p * p).sum(1)) + 1e-12
            rs.append(((p[1:] * p[:-1]).sum(1) / (nrm[1:] * nrm[:-1])).mean())
            vals.append(restored[b].ravel())
        return float(np.mean(rs)), float(np.concatenate(vals).std())

    ah, sh = adj_std(REF_BOXES[task]['hole'])
    ar, sr = adj_std(REF_BOXES[task]['ref'])
    return ah, ar, sh / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['inpainting', 'outpainting'])
    ap.add_argument('--ckpts', nargs='*', default=None,
                    help='по умолчанию все runs/<task>_*/best.pt и final.pt')
    ap.add_argument('--n_test', type=int, default=400, help='как --n_val при обучении')
    ap.add_argument('--seed', type=int, default=42, help='как --seed при обучении')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--no_cube', action='store_true', help='не считать adjR/std на реальном пропуске')
    ap.add_argument('--out', default=None, help='по умолчанию results/<task>_compare.json')
    args = ap.parse_args()

    sl.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cube = sl.load_cube(args.task)
    mask = sl.hole_mask(args.task, cube.shape)
    norm = sl.fit_norm(cube, mask)
    ckpts = args.ckpts or sorted(glob.glob(f'runs/{args.task}_*/best.pt') + glob.glob(f'runs/{args.task}_*/final.pt'))

    loaders = {}

    def test_loader(k, stride):
        if (k, stride) not in loaders:
            te = make_datasets(args.task, cube, mask, norm, 1, args.n_test, args.seed, k=k, stride=stride)[2]
            loaders[(k, stride)] = DataLoader(te, batch_size=args.bs)
        return loaders[(k, stride)]

    # Одни и те же test-патчи для разных k: центральный срез и маска совпадают
    _, y1, m1 = next(iter(test_loader(1, 1)))
    _, y5, m5 = next(iter(test_loader(5, 2)))
    assert torch.equal(y1, y5) and torch.equal(m1, m5), 'test-патчи различаются для разных k'

    rows = []
    # --- бейзлайн
    acc = RAcc()
    for _, y, m in test_loader(1, 1):
        acc.add(baseline_pred(args.task, y, m), y, m)
    rm, rg = acc.result()
    row = {'name': 'baseline_' + ('linear' if args.task == 'inpainting' else 'nearest'),
           'k': '', 'up': '', 'r_mean': rm, 'r_global': rg, 'r_mean_tta': rm, 'r_global_tta': rg}
    if not args.no_cube:
        row['adj_r'], row['adj_r_ref'], row['std'] = shape_stats(args.task, baseline_restore(args.task, cube, mask))
    rows.append(row)
    print(row, flush=True)

    # --- модели
    for path in ckpts:
        ck = torch.load(path, map_location='cpu')
        model, cfg, _ = sl.load_model(path, device)
        name = str(Path(path).parent.name) + ('/final' if Path(path).stem == 'final' else '')
        row = {'name': name, 'k': cfg['k'], 'up': cfg['up_mode'], 'path': path,
               'epoch': ck.get('epoch'), 'full': bool(ck.get('full', False))}
        if not row['full']:
            for tta in (False, True):
                acc = RAcc()
                for x, y, m in test_loader(cfg['k'], cfg['stride']):
                    x, y, m = x.to(device), y.to(device), m.to(device)
                    acc.add(model_pred(model, x, args.task, cfg['k'], tta), y, m)
                rm, rg = acc.result()
                sfx = '_tta' if tta else ''
                row['r_mean' + sfx], row['r_global' + sfx] = rm, rg
        if not args.no_cube:
            restored = sl.restore_cube(args.task, cube, model, cfg, norm, device=device, tta=True)
            row['adj_r'], row['adj_r_ref'], row['std'] = shape_stats(args.task, restored)
            del restored
        rows.append(row)
        print(row, flush=True)
        del model
        torch.cuda.empty_cache()

    # Таблица
    def f(v):
        return f'{v:.4f}' if isinstance(v, float) else '—'

    print(f'\n### {args.task}: test-блок ({args.n_test} синтетических пропусков) + форма заливки реального пропуска\n')
    print('| модель | k | up | R_mean | R_global | R_mean TTA | R_global TTA | adjR пропуск | adjR данные | std (≈1) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['name']} | {r['k']} | {r['up']} | {f(r.get('r_mean'))} | {f(r.get('r_global'))} | "
              f"{f(r.get('r_mean_tta'))} | {f(r.get('r_global_tta'))} | {f(r.get('adj_r'))} | {f(r.get('adj_r_ref'))} | {f(r.get('std'))} |")

    out = Path(args.out or f'results/{args.task}_compare.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'task': args.task, 'n_test': args.n_test, 'seed': args.seed, 'rows': rows},
              open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nsaved', out)


if __name__ == '__main__':
    main()
