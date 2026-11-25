import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import rankdata


def dip_fill(known, mask):
    """Продлеваем границы с обеих сторон пропуска по наклонам, оцененным на известных трассах"""
    rows, cols = np.where(mask)
    top, bottom = rows.min(), rows.max() + 1
    start, stop = cols.min(), cols.max() + 1
    lateral, depth = np.mgrid[top:bottom, start:stop]
    values, weights = [], []
    for strip, side, distance in (
        (known[max(0, top - 16):top], -1, lateral - top + 1),
        (known[bottom:bottom + 16], 0, bottom - lateral),
    ):
        gx, gz = np.gradient(strip)
        cross = gaussian_filter(gx * gz, (2, 5))[side]
        energy = gaussian_filter(gz * gz, (2, 5))[side]
        slope = np.clip(-cross / (energy + 0.02 * energy.mean() + 1e-6), -2, 2)
        direction = 1 if side == -1 else -1
        source = depth - direction * distance * slope[depth]
        values.append(map_coordinates(strip[side], [source], order=1, mode='nearest'))
        weights.append(((source >= 0) & (source <= known.shape[1] - 1)) / distance)
    # Если боковые проекции выходят за пределы окна, используем вертикальную интерполяцию
    frac = (depth - start + 1) / (stop - start + 1)
    vertical = (1 - frac) * known[lateral, start - 1] + frac * known[lateral, stop]
    weight_z = 0.25 / np.minimum(depth - start + 1, stop - depth)
    out = known.copy()
    out[top:bottom, start:stop] = (
        (values[0] * weights[0] + values[1] * weights[1] + vertical * weight_z)
        / (weights[0] + weights[1] + weight_z)
    )
    return out


class Spearman:
    def __init__(self, diagnostics=False):
        self.predictions = []
        self.targets = []
        self.diagnostics = diagnostics
        self.window_scores = []
        self.center_scores = []

    @staticmethod
    def score(pred, target):
        p, t = rankdata(pred), rankdata(target)
        p -= p.mean()
        t -= t.mean()
        denom = np.sqrt(np.dot(p, p) * np.dot(t, t))
        return float(np.dot(p, t) / denom) if denom else 0.0

    def add(self, pred, target, mask):
        mask = mask.cpu().numpy() > 0.5
        pred, target = pred.cpu().numpy(), target.cpu().numpy()
        self.predictions.append(pred[mask])
        self.targets.append(target[mask])
        if self.diagnostics:
            for p, t, m in zip(pred[:, 0], target[:, 0], mask[:, 0]):
                self.window_scores.append(self.score(p[m], t[m]))
                rows, cols = np.where(m)
                center = np.s_[rows.min() + 25:rows.max() - 24, cols.min() + 25:cols.max() - 24]
                self.center_scores.append(self.score(p[center], t[center]))

    def result(self):
        scores = {'spearman_global': self.score(np.concatenate(self.predictions),
                                               np.concatenate(self.targets))}
        if self.diagnostics:
            scores.update(spearman_mean=float(np.mean(self.window_scores)),
                          spearman_center_mean=float(np.mean(self.center_scores)))
        return scores
