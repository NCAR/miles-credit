"""Compare two sdl_spread_check.py .npz outputs (untrained vs fine-tuned): SSR and noise/err per variable."""

import sys

import numpy as np

REF_HPA = (200, 500, 850)


def load(path):
    d = np.load(path)
    n, M = int(d["n_init"]), int(d["members"])
    out = {}
    for key in d.files:
        if "|" not in key:
            continue
        var, stat = key.split("|")
        out.setdefault(var, {})[stat] = d[key]
    return out, n, M, d["p_ref"]


def summary(s, n, M, idx=None):
    g = {k: (v if idx is None else v[idx : idx + 1]).mean() / n for k, v in s.items()}
    det, ensm, spr = (float(np.sqrt(g[k])) for k in ("det_se", "ensmean_se", "spread_var"))
    ssr = spr / ensm * np.sqrt((M + 1) / M)
    noise = float(np.sqrt(g["member_det_diff_sq"])) / det
    return det, ensm, ssr, noise


a, na, Ma, p_ref = load(sys.argv[1])
b, nb, Mb, _ = load(sys.argv[2])
lev = {h: int(np.abs(p_ref - h).argmin()) for h in REF_HPA}
print(f"{'variable':24s} {'det RMSE':>9s} {'ensm/det':>17s} {'SSR':>13s} {'noise/err':>13s}")
print(f"{'':24s} {'':9s} {'before':>8s} {'after':>8s} {'before':>6s} {'after':>6s} {'before':>6s} {'after':>6s}")
for var in a:
    short = "/".join(var.split("/")[-2:])
    rows = [(f"{short}", None)]
    if a[var]["det_se"].size > 1:
        rows = [(f"{short} (lev mean)", None)] + [(f"{short} {h} hPa", i) for h, i in lev.items()]
    for name, idx in rows:
        da, ea, sa, xa = summary(a[var], na, Ma, idx)
        db, eb, sb, xb = summary(b[var], nb, Mb, idx)
        print(f"{name:24s} {da:9.4g} {ea / da:8.3f} {eb / db:8.3f} {sa:6.2f} {sb:6.2f} {xa:6.2f} {xb:6.2f}")
