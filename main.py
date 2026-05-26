from pathlib import Path
import argparse
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl


plt.rcParams["axes.unicode_minus"] = False
pi = math.pi


def ue(x, y):
    return np.cos(pi * x) * np.cos(pi * y)


def ux(x, y):
    return -pi * np.sin(pi * x) * np.cos(pi * y)


def uy(x, y):
    return -pi * np.cos(pi * x) * np.sin(pi * y)


def ff(x, y, e, kind):
    u = ue(x, y)
    if kind == "made":
        return 2.0 * pi * pi * u + e * u ** 3
    return 2.0 * pi * pi * u


def msh(n):
    p = []
    for j in range(n + 1):
        for i in range(n + 1):
            p.append((i / n, j / n))
    t = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + 1
            c = a + n + 1
            d = c + 1
            t.append((a, b, c))
            t.append((b, d, c))
    return np.array(p, float), np.array(t, int)


def _geo(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    ar = abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])) / 2.0
    bx = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2.0 * ar)
    by = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2.0 * ar)
    return ar, bx, by


def asm(n):
    p, t = msh(n)
    n0 = len(p)
    rr = []
    cc = []
    kd = []
    md = []
    mv = np.zeros(n0)
    for tri in t:
        xy = p[tri]
        ar, bx, by = _geo(xy)
        ke = ar * (np.outer(bx, bx) + np.outer(by, by))
        me = ar * (np.ones((3, 3)) + np.eye(3)) / 12.0
        mv[tri] += ar / 3.0
        for i in range(3):
            for j in range(3):
                rr.append(tri[i])
                cc.append(tri[j])
                kd.append(ke[i, j])
                md.append(me[i, j])
    k = sp.coo_matrix((kd, (rr, cc)), shape=(n0, n0)).tocsr()
    m = sp.coo_matrix((md, (rr, cc)), shape=(n0, n0)).tocsr()
    return p, t, k, m, mv


qa = np.array([
    [1 / 3, 1 / 3, 1 / 3],
    [0.05971587178977, 0.47014206410511, 0.47014206410511],
    [0.47014206410511, 0.05971587178977, 0.47014206410511],
    [0.47014206410511, 0.47014206410511, 0.05971587178977],
    [0.79742698535309, 0.10128650732346, 0.10128650732346],
    [0.10128650732346, 0.79742698535309, 0.10128650732346],
    [0.10128650732346, 0.10128650732346, 0.79742698535309],
])

qw = np.array([
    0.225,
    0.13239415278851,
    0.13239415278851,
    0.13239415278851,
    0.12593918054483,
    0.12593918054483,
    0.12593918054483,
])


def rhs(p, t, e, kind):
    b = np.zeros(len(p))
    for tri in t:
        xy = p[tri]
        ar, _, _ = _geo(xy)
        for la, w in zip(qa, qw):
            x, y = la @ xy
            b[tri] += ar * w * ff(x, y, e, kind) * la
    return b


def rj(p, t, k, u, e, b):
    r = k @ u - b
    rr = []
    cc = []
    dd = []
    for tri in t:
        xy = p[tri]
        ar, _, _ = _geo(xy)
        uv = u[tri]
        for la, w in zip(qa, qw):
            uh = la @ uv
            r[tri] += e * ar * w * uh ** 3 * la
            je = 3.0 * e * ar * w * uh * uh * np.outer(la, la)
            for i in range(3):
                for j in range(3):
                    rr.append(tri[i])
                    cc.append(tri[j])
                    dd.append(je[i, j])
    jn = sp.coo_matrix((dd, (rr, cc)), shape=k.shape).tocsr()
    return r, k + jn


def bsol(lo, up, d, r):
    y = sl.spsolve_triangular(lo, r, lower=True)
    z = sl.spsolve_triangular(up, d * y, lower=False)
    return z


def atm(a, b, tol=1e-10, imax=5000, om=1.0):
    a = a.tocsr()
    d = a.diagonal()
    lo = sp.diags(d) + om * sp.tril(a, -1)
    up = sp.diags(d) + om * sp.triu(a, 1)
    x = np.zeros_like(b, dtype=float)
    nb = np.linalg.norm(b)
    if nb == 0:
        return x, {"it": 0, "res": 0.0}
    r = b - a @ x
    z = bsol(lo, up, d, r)
    p = z.copy()
    rz = float(r @ z)
    res = np.linalg.norm(r)
    for it in range(1, imax + 1):
        if res <= tol * nb:
            break
        ap = a @ p
        q = float(p @ ap)
        if q <= 0 or not np.isfinite(q):
            break
        al = rz / q
        x += al * p
        r -= al * ap
        res = np.linalg.norm(r)
        if res <= tol * nb:
            break
        z = bsol(lo, up, d, r)
        rz2 = float(r @ z)
        if rz == 0:
            break
        be = rz2 / rz
        p = z + be * p
        rz = rz2
    r = b - a @ x
    res = np.linalg.norm(r)
    return x, {"it": it, "res": res}


def mean0(u, mv):
    return u - (mv @ u) / mv.sum()


def first(p, mv):
    u = ue(p[:, 0], p[:, 1])
    return mean0(u, mv)


def sol(n, e, kind="made", u0=None, ntol=1e-9):
    p, t, k, m, mv = asm(n)
    b = rhs(p, t, e, kind)
    if u0 is None or len(u0) != len(p):
        u = first(p, mv)
    else:
        u = mean0(u0.copy(), mv)
    hist = []
    nr = None
    for ni in range(1, 24):
        r, j = rj(p, t, k, u, e, b)
        nr = np.linalg.norm(r) / math.sqrt(len(u))
        if nr < ntol:
            break
        du, st = atm(j, -r, tol=2e-9, imax=9000)
        a = 1.0
        ok = False
        for _ in range(12):
            v = mean0(u + a * du, mv)
            rr, _ = rj(p, t, k, v, e, b)
            nn = np.linalg.norm(rr) / math.sqrt(len(u))
            if nn < nr:
                u = v
                nr = nn
                ok = True
                break
            a *= 0.5
        if not ok:
            u = mean0(u + a * du, mv)
        hist.append({"newton": ni, "lin_it": st["it"], "res": nr, "step": a})
        if nr < ntol:
            break
    return {"p": p, "t": t, "k": k, "m": m, "mv": mv, "u": u, "hist": hist, "res": nr, "n": n, "eps": e, "kind": kind}


def err(s):
    p = s["p"]
    t = s["t"]
    u = s["u"]
    l2 = 0.0
    h1 = 0.0
    for tri in t:
        xy = p[tri]
        ar, bx, by = _geo(xy)
        uv = u[tri]
        gx = float(bx @ uv)
        gy = float(by @ uv)
        for la, w in zip(qa, qw):
            x, y = la @ xy
            uh = float(la @ uv)
            l2 += ar * w * (uh - ue(x, y)) ** 2
            h1 += ar * w * ((gx - ux(x, y)) ** 2 + (gy - uy(x, y)) ** 2)
    inf = float(np.max(np.abs(u - ue(p[:, 0], p[:, 1]))))
    return {"l2": math.sqrt(l2), "h1": math.sqrt(h1), "inf": inf}


def dif(s, v):
    w = s["u"] - v
    l2 = math.sqrt(max(float(w @ (s["m"] @ w)), 0.0))
    h1 = math.sqrt(max(float(w @ (s["k"] @ w)), 0.0))
    return {"dl2": l2, "dh1": h1}


def wr(path, rows, head):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fig_sol(s, path):
    p = s["p"]
    t = s["t"]
    u = s["u"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=170)
    q = ax.tricontourf(p[:, 0], p[:, 1], t, u, levels=36, cmap="viridis")
    ax.tricontour(p[:, 0], p[:, 1], t, u, levels=12, colors="white", linewidths=0.35, alpha=0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$u_h$, $n=%d$, $\varepsilon=%.0e$" % (s["n"], s["eps"]))
    fig.colorbar(q, ax=ax, shrink=0.84)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_mesh(path, n=6):
    p, t = msh(n)
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=170)
    ax.triplot(p[:, 0], p[:, 1], t, color="#2b4c7e", lw=0.75)
    ax.plot(p[:, 0], p[:, 1], "o", ms=2.4, color="#c65f2b")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"triangulation, $n=%d$" % n)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_exact(path):
    x = np.linspace(0.0, 1.0, 160)
    y = np.linspace(0.0, 1.0, 160)
    xx, yy = np.meshgrid(x, y)
    zz = ue(xx, yy)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=170)
    q = ax.contourf(xx, yy, zz, levels=40, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.contour(xx, yy, zz, levels=12, colors="white", linewidths=0.35, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$u_*(x,y)=\cos(\pi x)\cos(\pi y)$")
    fig.colorbar(q, ax=ax, shrink=0.84)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_err(s, path):
    p = s["p"]
    t = s["t"]
    z = np.abs(s["u"] - ue(p[:, 0], p[:, 1]))
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=170)
    q = ax.tricontourf(p[:, 0], p[:, 1], t, z, levels=36, cmap="magma")
    ax.tricontour(p[:, 0], p[:, 1], t, z, levels=10, colors="white", linewidths=0.3, alpha=0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$|u_h-u_*|$, $n=%d$" % s["n"])
    fig.colorbar(q, ax=ax, shrink=0.84)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_eps_grid(sols, path):
    take = []
    for e in [1e-1, 1e-2, 1e-3, 1e-4]:
        best = min(sols, key=lambda s: abs(s["eps"] - e))
        take.append(best)
    v0 = min(float(s["u"].min()) for s in take)
    v1 = max(float(s["u"].max()) for s in take)
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 7.1), dpi=170, constrained_layout=True)
    q = None
    for ax, s in zip(axs.ravel(), take):
        p = s["p"]
        t = s["t"]
        q = ax.tricontourf(p[:, 0], p[:, 1], t, s["u"], levels=34, cmap="viridis", vmin=v0, vmax=v1)
        ax.tricontour(p[:, 0], p[:, 1], t, s["u"], levels=9, colors="white", linewidths=0.25, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"$\varepsilon=%.0e$" % s["eps"])
    fig.colorbar(q, ax=axs.ravel().tolist(), shrink=0.82)
    fig.savefig(path)
    plt.close(fig)


def fig_conv(rows, path):
    h = np.array([r["h"] for r in rows], float)
    l2 = np.array([r["l2"] for r in rows], float)
    h1 = np.array([r["h1"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=170)
    ax.loglog(h, l2, "o-", label=r"$L_2$")
    ax.loglog(h, h1, "s-", label=r"$H^1$")
    ax.invert_xaxis()
    ax.grid(True, which="both", ls=":", lw=0.6)
    ax.set_xlabel("h")
    ax.set_ylabel("error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_eps(rows, path):
    use = [r for r in rows if r["dl2"] > 0]
    e = np.array([r["eps"] for r in use], float)
    l2 = np.array([r["dl2"] for r in use], float)
    it = np.array([r["lin_sum"] for r in use], float)
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=170)
    ax.loglog(e, l2, "o-", color="#1b6ca8", label=r"$\|u_\varepsilon-u_{10^{-4}}\|_{L_2}$")
    ax.grid(True, which="both", ls=":", lw=0.6)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("L2 error")
    bx = ax.twinx()
    bx.semilogx(e, it, "s--", color="#b44d12", label="ATM steps")
    bx.set_ylabel("ATM steps")
    a, b = ax.get_legend_handles_labels()
    c, d = bx.get_legend_handles_labels()
    ax.legend(a + c, b + d, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figd = out / "figs"
    datd = out / "data"
    figd.mkdir(exist_ok=True)
    datd.mkdir(exist_ok=True)
    fig_mesh(figd / "mesh.png")
    fig_exact(figd / "exact.png")
    conv = []
    for n in [6, 8, 12, 16, 22, 28]:
        s = sol(n, 1e-2, "made", ntol=2e-7)
        e = err(s)
        li = sum(x["lin_it"] for x in s["hist"])
        conv.append({"n": n, "h": 1 / n, "l2": e["l2"], "h1": e["h1"], "inf": e["inf"], "newton": len(s["hist"]), "lin_sum": li, "res": s["res"]})
    wr(datd / "convergence.csv", conv, ["n", "h", "l2", "h1", "inf", "newton", "lin_sum", "res"])
    fig_conv(conv, figd / "convergence.png")
    s0 = sol(28, 1e-2, "made", ntol=2e-7)
    fig_sol(s0, figd / "solution.png")
    fig_err(s0, figd / "error.png")
    eps_rows = []
    eps_sol = []
    ulast = None
    for e in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        s = sol(28, e, "fixed", u0=ulast, ntol=1e-8)
        ulast = s["u"]
        er = err(s)
        row = {"eps": e, "l2": er["l2"], "h1": er["h1"], "inf": er["inf"], "newton": len(s["hist"]), "lin_sum": sum(x["lin_it"] for x in s["hist"]), "res": s["res"]}
        eps_rows.append(row)
        eps_sol.append(s)
    ref = eps_sol[-1]["u"]
    for row, s in zip(eps_rows, eps_sol):
        row.update(dif(s, ref))
    wr(datd / "eps.csv", eps_rows, ["eps", "l2", "h1", "inf", "dl2", "dh1", "newton", "lin_sum", "res"])
    fig_eps(eps_rows, figd / "eps.png")
    fig_eps_grid(eps_sol, figd / "eps_solutions.png")
    l2s = np.array([r["l2"] for r in conv])
    hs = np.array([r["h"] for r in conv])
    p_l2 = float(np.polyfit(np.log(hs), np.log(l2s), 1)[0])
    h1s = np.array([r["h1"] for r in conv])
    p_h1 = float(np.polyfit(np.log(hs), np.log(h1s), 1)[0])
    use = [r for r in eps_rows if r["dl2"] > 0]
    es = np.array([r["eps"] for r in use])
    ee = np.array([r["dl2"] for r in use])
    p_eps = float(np.polyfit(np.log(es), np.log(ee), 1)[0])
    meta = {"p_l2": p_l2, "p_h1": p_h1, "p_eps": p_eps, "n_eps": 28}
    (datd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    run(a.out)


if __name__ == "__main__":
    cli()
