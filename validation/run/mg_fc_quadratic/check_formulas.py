# Check the quadratic level-jump ghost formulas (ProlongateFCMG / FillCoarseMG):
# (1) fine ghost and coarse ghost errors are O(h^3) for smooth phi (exact for quadratics),
# (2) flux matching g_c - c1 = 2 mean_4(f1 - g_f) holds to round-off with random data
#     (incl. one-sided transverse stencils at a coarse block edge).
import numpy as np
rng = np.random.default_rng(1)

def coarse_at_ghosts(w, ca, cb):
    ha, hb = ca - 0.5, cb - 0.5
    da = np.array([-1 + ha, 1 - 2*ha, ha]); db = np.array([-1 + hb, 1 - 2*hb, hb])
    c = w[ca, cb]
    sa = da @ w[:, cb]; sb = db @ w[ca, :]; mx = da @ w @ db
    d2 = (w[0, cb] - 2*w[1, cb] + w[2, cb]) + (w[ca, 0] - 2*w[ca, 1] + w[ca, 2])
    out = np.zeros((2, 2))
    for ib, xb in enumerate((-0.25, 0.25)):
        for ia, xa in enumerate((-0.25, 0.25)):
            out[ib, ia] = c + d2/32 + xa*sa + xb*sb + xa*xb*mx
    return out, d2

def phi(x, y, z):
    return np.sin(1.3*x + 0.7) * np.cos(0.9*y - 0.4) * np.exp(0.5*z) + 0.3*x*y*z

# geometry: face at x = 0, coarse side x < 0 (cells 2h), fine side x > 0 (cells h).
# coarse face cell centred at (Y,Z) = (0,0); its 3x3 patch offset by (ca,cb).
for ca, cb in [(1, 1), (0, 1), (2, 0), (0, 2)]:
    print("stencil centre (ca,cb) =", (ca, cb))
    for h in [0.1, 0.05, 0.025, 0.0125]:
        H = 2*h
        w = np.array([[phi(-h, (ia - ca)*H, (ib - cb)*H) for ib in range(3)] for ia in range(3)])
        ct, d2 = coarse_at_ghosts(w, ca, cb)
        errf = 0.0; s = 0.0
        for ib, zb in enumerate((-h/2, h/2)):
            for ia, ya in enumerate((-h/2, h/2)):
                f1, f2 = phi(h/2, ya, zb), phi(3*h/2, ya, zb)
                g = 8/15*ct[ib, ia] + 2/3*f1 - 0.2*f2
                errf = max(errf, abs(g - phi(-h/2, ya, zb)))
        A1 = np.mean([phi(h/2, ya, zb) for ya in (-h/2, h/2) for zb in (-h/2, h/2)])
        A2 = np.mean([phi(3*h/2, ya, zb) for ya in (-h/2, h/2) for zb in (-h/2, h/2)])
        gc = -w[ca, cb]/15 + (2/3*A1 + 0.4*A2) - d2/30
        errc = abs(gc - phi(h, 0, 0))
        print(f"  h={h:7.4f} fine ghost err {errf:.3e}  coarse ghost err {errc:.3e}")

# conservation with random data
worst = 0.0
for trial in range(1000):
    ca, cb = rng.integers(0, 3, 2)
    w = rng.normal(size=(3, 3)); f1 = rng.normal(size=(2, 2)); f2 = rng.normal(size=(2, 2))
    ct, d2 = coarse_at_ghosts(w, ca, cb)
    gf = 8/15*ct + 2/3*f1 - 0.2*f2
    gc = -w[ca, cb]/15 + (2/3*f1.mean() + 0.4*f2.mean()) - d2/30
    worst = max(worst, abs((gc - w[ca, cb]) - 2*np.mean(f1 - gf)))
print("max |(g_c - c1) - 2 mean(f1 - g_f)| over 1000 random patches:", worst)
