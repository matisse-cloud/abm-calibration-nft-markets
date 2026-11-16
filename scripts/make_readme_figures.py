"""English summary figure for the README, drawn from the tables printed in
notebooks/02 (outputs of the original run; no re-estimation)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abm.plotting import style_ax, new_fig, WHITE, LGRAY, TEAL, CORAL, BLUE, AMBER  # noqa: E402

# Transition matrix (%), rows = period-1 label, cols = period-2 label (Section 6)
labels = ["mean-reverting", "neutral", "trend-following"]
tm = np.array([[24.40, 43.66, 31.94],
               [29.21, 44.73, 26.06],
               [23.89, 46.76, 29.35]])
# Persistence of top / bottom quintile (Section 9), N = 718 wallets, 143 per quintile
T = np.array([1, 5, 10, 20, 30, 50])
top = np.array([23.08, 20.28, 25.17, 23.78, 23.08, 18.88])
bot = np.array([20.98, 21.68, 18.88, 21.68, 20.28, 20.98])
se = 100 * np.sqrt(0.2 * 0.8 / 143)
# Out-of-sample R² of aggregate flows (Section 10)
Tp = np.array([1, 5, 10, 20, 30])
r2_ols = np.array([-0.078, -0.233, -0.915, -0.386, -0.174])
r2_ridge = np.array([-0.061, -0.130, -0.616, -0.367, -0.166])

fig, ax = new_fig(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.05, 1, 1]})
im = ax[0].imshow(tm, cmap="viridis", vmin=15, vmax=50)
ax[0].set_xticks(range(3), ["MR", "neutral", "TF"]); ax[0].set_yticks(range(3), ["MR", "neutral", "TF"])
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f"{tm[i, j]:.0f}%", ha="center", va="center", color="white", fontweight="bold")
ax[0].set_xlabel("label in period 2"); ax[0].set_ylabel("label in period 1")
style_ax(ax[0], "1. Labels are not persistent\n(rows ≈ identical → independence)"); ax[0].grid(False)

ax[1].axhspan(20 - 2 * se, 20 + 2 * se, color=LGRAY, alpha=0.2, label="chance ± 2 s.e.")
ax[1].axhline(20, color=LGRAY, ls="--", lw=1)
ax[1].plot(T, top, "o-", color=TEAL, lw=2, label="top quintile stays top")
ax[1].plot(T, bot, "s-", color=CORAL, lw=2, label="bottom quintile stays bottom")
ax[1].set_xlabel("horizon T (days)"); ax[1].set_ylabel("% of wallets"); ax[1].set_ylim(10, 30)
ax[1].legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)
style_ax(ax[1], "2. Past performance does not persist\n(718 wallets, 50/50 time split)")

w = 1.6
ax[2].bar(Tp - w / 2, r2_ols, width=w, color=BLUE, label="OLS")
ax[2].bar(Tp + w / 2, r2_ridge, width=w, color=AMBER, label="ridge")
ax[2].axhline(0, color=LGRAY, lw=1)
ax[2].set_xlabel("forecast horizon T (days)"); ax[2].set_ylabel("out-of-sample R²")
ax[2].legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)
style_ax(ax[2], "3. Trader flows do not predict returns\n(out-of-sample R² < 0 everywhere)")

fig.suptitle("NFT wallets: in-sample patterns that do not survive out of sample",
             color=WHITE, fontsize=13, fontweight="bold", y=1.04)
plt.tight_layout()
fig.savefig(ROOT / "figures" / "readme_nft_out_of_sample.png", dpi=110, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("ok")
