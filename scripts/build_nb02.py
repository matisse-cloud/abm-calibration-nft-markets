"""Build notebooks/02_nft_trader_behaviour.ipynb from the original Day-3 notebook,
keeping its saved outputs."""
from nbtools import ROOT, load, code_from, md_from, md, save_png, write

J3 = load("J3_Trader_Behaviour1")
FIG = ROOT / "figures"
cells = []
add = cells.append

add(md(r"""
# 02 · Who trades NFTs, and does their behaviour carry information?

**Data.** 2,382,584 OpenSea-style NFT sales (May 2021 → April 2025): buyer and seller wallets (anonymised), token, collection, USD price and timestamp. The dataset was provided for the course and is not redistributed (see `data/README.md`). **This notebook is therefore shown with the outputs of the original run**; code comments and printed labels were translated to English, while figure labels are still in French as rendered at the time.

**Questions**
1. How concentrated is activity across NFTs and across wallets (power law or log-normal)?
2. Can wallets be classified as *trend-followers*, *mean-reverters* or *neutral* from how their net inventory reacts to past returns?
3. Are these labels stable over time, and at which horizon do traders react?
4. Does anyone have skill: do directional trades earn money, does performance persist, and do aggregate flows predict market returns out of sample?

**Methods.** Polars (lazy group-bys on 2.4 M rows) · MLE power-law fits with likelihood-ratio tests (Clauset, Shalizi & Newman 2009, `powerlaw`) · per-wallet Pearson tests · transition matrices · directional PnL and Sharpe ratios · 50/50 out-of-sample split · linear and ridge regression.

**Results in one paragraph.** Activity is extremely concentrated (median wallet: 2 purchases; maximum: 32,602), with heavy tails that are clearly not exponential but cannot be told apart from a log-normal. About half of the 9,649 active wallets show a significant correlation between inventory changes and the previous day's return, split almost evenly between trend-following and mean-reverting. **None of the skill measures survives an out-of-sample test**: labels assigned in the first half of the sample are statistically independent of the labels in the second half, top-quintile traders stay top-quintile at the chance rate (19–25% vs 20%), and aggregate trader flows give a negative out-of-sample R² at every horizon. The limitations section explains why part of the in-sample signal may be mechanical.
"""))

add(md("## 0 · Setup and data loading"))
add(code_from(J3, 2))
add(code_from(J3, 3, replace=[
    ('pl.read_csv("sales.csv")', 'pl.read_csv("../data/sales.csv")'),
    ('df.write_parquet("sales.parquet")', 'df.write_parquet("../data/sales.parquet")'),
    ('pl.read_parquet("sales.parquet")', 'pl.read_parquet("../data/sales.parquet")'),
]))

# ---------------- Section 1
add(md_from(J3, 4, replace=[("> **Note:** In this analysis, the results are inconclusive. The CCDF shows curvature, suggesting that the transaction distribution by NFT does not clearly follow a pure power law.", "")]))
add(code_from(J3, 5))
add(md(r"""
> **Caveat.** `nft_identifier` is a token number *within* a collection, so grouping on it alone merges, for example, token #689 of every collection. The figures below therefore describe activity per token number rather than per individual NFT; grouping on `(nft_collection, nft_identifier)` is the correct key (used in Section 3).
"""))
add(code_from(J3, 6))
add(code_from(J3, 7))
add(md(r"""
**Reading.** α ≈ 2.92 above $x_{\min} = 406$, but the likelihood-ratio test against a log-normal is flat (R = −0.009, p = 0.94): the data cannot discriminate between a preferential-attachment story (power law) and the compounding of independent multiplicative shocks (log-normal). Only the top of the distribution (tokens with more than 406 trades) enters the fit, so the conclusion concerns the most active tokens only.
"""))
add(code_from(J3, 9))

# ---------------- Section 2
add(md_from(J3, 10))
add(code_from(J3, 11))
add(md(r"""
**The distribution is extremely skewed.** Median 2 purchases, mean 7.8, standard deviation 110, maximum 32,602. Three quarters of the 306,253 buying wallets made 4 purchases or fewer; volume comes from a small group of very active wallets (bots, market makers, possibly wash traders).
"""))
add(code_from(J3, 13))
add(code_from(J3, 14))
save_png(cells[-1], FIG / "02_purchases_per_trader.png")
add(md(r"""
| Indicator | Purchases | Reading |
|---|---|---|
| α (MLE) | ≈ 2.20, σ(α) ≈ 0.010 | heavy tail in the usual range for activity data |
| $x_{\min}$ | 22 | the power-law regime only covers wallets with 22+ purchases |
| LR vs exponential | R ≫ 0, p ≈ 0 | exponential clearly rejected |
| LR vs log-normal | R < 0, p > 0.05 | not significant: log-normal is as good |
"""))
add(code_from(J3, 17))
add(md(r"""
Sales look the same: α ≈ 2.20, exponential rejected, power law vs log-normal undecided (p ≈ 0.84). The tail starts earlier ($x_{\min}$ = 10 vs 22), i.e. there are relatively more very active sellers than very active buyers.
"""))

# ---------------- Section 3
add(md_from(J3, 19))
add(code_from(J3, 20))
add(code_from(J3, 21))
add(md(r"""
**Reading.** This is the distribution of the **number of trades per day** for Meebit #689 (1,491 trades in total). Both likelihood-ratio tests reject the power law decisively (R ≈ −40, p < 0.001) in favour of the log-normal and the exponential: the daily activity of the most traded token has a thin tail, with bursts bounded at about 15–20 trades a day rather than occasional extreme spikes.
"""))
add(code_from(J3, 23))
add(code_from(J3, 25))
add(md(r"""
Within the Meebits collection, buy and sell counts per wallet are almost symmetric (α ≈ 2.08 and 2.01). The empirical CCDFs bend below the fitted power law in the far tail: fewer extreme wallets than a pure power law would predict.
"""))

# ---------------- Section 4
add(md_from(J3, 27))
add(code_from(J3, 28))
add(code_from(J3, 29))
add(md(r"""
**Data-quality note.** The daily and hourly maxima are both exactly **92,873**, which is the number of rows whose timestamp fails to parse (`strict=False` returns null). Those rows end up in a single null bucket; they are dropped by the inner joins later on, but they distort these summary counts (the 1,462 "active days" include that bucket). Notebook 03 drops them explicitly.

The median gap between consecutive sales is 0 minutes, so the market is effectively continuous; **daily** aggregation is kept as the working frequency.
"""))

# ---------------- Section 5
add(md_from(J3, 31))
add(code_from(J3, 32))
add(code_from(J3, 33))
add(code_from(J3, 34))
add(code_from(J3, 35))
save_png(cells[-1], FIG / "02_trader_classification.png")
add(md(r"""
*Note: the saved figure above was produced by an earlier version of this cell (two panels, with reference lines at r = ±0.1); the counts and shares it shows are identical to the table printed by the current code.*
"""))
add(md(r"""
### Results: classification

Among the **9,649 wallets** with at least 30 active days:

| Type | Count | Share | Mean correlation |
|---|---|---|---|
| Neutral (p ≥ 0.05) | 5,044 | 52% | 0.00 |
| Trend-following | 2,354 | 24% | +0.19 |
| Mean-reverting | 2,251 | 23% | −0.19 |

Under the null of no relation, 5% of wallets would be flagged by chance (≈ 480), so 48% significant is far more than noise alone. The near-perfect symmetry between the two directional groups, together with Section 6, suggests caution before reading this as a population of genuine strategies (see *Limitations*).
"""))

# ---------------- Section 6
add(md_from(J3, 37))
add(code_from(J3, 38))
add(code_from(J3, 39))
add(code_from(J3, 40))
save_png(cells[-1], FIG / "02_transition_matrix.png")
add(md(r"""
### Results: stability

For the 2,287 wallets classified in both halves of the sample, **the three rows of the transition matrix are almost identical**: whatever the label in period 1, the label in period 2 is roughly 24–29% mean-reverting, 44–47% neutral and 26–32% trend-following. A wallet labelled mean-reverting in period 1 is even slightly *less* likely to be mean-reverting in period 2 (24.4%) than a wallet that was neutral (29.2%).

In other words, the period-2 label is statistically independent of the period-1 label. The directional labels are not persistent traits of wallets. That is compatible with agents switching strategies (Brock–Hommes-type models), but also with labels that mostly reflect noise.
"""))

# ---------------- Section 7
add(md_from(J3, 42))
add(code_from(J3, 43))
add(code_from(J3, 44))
add(code_from(J3, 45))
add(md(r"""
### Results: reaction horizons

T* is spread over the whole 1–50-day range with no dominant horizon, and the best correlations remain small (|r| < 0.2). Picking, for each wallet, the best of 50 correlated horizons is itself a multiple-testing procedure, so this histogram should be read as "no characteristic horizon" rather than as evidence of multi-scale strategies.
"""))

# ---------------- Section 8
add(md_from(J3, 48))
add(code_from(J3, 49))
add(code_from(J3, 50))
add(code_from(J3, 51))
add(md_from(J3, 52))

# ---------------- Section 9
add(md_from(J3, 53))
add(code_from(J3, 54))
add(code_from(J3, 55))
save_png(cells[-1], FIG / "02_performance_persistence.png")
add(md(r"""
### Results: persistence

With 718 wallets present in both halves:
- the correlation between period-1 and period-2 gains ranges from 0.00 to 0.08 across horizons (standard error ≈ 0.037), so at most marginally positive at T = 10–20 days, and not robust across horizons;
- the share of top-quintile wallets that stay top-quintile is **19–25% against 20% for pure chance** (binomial standard error ≈ 3.3 points), and similarly for the bottom quintile.

**Past directional performance does not identify future winners.**
"""))

# ---------------- Section 10
add(md_from(J3, 57))
add(code_from(J3, 58))
add(code_from(J3, 59))
add(code_from(J3, 60))
add(code_from(J3, 61))
save_png(cells[-1], FIG / "02_flow_prediction.png")
add(md(r"""
### Results: prediction

The out-of-sample R² is **negative at every horizon** (OLS: from −0.08 at T = 1 to −0.91 at T = 10; ridge shrinks the damage but stays negative), and prediction/realisation correlations stay within about ±0.05. Ten features fitted on about 700 overlapping daily observations overfit the training half; the flows of "good" traders add nothing over the market-wide flow.
"""))

# ---------------- Conclusion
add(md(r"""
## Conclusions

| # | Question | Answer |
|---|---|---|
| 1 | Is activity per token a power law? | Heavy-tailed, but power law and log-normal are statistically tied |
| 2 | Is activity per wallet heavy-tailed? | Yes: α ≈ 2.2 for purchases and sales, exponential rejected, log-normal not |
| 3 | Is the most traded NFT special? | Its daily activity is thin-tailed (power law rejected, p < 0.001) |
| 5 | Can wallets be classified? | In-sample: 52% neutral, 24% trend-following, 23% mean-reverting |
| 6 | Are the labels stable? | **No**: period-2 labels are independent of period-1 labels |
| 7 | Is there a typical reaction horizon? | No characteristic horizon between 1 and 50 days |
| 9 | Does performance persist? | **No**: top-quintile persistence 19–25% vs 20% by chance |
| 10 | Do flows predict returns? | **No**: negative out-of-sample R² at all horizons |

### Limitations

- **The "market price" is the median USD price of all sales of the day.** It mixes collections whose prices differ by orders of magnitude, so its daily log-changes (std ≈ 0.58, i.e. 58% per day) are dominated by changes in the *composition* of what trades, not by price moves. A collection-level floor-price index or a repeat-sales index would be a much better market factor, and is the first thing I would change.
- **Composition noise can create mechanical correlations.** Large, noisy and mean-reverting index moves interact with wallets whose own trades enter the median. Part of the 48% "significant" classifications may come from this rather than from behaviour, which is consistent with the absence of persistence.
- **Multiple testing.** ~9,600 wallet-level tests at 5% and a best-of-50 horizon search per wallet; no False Discovery Rate correction (e.g. Benjamini–Hochberg) was applied.
- **Overlapping horizons.** Forward returns over T days overlap for consecutive days, which inflates apparent significance; HAC standard errors or non-overlapping samples would be needed for formal tests.
- **Data issues.** Grouping on `nft_identifier` alone (Section 1) and the 92,873 unparsed timestamps (Section 4), both noted above.
- A 10-minute-window variant of the classification was also explored by the team. It is not reproduced here because it relied on `nft_updated_at` (the metadata update time, not the trade time) and allowed wallets with only 3 observations.
"""))

write(cells, ROOT / "notebooks" / "02_nft_trader_behaviour.ipynb")
