# Purchase Decision Pattern Analysis in E-Commerce

## IB94V0 Individual Assignment — Technical Documentation

Github Link: [github.com/norashang442-cloud/Data-Analytics-AI-Assignment---Purchase-Decision-Pattern-Analysis-in-E-Commerce](https://github.com/norashang442-cloud/Data-Analytics-AI-Assignment---Purchase-Decision-Pattern-Analysis-in-E-Commerce)

---

## Project Overview

This project analyses purchase decision patterns using clickstream data from the Retailrocket E-Commerce Dataset. The analysis unit is a **single purchase decision** (visitorid + itemid pair), not a user — the same visitor may appear multiple times with different decision characteristics as their experience accumulates. This design choice enables detection of within-platform behavioural heterogeneity that user-level aggregation would obscure.

---

## File Structure

```
├── events.csv                        # Raw Retailrocket clickstream log (primary input)
├── item_properties_part1.csv         # Item attribute snapshots (part 1)
├── item_properties_part2.csv         # Item attribute snapshots (part 2)
├── price.csv                         # Inferred price data (constructed in notebook)
├── events2.csv                       # Buyer-only events with timestamps preserved
├── events3.csv                       # Decision-level features incl. view_interval_mean
├── events4.csv                       # Main analytical dataset (view_interval_mean dropped)
├── events5.csv                       # events4 + quadrant + cluster + price/log_price
├── analysis.ipynb                    # Main analysis notebook (run top to bottom)
├── Business_Report.docx              # Submitted report
└── README.md                         # This file
```

---

## Running the Notebook

Run all cells sequentially from top to bottom. No intermediate saves are required — all intermediate DataFrames are constructed within the notebook. If re-running from a checkpoint, restart from Section 0.

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `statsmodels`

```bash
pip install pandas numpy matplotlib scipy scikit-learn statsmodels
```

**Data files** must be in the same directory as the notebook:

- `events.csv` — download from [Kaggle: Retailrocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- `item_properties_part1.csv` / `item_properties_part2.csv` — same Kaggle package

---

## Dataset Selection Note

The originally selected dataset (*Ecommerce Consumer Behavior Analysis Data*, Kaggle) was a fully synthetic dataset with variables generated independently — all pairwise correlations |r| < 0.11 and KMeans silhouette ≈ 0. It was replaced with the Retailrocket dataset, which is real clickstream data with genuine correlation structure (r = 0.29 between core variables; silhouette = 0.50 at k=4).

`item_properties_part1.csv` and `item_properties_part2.csv` are not included in this submission due to file size constraints. Download from the same Kaggle dataset package: [Retailrocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

---

## Key Data Processing Decisions

### Analysis Unit

Each row in the final dataset represents one purchase decision (visitorid + itemid). A single visitor with 10 purchases contributes 10 rows, each with that decision's own feature values — not a broadcast of their full-history aggregate.

### Exclusion of research_views = 0

2,532 records had no view events before transaction. These are excluded because their decision path is unobservable within the platform: either view events were not logged, or the purchase was initiated from an external link (e.g., ad or search engine). Both cases lie outside the scope of platform-observable decision analysis.

### Look-Ahead Bias Correction

`total_purchases`, `active_days`, and `unique_items_viewed` were initially computed from each visitor's **full event history** and broadcast to all their decisions via a simple groupby + merge — this introduced future information (events occurring after the current decision) into the feature values of past decisions.

**Fix:** All three variables were reconstructed as cumulative-to-decision-timestamp values:

- `total_purchases`: `cumcount()` on transaction events ordered by timestamp, matched to each decision's `first_trans`
- `active_days`: days elapsed from visitor's earliest event to this decision's `first_trans`
- `unique_items_viewed`: cumulative distinct item count up to `first_trans`, computed via `merge_asof`

Post-correction, the `log_total_purchases` OLS coefficient strengthened from −0.14 to −0.21, confirming the original version had suppressed the true effect rather than inflating it.

### view_interval_mean

Mean inter-view gap was computed for all decisions but excluded from the main analysis: 54.6% of records have `research_views = 1`, producing structural zeros (not missing data) that would dominate clustering distance calculations. It is retained in `events3.csv` for the sub-sample analysis (Section 13, `research_views ≥ 2`).

### Price Field Inference

No field in `item_properties` is explicitly labelled "price". Property 790 was identified as the price candidate based on: 100% numeric format (`nXXXXX.000`), 100% item coverage, time-varying values (avg. 4.3 snapshots per item), and right-skewed distribution consistent with pricing data. A sentinel value (1,199,999,999.88) appearing identically across multiple items at the same timestamp was removed as a system placeholder. Price is matched to each decision via `merge_asof` (nearest price snapshot before `first_trans`), achieving 91.3% coverage. Price is **excluded from clustering** as a product attribute rather than a behavioural variable; it is used only for post-hoc cluster profiling and regression robustness checks.

### Collinearity: unique_items_viewed Dropped

`log_unique_items_viewed` and `log_total_purchases` have r = 0.94 — behaviourally inevitable (more purchases → more items seen). `log_unique_items_viewed` was dropped from clustering inputs; `log_total_purchases` retained as the more directly interpretable experience proxy.

### Visitor-Level Non-Independence Correction

18,738 decisions come from 11,719 visitors (~1.6 decisions/visitor), so rows are not fully independent — a single visitor can contribute several correlated decisions. Three supervised-learning steps were corrected to account for this (K-Means and the Kruskal-Wallis validation tests do not use a train/test split, so they are not affected in the same way):

- **OLS regression**: `.fit(cov_type='cluster', cov_kwds={'groups': visitorid})` replaces the default i.i.d. standard errors. Coefficients are unchanged (`cov_type` only affects SE calculation); `has_addtocart`'s p-value moves slightly from 0.833/0.881 to 0.867/0.904 (still not significant), all other variables remain p<0.001.
- **5-fold CV model comparison** (OLS/Ridge/Lasso/RandomForest): `KFold` replaced with `GroupKFold` keyed on `visitorid`, so a visitor's decisions cannot span both the training and validation fold. R² shifts by ≤0.002 across all four models (e.g. RandomForest 0.5396 → 0.5375) — the original random-split estimates were only marginally optimistic.
- **Decision tree train/test split**: `train_test_split(..., stratify=y)` replaced with `StratifiedGroupKFold` (grouped by `visitorid`, one fold used as the test set), preserving cluster proportions without letting a visitor appear in both train and test. Test accuracy actually rose (91.3% → 93.4%) and train/test accuracy converged, indicating the original split was not meaningfully inflated; the tree also stopped using `total_purchases` (importance 0.050 → 0.000) since the random split had let the model exploit visitor identity as an implicit proxy for it. Per-class recall/precision were also computed for the first time (previously only overall accuracy was reported) — notably Cluster 3 (no-add-to-cart) recall rose from 78.3% (random split) to 100% (grouped split), confirming the "~9% coverage ceiling" finding is not an artefact of visitor leakage.

---

## Excluded Analytical Directions

| Approach                            | Reason for Exclusion                                                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Session-level browsing duration     | Single view events have no end timestamp; estimating dwell time requires assuming fixed session length — assumption not justifiable |
| view_interval_mean in main analysis | 54.6% structural zeros at research_views = 1; would dominate clustering                                                              |
| product categoryid as variable      | Retailrocket has not published the category label mapping; category IDs are uninterpretable without it                               |
| Location / demographic variables    | Dataset is fully anonymised; no demographic fields available                                                                         |

---

## Intermediate File Reference

| File        | Contents                                                             | Rows      |
| ----------- | -------------------------------------------------------------------- | --------- |
| events2.csv | All events for purchasing visitors, with timestamps                  | 230,678   |
| events3.csv | Decision-level features including view_interval_mean and first_trans | 18,738    |
| events4.csv | Main analytical dataset; view_interval_mean dropped                  | 18,738    |
| events5.csv | events4 + quadrant label + cluster label + price + log_price         | 18,738    |

---

## Quantifying Recommendation Trade-offs (Targeting Efficiency)

The business recommendations (streamline path for high-experience decisions, deploy trust content for high-friction decisions) were originally qualitative, with no cost-benefit backing. Unlike a binary classification problem (churn/default) with a natural false-positive/false-negative cost matrix, this project's recommendations rest on unsupervised segmentation, so a cost-matrix framing does not map cleanly onto it. Instead, targeting efficiency — how precisely an intervention rule reaches its intended segment — is quantified using the per-class precision/recall from the decision tree (see above), without inventing any cost or revenue figures the dataset does not support:

| Target cluster | Recall | Precision | Meaning |
| --- | --- | --- | --- |
| C2 High-friction (trust content) | 89.4% | 93.3% | 93.3% of decisions targeted are genuinely high-friction; 89.4% of true high-friction decisions are reached |
| C0 High-experience (streamlined path) | 97.6% | 84.2% | 97.6% of decisions that should be left alone are correctly identified; the 15.8% precision error is a false "streamline" flag on fast Lightweight decisions, a low-risk mistake |
| C3 No-add-to-cart (structural blind spot) | 100.0% | 95.3% | Confirms the ~9% coverage-ceiling estimate is not an artefact of misclassification |

This reframes the recommendations from "spread one intervention to 100% of add-to-cart events" to "target ~1/8–1/4 of decisions with segment-specific content at 84–93% precision" — a quantified reach/precision trade-off rather than a dollar-cost estimate, which is the honest limit of what an observational (non-experimental) dataset supports. Converting this into an expected-cost model would require real intervention cost and conversion-uplift data, which is out of scope (see Future Work in Note2.md).
| price.csv   | Inferred price snapshots from item_properties (property 790)         | 1,790,512 |
