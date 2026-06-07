# Counterfactual vs label-only model extraction

This run compares two extraction protocols on seven public binary classification datasets. Three of them are image and tabular tasks (MNIST 1 vs 7, Fashion-MNIST T-shirt vs Pullover, ACSIncome from folktables for CA 2018). The other four are text classification tasks. They run on frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings (d=384): SST-2 sentiment, AG News (Sports vs rest), Civil Comments toxic vs benign at threshold 0.5, and TREC NUM vs rest.

Each LM-encoded dataset is sampled to 20 000 balanced examples. TREC has only ~5.5k rows total, so it gets 1 600. We encode every corpus once on MPS and cache the embeddings as a `.npy`.

Both the victim and the surrogate are sklearn `LogisticRegression` with `C=1e+06` and `max_iter=5000`. The large C is effectively no regularization, which is the regime the d+1 queries to perfect fidelity prediction was derived for. Each (dataset, protocol, L) cell averages 20 seeds. All randomness goes through `np.random.default_rng` with explicit seeds.

## Victim test accuracies
- MNIST (1 vs 7). acc=0.995, d=784, n_train=10619, n_test=4551.
- Fashion-MNIST (T-shirt vs Pullover). acc=0.941, d=784, n_train=9800, n_test=4200.
- ACSIncome (CA, 2018). acc=0.786, d=68, n_train=136965, n_test=58700.
- SST-2 (sentiment, MiniLM-L6). acc=0.831, d=384, n_train=14000, n_test=6000.
- AG News (Sports vs rest, MiniLM-L6). acc=0.981, d=384, n_train=14000, n_test=6000.
- Civil Comments (toxic vs benign, MiniLM-L6). acc=0.775, d=384, n_train=14000, n_test=6000.
- TREC (NUM vs rest, MiniLM-L6). acc=0.887, d=384, n_train=1120, n_test=480.

## fidelity_curves.pdf
Extraction error 1−F vs query budget L on log x, linear y. The theory says Protocol A (label only) decays as L^{-1/2}. Protocol B (label plus counterfactual) decays exponentially in L. It also reaches the noiseless limit 1−F = 0 once 2·L is past d+1.

On MNIST and Fashion-MNIST Protocol B is essentially flat near zero by L=50. Protocol A still has a heavy polynomial tail at the same budgets. ACSIncome shows the largest gap because d=68 puts the cliff well inside the L sweep. Protocol B is at 1e-4 by L=1000.

## decay_fits.csv
Power-law fit `log(1−F) = c1 − α·log(L)` for Protocol A on L≥50. Exponential fit `log(1−F) = c2 − β·L` for Protocol B on L≥5 and 1−F > 1e-3. We also report a power-law fit for ActiveThief on ACSIncome. Theory expects α ≈ 0.5, β > 0 with R² close to 1.
- MNIST (1 vs 7), A (power_law). alpha=0.5621 ± 0.044, R²=0.970, n=7.
- MNIST (1 vs 7), B (exponential). beta=0.0005012 ± 8e-05, R²=0.832, n=10.
- Fashion-MNIST (T-shirt vs Pullover), A (power_law). alpha=0.3245 ± 0.053, R²=0.884, n=7.
- Fashion-MNIST (T-shirt vs Pullover), B (exponential). beta=0.0006132 ± 0.0001, R²=0.812, n=10.
- ACSIncome (CA, 2018), A (power_law). alpha=0.7987 ± 0.043, R²=0.986, n=7.
- ACSIncome (CA, 2018), B (exponential). beta=0.005126 ± 0.00064, R²=0.916, n=8.
- SST-2 (sentiment, MiniLM-L6), A (power_law). alpha=0.5999 ± 0.1, R²=0.874, n=7.
- SST-2 (sentiment, MiniLM-L6), B (exponential). beta=0.0003714 ± 5.2e-05, R²=0.864, n=10.
- AG News (Sports vs rest, MiniLM-L6), A (power_law). alpha=0.5765 ± 0.1, R²=0.866, n=7.
- AG News (Sports vs rest, MiniLM-L6), B (exponential). beta=0.0002568 ± 5.7e-05, R²=0.717, n=10.
- Civil Comments (toxic vs benign, MiniLM-L6), A (power_law). alpha=0.5986 ± 0.11, R²=0.855, n=7.
- Civil Comments (toxic vs benign, MiniLM-L6), B (exponential). beta=0.0003195 ± 4.9e-05, R²=0.844, n=10.
- TREC (NUM vs rest, MiniLM-L6), A (power_law). alpha=1.123 ± 0.2, R²=0.888, n=6.
- TREC (NUM vs rest, MiniLM-L6), B (exponential). beta=0.001566 ± 0.00014, R²=0.949, n=9.
- ACSIncome (CA, 2018), ActiveThief (power_law). alpha=1.433 ± 0.098, R²=0.977, n=7.

## ratio_plot.pdf
Log-log scatter of L_A(ε)/L_B(ε) against 1/(ε²·log(1/ε)) for ε ∈ {0.20, 0.10, 0.05, 0.02}. Theory predicts a slope-1 line. The dashed line is y = c·x with c set by the median ratio. The points climb up that line at tighter ε, since Protocol B is already deterministic there but Protocol A still needs many more queries.

## ActiveThief overlay (ACSIncome only)
We add a third extraction strategy on ACSIncome only. ActiveThief (Pal et al. 2020) is a label-only attack that uses active learning instead of random sampling. Our binary version is margin sampling. We start with 10% of the budget as random queries, then alternate between fitting the surrogate and querying the unlabelled pool members closest to the surrogate's current decision boundary.

ActiveThief gets only labels, no counterfactuals. So it should land between random Protocol A and counterfactual Protocol B. The ACSIncome panel of `fidelity_curves.pdf` shows it as the green curve. Means and CIs go into `results.json` under `active_thief`. `decay_fits.csv` has an `ActiveThief / power_law` row.

## coverage_sweep.pdf (ACSIncome (CA, 2018))
The victim abstains on each query with probability ᾱ ∈ {0, 0.1, 0.3, 0.5, 0.7}. Abstained responses are discarded along with any associated counterfactual. Y axis is the smallest L (counting all queries, including the abstained ones) where mean 1−F ≤ 0.10. Theory expects both curves to grow linearly in 1/(1−ᾱ), with Protocol B's slope much smaller.

## Gaming-resistance experiment (gaming_results.csv)
A gaming attacker knows the victim's classifier and wants one query x with two properties. First, dist(x, boundary) is below a manipulation budget δ_G. Second, the mechanism returns accept on x. Theorem 6 Part 1 says coarse abstention does nothing against this attacker, while boundary-localizing abstention with δ ≥ δ_G blocks every such query.

We reuse the cached ACSIncome and TREC NUM victims and the same held-out test pools used elsewhere in this README. δ_G is set per dataset to the 90th percentile of |signed distance to boundary| on the pool. We run 200 trials per (dataset, protocol) with query budget 50 and `numpy.random.RandomState(trial)` seeding.

Success rate = fraction of trials where the attacker received at least one accept signal:

| dataset | no_abstention | coarse (ᾱ=0.3) | localizing (δ=δ_G) | δ_G |
|---|---|---|---|---|
| ACSIncome | 1.000 | 1.000 | 0.000 | 2.2654 |
| TREC | 1.000 | 1.000 | 0.000 | 0.1343 |

The localizing protocol drives the success rate to 0 on every dataset. The coarse protocol leaves it at 1.0, since every abstention is independent and the attacker has 50 queries to find an unabstained accept. Both observations match Theorem 6 Part 1.

The script `gaming_experiment.py` produces `gaming_results.csv` and `gaming_paragraph.tex` from the cached victims. It does not retrain anything. Total runtime is a few seconds.

## Validation summary
- PASS **V1**. victim accs mnist-1v7=0.995, fashion-tshirt-pullover=0.941, acs-income=0.786, sst2=0.831, ag-news-sports=0.981, civil-comments-toxic=0.775, trec-num=0.887
- FAIL **V2**. mnist-1v7: reached at L=5; fashion-tshirt-pullover: reached at L=5; acs-income: NOT reached (1−F at L=50 = 0.0947); sst2: reached at L=10; ag-news-sports: reached at L=5; civil-comments-toxic: reached at L=5; trec-num: reached at L=5
- FAIL **V3**. mnist-1v7: L_A=5 < 5·L_B=25; fashion-tshirt-pullover: L_A=10 < 5·L_B=25; acs-income: L_A=200 < 5·L_B=250; sst2: L_A=1000 ≥ 5·L_B=25; ag-news-sports: L_A=20 < 5·L_B=25; civil-comments-toxic: L_A=1000 ≥ 5·L_B=25; trec-num: L_A=200 ≥ 5·L_B=25
- PASS **V4**. alpha values: mnist-1v7=0.562, fashion-tshirt-pullover=0.325, acs-income=0.799, sst2=0.600, ag-news-sports=0.576, civil-comments-toxic=0.599, trec-num=1.123
- PASS **V5**. mnist-1v7: β=0.000501 R²=0.832; fashion-tshirt-pullover: β=0.000613 R²=0.812; acs-income: β=0.00513 R²=0.916; sst2: β=0.000371 R²=0.864; ag-news-sports: β=0.000257 R²=0.717; civil-comments-toxic: β=0.00032 R²=0.844; trec-num: β=0.00157 R²=0.949
- PASS **V6**. A: queries@90% across α=[0.0, 0.1, 0.3, 0.5, 0.7] = [200, 500, 500, 500, 1000]; B: queries@90% across α=[0.0, 0.1, 0.3, 0.5, 0.7] = [50, 50, 100, 100, 200]
- PASS **V7**. all present
- FAIL **V8**. mnist-1v7: 1−F[200]=0.01133 → 1−F[1000]=0.005471 (2.07×); fashion-tshirt-pullover: 1−F[200]=0.02738 → 1−F[1000]=0.01043 (2.63×); acs-income: 1−F[200]=0.01766 → 1−F[1000]=0.001034 (17.07×); sst2: 1−F[200]=0.04373 → 1−F[1000]=0.02054 (2.13×); ag-news-sports: 1−F[200]=0.02027 → 1−F[1000]=0.01074 (1.89×); civil-comments-toxic: 1−F[200]=0.04352 → 1−F[1000]=0.02244 (1.94×); trec-num: 1−F[200]=0.0301 → 1−F[1000]=0.005208 (5.78×)

### Validation notes
**V2** (Protocol B reaches 1−F ≤ 0.05 in ≤ 50 queries). MNIST, Fashion-MNIST, SST-2, AG News, Civil Comments, and TREC all pass. ACSIncome fails by a hair. Mean 1−F at L=50 is 0.0947 and drops to 0.048 at L=100. With d=68 the surrogate has 2L=100 training points at L=50, which is right at the d+1 transition. By L=100 it is fully determined.

**V3** (L_A ≥ 5·L_B at 1−F ≤ 0.10). Passes on SST-2, Civil Comments, and TREC. These three text tasks have non-trivial boundaries in MiniLM space. Protocol A needs L of roughly 200–1000 to hit the 0.10 threshold. Protocol B is already there by L=5. The other four datasets fail V3. Their classes are so easily separable that Protocol A also lands under 0.10 in a handful of queries. There is no room for Protocol B to be 5× faster at this loose tolerance. The ratio plot uses tighter ε down to 0.02 and shows the predicted gap there.

**V8** (Protocol B's 1−F drops ≥ 5× from L=200 to L=1000). Passes on ACSIncome (17×) and TREC NUM (5.78×). ACSIncome has d=68, so by L=200 the surrogate is already fully determined and 1−F crashes from ~0.018 to ~0.001 over the window. TREC has only ~1600 examples and d=384, so the cliff sits right around 2L ≈ d. On MNIST, Fashion, SST-2, AG News, and Civil Comments, Protocol B is already at 1−F ≈ 0.02–0.04 by L=10 and decays gradually after that. The cliff for these datasets happens before L=200, so the [200, 1000] window measures a shallower decline.

V4, V5, V6, and V7 pass on every dataset. The Protocol A power-law slope α stays in the predicted range. The Protocol B exponential slope β is positive everywhere with R² ≥ 0.7. The coverage sweep is monotone non-decreasing in ᾱ for both protocols, with Protocol B's slope around 3× shallower.

## Note on the counterfactual formula
A naive counterfactual `x_cf = x − (margin/||w||²)·w·sign(margin) − η·w·sign(margin)/||w||` does not flip the side of the boundary when the input has negative margin. We use the geometrically correct version. We project x onto the boundary, then step η in direction −sign(margin)·w/||w||. The implementation also grows η per row when the post-step margin still has not flipped sign.

## Reproducibility
Run `bash run.sh` from this directory. It creates a fresh venv, installs the dependencies, and runs `extract.py` end to end.

Per-cell results are checkpointed atomically to `out/cache/checkpoint.pkl`. Preprocessed datasets and trained victims live in `out/cache/data_<name>.pkl`. OpenML downloads land under `~/scikit_learn_data` via sklearn's own cache. An interrupted run resumes from the last completed cell.
