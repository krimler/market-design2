# Information Design Against Learning Adversaries — experiments

Experiment code for the paper *Information Design Against Learning Adversaries*
(GameSec 2026). Reproduces the figures and tables in the experiments section.

## Background

A principal deploys a binary classifier behind an API and may abstain on a
bounded fraction of queries. Two adversaries pull the abstention rule in
opposite directions. A gaming adversary knows the boundary and wants an accepted
query near it; the principal counters by abstaining near the boundary. A
learning adversary does not know the boundary and queries to reconstruct it, and
against it abstaining near the boundary is the worst rule, since each abstention
reveals that the boundary is close and drives a binary search. The paper shows
the two defenses are Blackwell-incomparable, and that reconstruction costs
`Θ̃(d/ε)` queries under fixed-rate abstention versus `Θ(d·log(1/ε))` under
boundary-localizing abstention.

## What the code does

The boundary-localizing channel is realized as label + counterfactual access:
alongside the label, the victim returns the closest opposite-label point on its
boundary. Three extraction strategies are compared:

| Strategy | Access | Paper object |
|---|---|---|
| Protocol A | random queries, label only | coarse channel `E_coarse` (abstain off) |
| ActiveThief ([Pal et al. 2020](https://arxiv.org/abs/2005.01807)) | active queries, label only | published label-only baseline |
| Protocol B | random queries, label + counterfactual | boundary-localizing channel `E_loc` |

These run on seven binary tasks across three input types: tabular (ACSIncome,
`d=68`), image pixels (MNIST 1-vs-7, Fashion-MNIST, `d=784`), and frozen
MiniLM-L6 sentence embeddings (SST-2, AG News, Civil Comments, TREC, `d=384`).
Protocol A's extraction error decays as `L^-1/2` and Protocol B's exponentially
in `L`; the resulting query gap reaches 200× on the harder text tasks.

## Repository layout

```
extract.py            Main experiment: Protocols A/B + ActiveThief on 7 datasets.
                      Cross-modal rate fits, coverage sweep. Writes everything to out/.
gaming_experiment.py  Gaming-resistance check (Divergence theorem, Part 1):
                      coarse abstention fails, boundary-localizing blocks every attack.
run.sh                Creates a venv, installs deps, runs extract.py end to end.
data/                 ACS PUMS California 2018 microdata (psam_p06.csv, via Git LFS).
out/                  Generated figures, CSVs, results.json, and run report.
out/README.md         Detailed per-dataset results, fitted rates, and validation log.
```

## Setup

Requires Python 3 and [Git LFS](https://git-lfs.com). The 255 MB ACSIncome
microdata (`data/2018/1-Year/psam_p06.csv`) is stored via LFS, so after cloning:

```bash
git lfs install   # once per machine
git lfs pull      # fetch the actual CSV (otherwise the path holds a pointer stub)
```

Skip this and `extract.py` crashes parsing the LFS pointer as a CSV. LFS is not
strictly required, though: the file is just folktables' download cache (its
default `root_dir` is `data/`). To run without LFS, delete the `data/` directory
and `extract.py` re-downloads the microdata on first run (`download=True`).

`run.sh` installs the Python dependencies into a local `.venv` (numpy, scipy,
scikit-learn, matplotlib, pandas, folktables, torch, sentence-transformers,
datasets).

## Running it

```bash
bash run.sh                  # main experiment → figures + CSVs in out/
python gaming_experiment.py  # gaming-resistance check (run after extract.py)
```

Run from the `code/` directory. `run.sh` `cd`s there itself; for ACSIncome,
folktables resolves `data/` relative to the working directory, so launching
`extract.py` from elsewhere re-downloads into a fresh `data/`.

`gaming_experiment.py` is a separate entry point, not invoked by `run.sh`. It
reuses the victim models cached by `extract.py` in `out/cache/` and retrains
nothing (a few seconds). The main run checkpoints per
`(dataset, protocol, L, seed)` cell to `out/cache/checkpoint.pkl` and resumes
from the last completed cell after an interruption. `out/cache/` is git-ignored,
so on a fresh clone run `extract.py` before `gaming_experiment.py`.

## Outputs (in `out/`)

| File | Paper element |
|---|---|
| `fidelity_curves.pdf` | Extraction error `1−F` vs query budget `L`, all 7 datasets |
| `ratio_plot.pdf` | Query-ratio scatter `L_A(ε)/L_B(ε)` vs `1/(ε²·log(1/ε))` |
| `coverage_sweep.pdf` | Queries-to-90%-fidelity vs abstention rate `ᾱ` (ACSIncome) |
| `decay_fits.csv` | Fitted `α` (Protocol A power law) and `β` (Protocol B exponential) |
| `gaming_results.csv` | Gaming-attack success rate per protocol (Divergence Thm, Part 1) |
| `results.json` | All per-cell means and confidence intervals |
| `README.md` | Full results report and validation log |

See [`out/README.md`](out/README.md) for the detailed numbers, the per-theorem
validation log, and a note on the counterfactual formula.
