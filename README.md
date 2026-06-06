# Information Design Against Learning Adversaries (experiment code)

Experiment code for the paper *Information Design Against Learning Adversaries*
(GameSec 2026). It reproduces the figures and tables in the paper's experiments
section. This README is self-contained. You do not need the paper to follow it.

## The problem in plain terms

A binary classifier sorts each input into one of two classes. Geometrically it
splits the input space into two regions. The surface between the regions is the
decision boundary. The side an input lands on is its predicted label.

Companies often expose such a classifier behind an API. You send it an input,
called a query, and it returns a label. This creates a risk called model
extraction. An attacker sends many queries and reads the returned labels. From
those answers the attacker builds a copy of the model. The copy is the
surrogate. The original is the victim.

Fidelity `F` measures how good the copy is. It is the fraction of test inputs
where the surrogate and the victim return the same label. An `F` of 1 is a
perfect copy. The extraction error is `1 − F`. The query budget `L` is the
number of queries the attacker may send. A larger `L` gives a better copy.

## The defense

The model owner is called the principal. The principal has one defensive lever.
The API may abstain on some queries instead of returning a label. Abstain means
it answers "I will not say". The principal chooses which queries to abstain on.
A fixed budget caps the fraction of abstained queries.

Two kinds of attacker make this choice hard. They are described next.

A gaming adversary already knows the boundary. It looks for an input that sits
just on the accept side. One example is a transaction tuned to just pass a fraud
filter. A good defense abstains on queries near the boundary. The attacker then
cannot confirm an accept close to the boundary.

A learning adversary does not know the boundary. It queries in order to rebuild
the boundary. This is the model-extraction attacker described above. For this
attacker, an abstention near the boundary is a clue. It reveals that the
boundary is close to that query. The clue lets the attacker locate the boundary
by binary search.

The paper proves the two defenses are Blackwell-incomparable. Neither defense's
answers can be reconstructed from the other's. The two defenses leak different
information. They also cost the attacker different numbers of queries to rebuild
the boundary. Under fixed-rate abstention the cost is about `Θ̃(d/ε)` queries.
Under boundary-near abstention the cost is about `Θ(d·log(1/ε))` queries.
`Θ(·)` means "grows on the order of". The tilde in `Θ̃` hides smaller
log-factors. The first cost grows like `1/ε`. The second grows like `log(1/ε)`.
The second is far lower for small `ε`.

## What the code measures

The code measures how fast the extraction error `1 − F` drops as `L` grows. It
compares three attacker strategies.

A counterfactual is the nearest input on the opposite side of the boundary. It
is the smallest change that flips the label. Returning a counterfactual gives
the attacker boundary-near information. Protocol B uses this. It stands in for
the boundary-near defense.

| Strategy | What the attacker gets back | Paper object |
|---|---|---|
| Protocol A | random queries, label only | coarse channel `E_coarse` |
| ActiveThief ([Pal et al. 2020](https://arxiv.org/abs/2005.01807)) | label only, queries chosen cleverly | published label-only baseline |
| Protocol B | label and counterfactual | boundary-localizing channel `E_loc` |

A channel is a rule for what the API reveals per query.

The runs confirm the predicted decay shapes. Protocol A's error shrinks like
`L^(−1/2)`, which is a power law. Protocol B's error shrinks exponentially in
`L`. On the harder datasets Protocol B reaches the same copy quality with up to
200 times fewer queries.

### Symbols

| Symbol | Meaning |
|---|---|
| `d` | input dimension, the count of numbers describing one input |
| `L` | query budget, the number of queries allowed |
| `F` | fidelity, the fraction of test inputs where surrogate and victim agree |
| `1 − F` | extraction error, lower means a better copy |
| `ε` | target error the attacker aims for |
| `ᾱ` | abstention rate, the fraction of queries the API declines |

### Datasets

Seven binary tasks across three kinds of input.

- **Tabular.** ACSIncome, US Census income data, `d=68`.
- **Image pixels.** MNIST 1-vs-7 and Fashion-MNIST T-shirt-vs-Pullover, both `d=784`.
- **Text.** SST-2, AG News, Civil Comments, and TREC.

For the text tasks, a fixed pretrained encoder (MiniLM-L6) turns each sentence
into a 384-number vector. Fixed means the encoder is never updated. The
classifier then sees `d=384` features.

## Repository layout

```
extract.py            Main experiment. Protocols A/B and ActiveThief on 7 datasets.
                      Fits the decay rates and runs the coverage sweep. Writes to out/.
gaming_experiment.py  Gaming-resistance check. Fixed-rate abstention fails to stop a
                      gaming attacker. Boundary-near abstention blocks it.
run.sh                Creates a venv, installs deps, runs extract.py end to end.
data/                 US Census (ACS PUMS) California 2018 data. psam_p06.csv, via Git LFS.
out/                  Generated figures, CSVs, results.json, and a results report.
out/README.md         Detailed per-dataset numbers, fitted rates, and validation log.
```

## Setup

Requires Python 3 and [Git LFS](https://git-lfs.com), which is Git's extension
for large files. The 255 MB Census file (`data/2018/1-Year/psam_p06.csv`) is
stored via LFS. After cloning, fetch it with the commands below.

```bash
git lfs install   # once per machine
git lfs pull      # download the real CSV. without this you only get a tiny placeholder
```

Without this step `extract.py` crashes. It tries to read the placeholder as a
CSV. LFS is not strictly required. That file is only a cached download. To run
without LFS, delete the `data/` directory. Then `extract.py` re-downloads the
Census data on first run.

`run.sh` installs the Python dependencies into a local virtual environment
(`.venv`). These are numpy, scipy, scikit-learn, matplotlib, pandas, folktables,
torch, sentence-transformers, and datasets.

## Running it

```bash
bash run.sh                  # runs the main experiment, writes figures and CSVs to out/
python gaming_experiment.py  # gaming-resistance check, run after the line above
```

Run these from the `code/` directory. `run.sh` switches into it for you. The
ACSIncome loader (folktables) looks for `data/` in the current directory.
Running `extract.py` from elsewhere downloads a fresh copy.

`gaming_experiment.py` is a separate step. `run.sh` does not call it. It reuses
the victim models that `extract.py` saved under `out/cache/`. It trains nothing
new and finishes in a few seconds. The main run saves progress after every
`(dataset, protocol, L, seed)` setting. The progress file is
`out/cache/checkpoint.pkl`. Stopping the run partway is safe. It resumes from
the last completed setting. `out/cache/` is not committed to git. So on a fresh
clone, run `extract.py` before `gaming_experiment.py`.

## Outputs (in `out/`)

| File | What it shows |
|---|---|
| `fidelity_curves.pdf` | Extraction error `1−F` vs query budget `L`, all 7 datasets |
| `ratio_plot.pdf` | Protocol B query count relative to Protocol A, across target errors `ε` |
| `coverage_sweep.pdf` | Queries to reach 90% fidelity vs abstention rate `ᾱ` (ACSIncome) |
| `decay_fits.csv` | Fitted decay rates `α` (Protocol A) and `β` (Protocol B) |
| `gaming_results.csv` | Gaming-attack success rate under each defense |
| `results.json` | All per-setting means and confidence intervals |
| `README.md` | Full results report and validation log |

See [`out/README.md`](out/README.md) for the detailed numbers, the per-result
validation log, and a note on the counterfactual formula.
