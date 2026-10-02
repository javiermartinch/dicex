# DiCEx: Directional Counterfactual Explanations

<p align="center">
  <a href="https://github.com/javiermartinch/dicex/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/javiermartinch/dicex/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/javiermartinch/dicex/actions/workflows/wheels.yml"><img alt="Wheels" src="https://github.com/javiermartinch/dicex/actions/workflows/wheels.yml/badge.svg"></a>
  <a href="https://javiermartinch.github.io/dicex"><img alt="Docs" src="https://github.com/javiermartinch/dicex/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/javiermartinch/dicex"><img alt="Coverage" src="https://codecov.io/gh/javiermartinch/dicex/graph/badge.svg"></a>
  <br>
  <a href="https://pypi.org/project/dicex/"><img alt="PyPI" src="https://img.shields.io/pypi/v/dicex?logo=pypi&logoColor=white"></a>
  <a href="https://pypi.org/project/dicex/"><img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue?logo=python&logoColor=white"></a>
  <a href="https://www.researchgate.net/publication/415152503_Directional_Counterfactual_Explanations"><img alt="Preprint" src="https://img.shields.io/badge/preprint-ResearchGate-00ccbb?logo=researchgate&logoColor=white"></a>
  <a href="https://github.com/javiermartinch/dicex/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
  <br>
  <a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://microsoft.github.io/pyright/"><img alt="Checked with pyright (strict)" src="https://img.shields.io/badge/pyright-strict-1674b1?logo=python&logoColor=white"></a>
  <a href="https://pre-commit.com"><img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit"></a>
  <a href="https://www.maturin.rs"><img alt="Rust + maturin" src="https://img.shields.io/badge/rust-maturin-orange?logo=rust&logoColor=white"></a>
  <a href="https://docs.renovatebot.com"><img alt="Renovate" src="https://img.shields.io/badge/renovate-enabled-brightgreen?logo=renovatebot"></a>
</p>

**DiCEx** tells you *in which direction* to change the features of a record to improve the prediction of a machine learning model, choosing the direction that remains reliable when the change is carried out imprecisely.

Classical counterfactual explanations prescribe a precise endpoint ("raise your income to exactly 52,300"). In practice, people execute recommendations with an uncertain magnitude, and an endpoint that sits next to a decision boundary can easily backfire. DiCEx instead prescribes a **direction of change** and evaluates it by the improvement it delivers when the length of the move is random, using a risk-averse criterion (the lower-tail CVaR) that rewards reliable gains rather than best-case ones. When no direction is reliably better than doing nothing, DiCEx recommends not acting.

DiCEx works with any fitted model that exposes `predict` (regression) or `predict_proba` (classification): scikit-learn estimators, gradient-boosted trees, neural networks, or your own black box. The formulation and the optimization algorithm (vMF-VNS, a derivative-free search on the sphere of directions with a Rust core) are described in the [preprint](https://www.researchgate.net/publication/415152503_Directional_Counterfactual_Explanations).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Inputs and outputs](#inputs-and-outputs)
- [Reproducing the paper](#reproducing-the-paper)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Installation

DiCEx requires Python 3.12 or later. Prebuilt wheels are provided for Linux (x86_64, aarch64), macOS (Apple silicon, Intel), and Windows (x86_64):

```bash
pip install dicex
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add dicex
```

To install from source (this compiles the Rust extension and needs a [Rust toolchain](https://rustup.rs)):

```bash
pip install git+https://github.com/javiermartinch/dicex.git
```

The only runtime dependencies are NumPy and SciPy.

## Quick start

A company's loan application has been rejected. In which direction should it improve its figures so that approval becomes likely, even if the improvements turn out larger or smaller than planned?

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dicex import Dicex, GaussianPerturbation

# Synthetic loan applications from companies. The three features are levers the company
# can act on, and they are all continuous, signed, and measured in percentage points.
features = ["operating margin (%)", "sales growth (%)", "working capital (% of sales)"]
rng = np.random.default_rng(0)
n = 3000
x_train = np.column_stack([rng.normal(5.0, 8.0, n), rng.normal(3.0, 10.0, n), rng.normal(10.0, 8.0, n)])
score = x_train @ np.array([0.10, 0.05, 0.07]) + rng.normal(0, 0.5, n)
y_train = (score > 1.2).astype(int)  # 1 = loan approved
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=0))
model.fit(x_train, y_train)

company = np.array([2.0, 0.0, 8.0])
print(f"P(approved) today: {model.predict_proba(company[None])[0, 1]:.2f}")

explainer = Dicex(
    model,
    task="classification",
    target_class=1,  # raise the probability of approval
    # The company will move along the recommended direction, but by an uncertain
    # amount: about 0.5 +/- 0.15 standard deviations of each feature.
    perturbation=GaussianPerturbation(mu=0.5, sigma=0.15),
    alpha=0.1,  # optimize the worst 10% of outcomes
    seed=42,
    verbose="none",
).fit(x_train)

result = explainer.explain(company)
# Recommended direction of change, in the units of each feature.
for name, c in zip(features, result.direction, strict=True):
    print(f"{name:>28}: {c:+.2f}")
print(f"Gain in P(approved) in the worst 10% of executions: {result.robust_value:+.2f}")
```

```text
P(approved) today: 0.15
        operating margin (%): +0.60
            sales growth (%): +0.57
working capital (% of sales): +0.56
Gain in P(approved) in the worst 10% of executions: +0.30
```

The direction gives the proportions of the change in the units of each feature, here percentage points: DiCEx recommends improving the three together, in similar measure, rather than betting on a single one. Moving along that direction with a random step raises the probability of approval by at least 0.30 in 90% of the executions.

## Inputs and outputs

**What you provide:**

| Input | Description |
|---|---|
| `model` | Any fitted model with `predict(X)` (regression) or `predict_proba(X)` (classification): scikit-learn, XGBoost, a neural network, or your own function wrapped in a class. |
| `task`, `target_class` | `"regression"` raises the prediction; `"classification"` raises the probability of the class `target_class`. |
| `perturbation` | How imprecisely the change will be carried out: the random step length `T` along the direction and, optionally, additional noise. `GaussianPerturbation(mu, sigma)`, `UniformPerturbation(mu, delta)` or `CustomPerturbation(sample_fn)`. |
| `alpha` | Risk level of the criterion: `1.0` maximizes the expected gain, `0.1` the mean of the worst 10% of outcomes. |
| `fit(x_train)` | Data used to standardize the features, so that the perturbation is expressed in standard deviations of each feature (`scaler="auto"`, the default). Use `scaler=None` to work in the original units instead. |
| `preset`, `seed` | Computational budget of the optimizer (`"low"`, `"mid"`, `"high"`) and a seed for reproducible results. |
| `explain(x0)` | The record to explain, as a 1-D array. `explain_batch(X)` explains each row of a 2-D array. |

**What you get:** an `ExplanationResult` with

| Field | Description |
|---|---|
| `direction` | Recommended direction of change: a unit vector in the original feature space. The zero vector means that no direction reliably beats not acting. |
| `robust_value` | Lower-tail CVaR, at level `alpha`, of the gain in the prediction when moving along `direction` with a random step. |
| `alpha` | Risk level used. |
| `metadata` | Details of the search, e.g. `beats_baseline` (whether acting beats not acting), `baseline_cvar` (the value of not acting), `direction_scaled`, and `n_model_evals`. |

The components of `direction` are proportions in the units of each feature, so when the features have different units, those measured on larger scales get larger components; `metadata["direction_scaled"]` gives the same direction in standard deviations of each feature, the space in which the optimizer works.

The [preprint](https://www.researchgate.net/publication/415152503_Directional_Counterfactual_Explanations) describes the formulation and the algorithm, and the [API reference](https://javiermartinch.github.io/dicex/api/) documents every option.

## Reproducing the paper

The folder [`experiments/`](https://github.com/javiermartinch/dicex/tree/main/experiments) contains the scripts, configurations, raw results, and output logs behind every figure and table of the paper and its electronic companion. It has its own locked environment; the `dicex` package installed from PyPI is only the library in `src/`.

```bash
cd experiments
uv sync              # see experiments/README.md for options without Rust or uv
./run_all.sh example figures
```

This regenerates the figures and tables from the included results in a few minutes; the collected figures are written to `experiments/paper_figures/` under the file names of the manuscript. [`experiments/README.md`](https://github.com/javiermartinch/dicex/blob/main/experiments/README.md) describes the full pipeline (runtimes, memory, random seeds) and maps each figure and table of the paper to the script that produces it.

## Development

The project uses [uv](https://docs.astral.sh/uv/) and a pinned Rust toolchain (`rust-toolchain.toml`):

```bash
git clone https://github.com/javiermartinch/dicex.git
cd dicex
uv sync --dev          # builds the Rust extension
uv run pytest          # tests
uv run pre-commit run --all-files   # ruff, pyright (strict), cargo fmt and clippy
```

See [CONTRIBUTING.md](https://github.com/javiermartinch/dicex/blob/main/CONTRIBUTING.md) for the contribution workflow.

## Citation

If you use DiCEx in your work, please cite:

> E. Carrizosa, J. Martín-Chávez, and C. Molero-Río. *Directional Counterfactual Explanations*. Preprint, 2026. <https://www.researchgate.net/publication/415152503_Directional_Counterfactual_Explanations>

```bibtex
@misc{carrizosa2026directional,
  title        = {Directional Counterfactual Explanations},
  author       = {Carrizosa, Emilio and Mart{\'\i}n-Ch{\'a}vez, Javier and Molero-R{\'\i}o, Cristina},
  howpublished = {Preprint},
  year         = {2026},
  url          = {https://www.researchgate.net/publication/415152503_Directional_Counterfactual_Explanations}
}
```

GitHub also shows this citation from [`CITATION.cff`](https://github.com/javiermartinch/dicex/blob/main/CITATION.cff).

## License

DiCEx is released under the [MIT License](https://github.com/javiermartinch/dicex/blob/main/LICENSE).
