# DiCEx: Directional Counterfactual Explanations

**DiCEx** tells you *in which direction* to change the features of a record to improve the prediction of a machine learning model, choosing the direction that remains reliable when the change is carried out imprecisely.

Classical counterfactual explanations prescribe a precise endpoint ("raise your income to exactly 52,300"). In practice, people execute recommendations with an uncertain magnitude, and an endpoint that sits next to a decision boundary can easily backfire. DiCEx instead prescribes a **direction of change** and evaluates it by the improvement it delivers when the length of the move is random, using a risk-averse criterion (the lower-tail CVaR) that rewards reliable gains rather than best-case ones. When no direction is reliably better than doing nothing, DiCEx recommends not acting.

DiCEx works with any fitted model that exposes `predict` (regression) or `predict_proba` (classification): scikit-learn estimators, gradient-boosted trees, neural networks, or your own black box. The formulation and the optimization algorithm (vMF-VNS, a derivative-free search on the sphere of directions with a Rust core) are described in the [preprint](https://www.researchgate.net/publication/415152503_Directional_Counterfactual_Explanations).

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

## Next steps

- [Quick Start](quick-start.md): a complete example, and what DiCEx takes as input and returns.
- [Reproducing the paper](reproducing.md): the experiments behind every figure and table of the paper.
- [API Reference](api/index.md): every public class and function.
