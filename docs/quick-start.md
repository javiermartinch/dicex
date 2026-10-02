# Quick Start

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

The [preprint](https://www.researchgate.net/publication/415152503_Directional_Counterfactual_Explanations) describes the formulation and the algorithm, and the [API reference](api/index.md) documents every option.
