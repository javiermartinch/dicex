# API Reference

The public API is exported from the top-level `dicex` package; the exact solutions, the scaler and the optimizer live in their submodules.

## Explainer

| Class | Description |
|---|---|
| [`Dicex`](explainer.md#dicex.Dicex) | High-level explainer: fits the scaler and returns the recommended direction for a record. |
| [`ExplanationResult`](explainer.md#dicex.ExplanationResult) | The direction, its robust value, the risk level and the details of the search. |

## Perturbation laws

Distributions of the random step length $T$ and of the additive noise $\varepsilon$ in the execution of a direction.

| Class | Description |
|---|---|
| [`GaussianPerturbation`](perturbations.md#dicex.GaussianPerturbation) | Gaussian step length (i.i.d., per-feature, or full covariance) plus additive noise. |
| [`UniformPerturbation`](perturbations.md#dicex.UniformPerturbation) | Step length with i.i.d. uniform coordinates on `[mu - delta, mu + delta]`, plus additive noise. |
| [`CustomPerturbation`](perturbations.md#dicex.CustomPerturbation) | Step length given by a user-defined sampling function. |

## Exact solutions

| Function | Description |
|---|---|
| [`linear_directional_functional`](analytical.md#dicex.analytical.linear.linear_directional_functional), [`linear_optimal_direction`](analytical.md#dicex.analytical.linear.linear_optimal_direction) | Closed-form objective and optimal direction for linear models under Gaussian perturbations. |
| [`logistic_directional_functional`](analytical.md#dicex.analytical.logistic.logistic_directional_functional), [`logistic_optimal_direction`](analytical.md#dicex.analytical.logistic.logistic_optimal_direction) | The same for logistic models. |

## Preprocessing and optimizer

| Class | Description |
|---|---|
| [`DicexScaler`](preprocessing.md#dicex.preprocessing.scaling.DicexScaler) | Standardization used by `Dicex`; maps directions back to the original units. |
| [`VmfVns`](optimizer.md#dicex.optim.vmf_vns.VmfVns) | The vMF-VNS optimizer; its options can be passed to `Dicex` as keyword arguments. |

## Exceptions

| Module | Description |
|---|---|
| [`dicex.exceptions`](exceptions.md) | Errors and warnings raised by the library. |
