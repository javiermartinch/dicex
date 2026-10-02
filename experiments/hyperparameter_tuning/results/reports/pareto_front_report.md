# DiCEx Hyperparameter Tuning Pareto Report

Three independent multi-objective Optuna studies were executed, one per dimension bucket.

## Bucket `low`

- Cases evaluated per trial: 64
- Requested trials: 100
- Selected trial: 47
- Selected precision loss: 0.113958
- Selected mean time: 2.6649s
- YAML config: experiments/hyperparameter_tuning/results/selected_configs/low_recommended.yaml

| Trial | Loss | Mean Time (s) | Mean Evals | Balance |
| --- | --- | --- | --- | --- |
| 47 | 0.113958 | 2.6649 | 2236600.0 | 0.3731 |
| 36 | 0.108529 | 2.9580 | 2817859.4 | 0.4271 |
| 33 | 0.122435 | 2.1998 | 3453700.0 | 0.4480 |
| 57 | 0.096528 | 3.1281 | 3726226.6 | 0.4566 |
| 19 | 0.131378 | 1.9433 | 1967515.6 | 0.5404 |
| 98 | 0.096420 | 3.6995 | 4410437.5 | 0.5476 |
| 37 | 0.085417 | 5.0278 | 7019789.1 | 0.7223 |
| 41 | 0.165291 | 1.6590 | 2258828.1 | 0.8910 |

## Bucket `high`

- Cases evaluated per trial: 64
- Requested trials: 100
- Selected trial: 63
- Selected precision loss: 0.122077
- Selected mean time: 12.0866s
- YAML config: experiments/hyperparameter_tuning/results/selected_configs/high_recommended.yaml

| Trial | Loss | Mean Time (s) | Mean Evals | Balance |
| --- | --- | --- | --- | --- |
| 63 | 0.122077 | 12.0866 | 11368878.1 | 0.4623 |
| 43 | 0.157054 | 11.6693 | 10957296.9 | 0.5194 |
| 75 | 0.161440 | 9.1684 | 6342209.4 | 0.5433 |
| 73 | 0.112605 | 15.2684 | 13278796.9 | 0.5682 |
| 92 | 0.179915 | 9.0845 | 7950556.2 | 0.6443 |
| 58 | 0.185085 | 7.2634 | 6028140.6 | 0.6725 |
| 47 | 0.189929 | 6.5150 | 4778628.1 | 0.6990 |
| 4 | 0.109789 | 20.9268 | 18569728.1 | 0.7136 |
