// Copyright (c) 2026 Javier Martín-Chávez
// SPDX-License-Identifier: MIT
#![allow(clippy::useless_conversion)]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::prelude::*;

/// Computes the lower confidence limit (LCI) of the paired CVaR difference by percentile bootstrap.
///
/// Each bootstrap replicate resamples indices with replacement, applies the same indices to both
/// arrays, and records CVaR(z_c) - CVaR(z_cprime). Replicate `b` uses a PCG64 generator seeded
/// with `seed.wrapping_add(b)`, so results are reproducible and independent of the number of threads.
///
/// Args:
///     z_c (np.ndarray): 1D array of improvements for the candidate direction c.
///     z_cprime (np.ndarray): 1D array of improvements for the incumbent direction c'.
///     alpha (float): Quantile level for the CVaR computation, in (0, 1].
///     eta (float): Target percentile for the LCI (e.g., 0.05 for 95% confidence).
///     n_bootstrap (int): Number of bootstrap replicates.
///     seed (int | None): Base seed for the replicate generators. If None, the fixed seed 42 is used.
///
/// Returns:
///     tuple[float, list[float]]: The LCI and the sorted bootstrap differences.
///
/// Raises:
///     ValueError: If `z_c` and `z_cprime` do not have the same length.
#[pyfunction]
#[pyo3(signature = (z_c, z_cprime, alpha, eta, n_bootstrap, seed=None))]
fn paired_percentile_bootstrap_lci_fast<'py>(
    _py: Python<'py>,
    z_c: PyReadonlyArray1<f64>,
    z_cprime: PyReadonlyArray1<f64>,
    alpha: f64,
    eta: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> PyResult<(f64, Vec<f64>)> {
    let z_c_slice = z_c.as_slice()?;
    let z_cprime_slice = z_cprime.as_slice()?;

    let n_samples = z_c_slice.len();
    if z_cprime_slice.len() != n_samples {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "z_c and z_cprime must have the same length.",
        ));
    }

    let mut k = (alpha * n_samples as f64).ceil() as usize;
    if k < 1 {
        k = 1;
    }

    let base_seed = seed.unwrap_or(42);

    // Compute the bootstrap CVaR differences in parallel, one independent generator per replicate.
    let mut deltas: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = Pcg64::seed_from_u64(base_seed.wrapping_add(b as u64));

            let mut full_c = Vec::with_capacity(n_samples);
            let mut full_cprime = Vec::with_capacity(n_samples);

            for _ in 0..n_samples {
                let idx = rng.gen_range(0..n_samples);
                full_c.push(z_c_slice[idx]);
                full_cprime.push(z_cprime_slice[idx]);
            }

            // Partition and compute the CVaR for c.
            let _ = full_c.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());
            let cvar_c: f64 = full_c[..k].iter().sum::<f64>() / (k as f64);

            // Partition and compute the CVaR for c'.
            let _ = full_cprime.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());
            let cvar_cprime: f64 = full_cprime[..k].iter().sum::<f64>() / (k as f64);

            cvar_c - cvar_cprime
        })
        .collect();

    // Compute the eta percentile of the differences by linear interpolation.
    deltas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let i = eta * (deltas.len() - 1) as f64;
    let i_floor = i.floor() as usize;
    let i_ceil = i.ceil() as usize;
    let frac = i - i.floor();

    let lci = deltas[i_floor] * (1.0 - frac) + deltas[i_ceil] * frac;

    Ok((lci, deltas))
}

/// Computes the empirical lower tail Conditional Value at Risk (CVaR).
///
/// Args:
///     samples (np.ndarray): 1D array of samples.
///     alpha (float): Quantile level for the tail, in (0, 1].
///
/// Returns:
///     float: The empirical CVaR.
///
/// Raises:
///     ValueError: If the samples array is empty.
#[pyfunction]
fn lower_tail_cvar_empirical_fast<'py>(
    _py: Python<'py>,
    samples: PyReadonlyArray1<f64>,
    alpha: f64,
) -> PyResult<f64> {
    let slice = samples.as_slice()?;
    let n = slice.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Samples array cannot be empty.",
        ));
    }

    let mut k = (alpha * n as f64).ceil() as usize;
    if k < 1 {
        k = 1;
    }

    let mut data: Vec<f64> = slice.to_vec();
    let _ = data.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());

    let sum: f64 = data[..k].iter().sum();
    Ok(sum / (k as f64))
}

/// Generates batched points perturbed by noise along given directions.
///
/// Row `i * n_dirs + j` of the output is `x0 + noise_i * direction_j + env_noise_i`, where the
/// product is elementwise when `noise_samples` is 2D.
///
/// Args:
///     x0 (np.ndarray): 1D array with the base point.
///     directions (np.ndarray): 2D array of directions, one per row.
///     noise_samples (np.ndarray): 1D or 2D array of directional noise samples.
///     env_noise_samples (np.ndarray): 2D array of environmental noise samples.
///
/// Returns:
///     np.ndarray: 2D array of generated points of shape (n_samples * n_dirs, d).
///
/// Raises:
///     ValueError: If the dimensions of the input arrays do not match.
///     TypeError: If `noise_samples` is not a 1D or 2D NumPy array.
#[pyfunction]
fn generate_batched_points_fast<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray1<f64>,
    directions: PyReadonlyArray2<f64>,
    noise_samples: &Bound<'py, PyAny>,
    env_noise_samples: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x0_slice = x0.as_slice()?;
    let d = x0_slice.len();

    let dirs_view = directions.as_array();
    let n_dirs = dirs_view.shape()[0];

    let env_view = env_noise_samples.as_array();
    let n_samples = env_view.shape()[0];

    if env_view.shape()[1] != d || dirs_view.shape()[1] != d {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Dimension mismatch between x0, directions, and env_noise_samples.",
        ));
    }

    let mut out = numpy::ndarray::Array2::<f64>::zeros((n_samples * n_dirs, d));

    if let Ok(noise_2d) = noise_samples.downcast::<PyArray2<f64>>() {
        let noise_view = noise_2d.readonly();
        let noise = noise_view.as_array();
        if noise.shape()[0] != n_samples || noise.shape()[1] != d {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "noise_samples must have shape (n_samples, d).",
            ));
        }

        out.as_slice_mut()
            .unwrap()
            .par_chunks_mut(n_dirs * d)
            .enumerate()
            .for_each(|(i, chunk_out)| {
                // chunk_out is a flat slice of length n_dirs * d.
                let mut idx = 0;
                for j in 0..n_dirs {
                    for k in 0..d {
                        chunk_out[idx] =
                            x0_slice[k] + noise[[i, k]] * dirs_view[[j, k]] + env_view[[i, k]];
                        idx += 1;
                    }
                }
            });
    } else if let Ok(noise_1d) = noise_samples.downcast::<PyArray1<f64>>() {
        let noise_view = noise_1d.readonly();
        let noise = noise_view.as_array();
        if noise.shape()[0] != n_samples {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "noise_samples must have length n_samples.",
            ));
        }

        out.as_slice_mut()
            .unwrap()
            .par_chunks_mut(n_dirs * d)
            .enumerate()
            .for_each(|(i, chunk_out)| {
                let n_i = noise[i];
                let mut idx = 0;
                for j in 0..n_dirs {
                    for k in 0..d {
                        chunk_out[idx] = x0_slice[k] + n_i * dirs_view[[j, k]] + env_view[[i, k]];
                        idx += 1;
                    }
                }
            });
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "noise_samples must be a 1D or 2D NumPy array.",
        ));
    }

    Ok(out.into_pyarray_bound(py))
}

#[pymodule]
fn _rust_ext(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(paired_percentile_bootstrap_lci_fast, m)?)?;
    m.add_function(wrap_pyfunction!(lower_tail_cvar_empirical_fast, m)?)?;
    m.add_function(wrap_pyfunction!(generate_batched_points_fast, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::IntoPyArray;
    use pyo3::Python;

    #[test]
    fn test_lower_tail_cvar_empirical_fast() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let data = vec![10.0, 5.0, 1.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0];
            let array = data.into_pyarray_bound(py);
            let readonly = array.readonly();

            // With alpha = 0.3, k = ceil(10 * 0.3) = 3 and the mean of {1, 2, 3} is 2.
            let res = lower_tail_cvar_empirical_fast(py, readonly, 0.3).unwrap();
            assert!((res - 2.0).abs() < 1e-6);
        });
    }

    #[test]
    fn test_paired_percentile_bootstrap_lci_fast() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let z_c_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let z_cprime_data = vec![1.5, 2.5, 3.5, 4.5, 5.5];

            let z_c = z_c_data.into_pyarray_bound(py);
            let z_cprime = z_cprime_data.into_pyarray_bound(py);

            let (lci, deltas) = paired_percentile_bootstrap_lci_fast(
                py,
                z_c.readonly(),
                z_cprime.readonly(),
                0.5,
                0.05,
                10,
                Some(42),
            )
            .unwrap();

            assert_eq!(deltas.len(), 10);
            // The CVaR of z_c is smaller than the CVaR of z_cprime on every replicate.
            assert!(lci < 0.0);
        });
    }

    #[test]
    fn test_generate_batched_points_fast_1d_noise() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let x0_data = vec![1.0, 1.0];
            let directions_data = numpy::ndarray::array![[1.0, 0.0], [0.0, 1.0]];
            let noise_data = vec![0.5, -0.5];
            let env_noise_data = numpy::ndarray::array![[0.1, 0.1], [-0.1, -0.1]];

            let x0 = x0_data.into_pyarray_bound(py);
            let directions = directions_data.into_pyarray_bound(py);
            let noise = noise_data.into_pyarray_bound(py);
            let env_noise = env_noise_data.into_pyarray_bound(py);

            let any_noise = noise.into_any();
            let res = generate_batched_points_fast(
                py,
                x0.readonly(),
                directions.readonly(),
                &any_noise,
                env_noise.readonly(),
            )
            .unwrap();

            let shape = res.readonly().as_array().shape().to_vec();
            // Two noise samples times two directions yield four points of dimension 2.
            assert_eq!(shape, vec![4, 2]);
        });
    }

    #[test]
    fn test_lower_tail_cvar_empirical_fast_empty() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let data: Vec<f64> = vec![];
            let array = data.into_pyarray_bound(py);
            let res = lower_tail_cvar_empirical_fast(py, array.readonly(), 0.3);
            assert!(res.is_err());
            let err_msg = res.unwrap_err().to_string();
            assert!(err_msg.contains("ValueError"));
        });
    }

    #[test]
    fn test_paired_percentile_bootstrap_lci_fast_mismatch() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let z_c_data = vec![1.0, 2.0, 3.0];
            let z_cprime_data = vec![1.5, 2.5];

            let z_c = z_c_data.into_pyarray_bound(py);
            let z_cprime = z_cprime_data.into_pyarray_bound(py);

            let res = paired_percentile_bootstrap_lci_fast(
                py,
                z_c.readonly(),
                z_cprime.readonly(),
                0.5,
                0.05,
                10,
                Some(42),
            );
            assert!(res.is_err());
            let err_msg = res.unwrap_err().to_string();
            assert!(err_msg.contains("ValueError"));
        });
    }

    #[test]
    fn test_generate_batched_points_fast_dim_mismatch() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let x0_data = vec![1.0, 1.0];
            // The directions have dimension 3, whereas x0 has dimension 2.
            let directions_data = numpy::ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
            let noise_data = vec![0.5, -0.5];
            let env_noise_data = numpy::ndarray::array![[0.1, 0.1], [-0.1, -0.1]];

            let x0 = x0_data.into_pyarray_bound(py);
            let directions = directions_data.into_pyarray_bound(py);
            let noise = noise_data.into_pyarray_bound(py);
            let env_noise = env_noise_data.into_pyarray_bound(py);

            let any_noise = noise.into_any();
            let res = generate_batched_points_fast(
                py,
                x0.readonly(),
                directions.readonly(),
                &any_noise,
                env_noise.readonly(),
            );
            assert!(res.is_err());
            let err_msg = res.unwrap_err().to_string();
            assert!(err_msg.contains("ValueError"));
        });
    }

    #[test]
    fn test_generate_batched_points_fast_2d_noise() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let x0_data = vec![1.0, 1.0];
            let directions_data = numpy::ndarray::array![[1.0, 0.0], [0.0, 1.0]];
            // 2D noise with two samples of dimension 2.
            let noise_data = numpy::ndarray::array![[0.5, -0.5], [0.2, 0.3]];
            let env_noise_data = numpy::ndarray::array![[0.1, 0.1], [-0.1, -0.1]];

            let x0 = x0_data.into_pyarray_bound(py);
            let directions = directions_data.into_pyarray_bound(py);
            let noise = noise_data.into_pyarray_bound(py);
            let env_noise = env_noise_data.into_pyarray_bound(py);

            let any_noise = noise.into_any();
            let res = generate_batched_points_fast(
                py,
                x0.readonly(),
                directions.readonly(),
                &any_noise,
                env_noise.readonly(),
            )
            .unwrap();

            let shape = res.readonly().as_array().shape().to_vec();
            assert_eq!(shape, vec![4, 2]);
        });
    }

    #[test]
    fn test_generate_batched_points_fast_invalid_noise_type() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            use pyo3::types::PyString;
            let x0_data = vec![1.0, 1.0];
            let directions_data = numpy::ndarray::array![[1.0, 0.0], [0.0, 1.0]];
            let env_noise_data = numpy::ndarray::array![[0.1, 0.1], [-0.1, -0.1]];

            let x0 = x0_data.into_pyarray_bound(py);
            let directions = directions_data.into_pyarray_bound(py);
            let env_noise = env_noise_data.into_pyarray_bound(py);

            let invalid_noise = PyString::new_bound(py, "invalid").into_any();

            let res = generate_batched_points_fast(
                py,
                x0.readonly(),
                directions.readonly(),
                &invalid_noise,
                env_noise.readonly(),
            );

            assert!(res.is_err());
            let err_msg = res.unwrap_err().to_string();
            assert!(err_msg.contains("TypeError"));
        });
    }
}
