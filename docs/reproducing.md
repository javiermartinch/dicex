# Reproducing the paper

The folder [`experiments/`](https://github.com/javiermartinch/dicex/tree/main/experiments) contains the scripts, configurations, raw results, and output logs behind every figure and table of the paper and its electronic companion. It has its own locked environment; the `dicex` package installed from PyPI is only the library in `src/`.

```bash
cd experiments
uv sync              # see experiments/README.md for options without Rust or uv
./run_all.sh example figures
```

This regenerates the figures and tables from the included results in a few minutes; the collected figures are written to `experiments/paper_figures/` under the file names of the manuscript. [`experiments/README.md`](https://github.com/javiermartinch/dicex/blob/main/experiments/README.md) describes the full pipeline (runtimes, memory, random seeds) and maps each figure and table of the paper to the script that produces it.
