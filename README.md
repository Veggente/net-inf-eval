# net-inf-eval
Network inference evaluation

Use this module to reproduce the figures in the paper ["Lower Bounds on
Information Requirements for Causal Network Inference" by Kang and
Hajek](https://arxiv.org/abs/2102.00055).  Note that the results may vary due
to randomness.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
```

Install the package:

```bash
pip install -e ".[dev]"
```

## Usage
1. Use `python -m net_inf_eval.ternary_er_nets` to reproduce:
- Fig. 1.  Histogram of spectral radii
- Fig. 2.  Spectral radius vs. variance
- Fig. 3.  Scatterplots of spectral radii

2. Use `python -m net_inf_eval.causal_inf_algs_vs_mlroc` to reproduce:
- Fig. 4  ROC Curves for oCSA and lasso n=10, p=0.2, T=20
- Fig. 5  ROC Curves for oCSA and lasso n=20, p=0.1, T=200

3. Use `python -m net_inf_eval.causal_inf_mlroc_vs_info_bnds` to reproduce:
- Fig. 6  ROC Curves vs info bounds n=10, p=0.2, T=20
- Fig. 7  ROC Curves vs info bounds  n=20, p=0.1, T=200

4. Use `python -m net_inf_eval.roc_vs_bnds_examples` to reproduce:
- Fig. 8 ROC and bounds for normX constant
- Fig. 9 FOC and bounds for Markov signal
