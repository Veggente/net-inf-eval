# net-inf-eval
Network inference evaluation

Use this module to reproduce the figures in the paper ["Lower Bounds on Information Requirements for Causal Network Inference" by Kang and Hajek](https://arxiv.org/abs/2102.00055).  Note that the results may vary due to randomness.

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
1. Use `python -m net_inf_eval.bhatta_bound` to reproduce Figure 1.
2. Use `python -m net_inf_eval.net_inf_eval` to reproduce Figures 2 and 3.
3. Use `python -m net_inf_eval.sampcomp` to reproduce Figure 4.
4. Use `python -c "import bhatta_bound; bhatta_bound.compare_auc_bounds(101)"`
   to reproduce Figure 5.