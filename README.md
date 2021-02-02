# net-inf-eval
Network inference evaluation

Use this module to reproduce the figures in the paper ["Lower Bounds on Information Requirements for Causal Network Inference" by Kang and Hajek](https://arxiv.org/abs/2102.00055).  Note the results may vary due to randomness.
1. Install Python packages `numpy`, `matplotlib`, `scipy`, `tqdm`, and `scikit-learn`.
1. Use `python bhatta_bound.py` to reproduce Figure 1.
1. Use `python net_inf_eval.py` to reproduce Figures 2 and 3.
1. Use `python sampcomp.py` to reproduce Figure 4.
1. Use `python -c "import bhatta_bound; bhatta_bound.compare_auc_bounds(101)"` to reproduce Figure 5.
