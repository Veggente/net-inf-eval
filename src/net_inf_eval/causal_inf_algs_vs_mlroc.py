"""This produces ROCs for algorithms and plots them along with ML ROC genie
bounds, removing all mentions of Bhattacharyya bounds.  In particular,
recreate_stb_single() and recreate_stb_multiple() are changed.
recreate_stb_multiple() returns a dictionary with ROCs.
"""

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from config import Config
from mleroc.roc import ROC
from tqdm import tqdm

from net_inf_eval.causal_inf_mlroc_vs_info_bnds import roc_upbnd_genie_mle
from net_inf_eval.causnet_bslr import ocse
from net_inf_eval.lasso import lasso_grn
from net_inf_eval.ternary_er_nets import erdos_renyi_ternary
from net_inf_eval.ternary_er_nets import scale_by_spec_rad

config = Config(
    "config-net-inf-eval.toml",
    template_path=Path(__file__).parent / "config.toml.template",
)


def recreate_stb_single(
    num_genes: int = 200,
    prob_conn: float = 0.05,
    num_times: int = 2000,
    lasso: list[float] | None = None,
    alpha: list[float] | None = None,
    stationary: bool = True,
    scale_factor: float = 10,  # This should be overridden if stationary = False
    spec_rad: float = 10,  # This should be overridden if stationary = True
    **kwargs,
) -> tuple[dict[str, dict[float, dict[str, int]]], float | list[float] | None]:
    """Recreates a single network simulation.

       Runs Sun-Taylor-Bollt and lasso algorithms with several choices of
       pseudo parameters to trade off pfa and pdet.

    Args:
        num_genes: Number of genes.
        prob_conn: Probability of connection.
        num_times: Number of times.
        lasso: Also use lasso with l1 regularizer coefficient.
        alpha: Significance level for permutation test.
        stationary: True if X[0] has equilibrium distn, False if X[0]=0.
        scale_factor: Used only if stationary = False.
        spec_rad: Used only if stationary = True.
        **obs_noise: float
            Observation noise variance.

    Returns:
        A dictionary of performance counts for the two algorithms
            {"ocse": {0.05: {"fn": 10, "pos": 50, "fp": 3, "neg": 50}},
             "lasso": {2.0: {"fn": 12, "pos": 50, "fp": 5, "neg": 50}}}

    """
    adj_mat_ter = erdos_renyi_ternary(num_genes, prob_conn)
    if stationary:
        adj_mat, _ = scale_by_spec_rad(adj_mat_ter, spec_rad)
        total_num_times = num_times * 10
        sampled_time = slice(
            -num_times, None
        )  # Defines slice object looking at last num_times of input.
    else:
        adj_mat = adj_mat_ter * scale_factor
        total_num_times = num_times
        sampled_time = slice(None)
    data_cell = [
        gen_lin_gaussian(
            total_num_times, adj_mat, **filter_kwargs(kwargs, gen_lin_gaussian)
        )[sampled_time, :]
    ]
    count = {}
    if alpha is not None:
        count["ocse"] = {}
        for this_alpha in alpha:
            count["ocse"][this_alpha] = {}
            parents, signs = ocse(
                data_cell, 100, alpha=this_alpha, **filter_kwargs(kwargs, ocse)
            )
            full_network = np.zeros((num_genes, num_genes))
            for j in range(num_genes):
                for idx, i in enumerate(parents[j]):
                    full_network[i, j] = signs[j][idx]
            fn, pos, fp, neg = get_errors(full_network, adj_mat)
            count["ocse"][this_alpha]["fn"] = fn
            count["ocse"][this_alpha]["pos"] = pos
            count["ocse"][this_alpha]["fp"] = fp
            count["ocse"][this_alpha]["neg"] = neg
    if lasso is not None:
        count["lasso"] = {}
        for this_lasso in lasso:
            count["lasso"][this_lasso] = {}
            parents, signs = lasso_grn(data_cell, this_lasso)
            full_network = np.zeros((num_genes, num_genes))
            for j in range(num_genes):
                for idx, i in enumerate(parents[j]):
                    full_network[i, j] = signs[j][idx]
            fn, pos, fp, neg = get_errors(full_network, adj_mat)
            count["lasso"][this_lasso]["fn"] = fn
            count["lasso"][this_lasso]["pos"] = pos
            count["lasso"][this_lasso]["fp"] = fp
            count["lasso"][this_lasso]["neg"] = neg
    return count


def recreate_stb_multiple(
    sims: int = 20, **kwargs
) -> tuple[dict[str, ROC], float | list[float] | None]:
    """Recreates error estimates in Sun, Taylor, Bollt.  Runs sims number of simulations
    and for each algorithm (ocse and lasso) counts numbers of pos, neg, and errors and
    adds over all simulations.  Does for only one value of spectral radius.

    Args:
        sims: Number of simulations.
        **num_genes: int
            Number of genes.
        **prob_conn: float
            Probability of connection.
        **num_times: int
            Number of times.
        **stationary: bool
            equals TRUE if wait til process is stationary, False of X[0]=0
        **scale_factor; float
             used only if stationary = False
        **spec_rad: float
             used only if stationary = True
        **lasso: Optional[List[float]]
            lasso L1 regularizer coefficient.  To get different points along ROC
        **alpha: Optional[List[float]]
            Significance level for permutation test.   To get different points along ROC
        **obs_noise: float
            Observation noise variance.

    Returns:
        False negative ratios and false positive ratios, and
        optionally the BCs squared.
    """
    # First loop.  Will add counts from multiple simulations.
    count = recreate_stb_single(**kwargs)
    # Rest of the loops.
    for _ in tqdm(range(sims - 1)):
        new_count = recreate_stb_single(**kwargs)
        for alg in count:
            for param in count[alg]:
                for metric in count[alg][param]:
                    count[alg][param][metric] += new_count[alg][param][metric]

    res = {}
    for alg in count:
        pfa = np.zeros(len(count[alg]))
        pdet = np.zeros(len(count[alg]))
        i = 0
        for param in count[alg]:
            pfa[i] = count[alg][param]["fp"] / count[alg][param]["neg"]
            pdet[i] = 1 - count[alg][param]["fn"] / count[alg][param]["pos"]
            i = i + 1
        res[alg] = ROC(pfa, pdet, alg)
    return res


def recreate_plot_stb(
    saveas: str, plot: bool = True, from_file: str = "", **kwargs
) -> None:
    """Recreates error plots.

    Args:
        saveas: Path to save figure to.
        stationary: True if wait tile process is stationary
        scale_factor_arr: scale_factor array
        spec_rad_arr: spectral radius array.
        plot: Plots the figure.
        from_file: Load data from file.
         **stationary: bool,
         **scale_factor_arr: List[float],
         **spec_rad_arr: List[float],
         **sims: int
             Number of simulations.
         **num_genes: int
             Number of genes.
         **prob_conn: float
             Probability of connection.
         **num_times: int
             Number of times.
         **lasso: Optional[List[float]]
             lasso L1 regularizer coefficient.  To get different points along ROC
         **alpha: Optional[List[float]]
             Significance level for permutation test.   To get different points along ROC
         **obs_noise: float
             Observation noise variance.

    Returns:
        Saves plot and/or data to files.
    """
    stationary = kwargs["stationary"]

    kwargs_str = "-".join(
        [
            key + "_" + str(kwargs[key])
            for key in kwargs
            if key not in ["lasso", "alpha", "scale_factor_arr", "spec_rad_arr"]
        ]
    )
    if from_file:
        with open(from_file) as f:
            errors = json.load(f)
    else:
        errors = {}
        if not stationary:
            for scale_factor in kwargs["scale_factor_arr"]:
                errors[scale_factor] = recreate_stb_multiple(
                    scale_factor=scale_factor, spec_rad=10, **kwargs
                )
                errors[scale_factor]["gLR_bnd"] = roc_upbnd_genie_mle(
                    scale_factor=scale_factor,
                    spec_rad=10,
                    **filter_kwargs(kwargs, roc_upbnd_genie_mle),
                )
        else:
            for spec_rad in kwargs["spec_rad_arr"]:
                errors[spec_rad] = recreate_stb_multiple(
                    scale_factor=10, spec_rad=spec_rad, **kwargs
                )
                errors[spec_rad]["gLR_bnd"] = roc_upbnd_genie_mle(
                    scale_factor=10,
                    spec_rad=spec_rad,
                    **filter_kwargs(kwargs, roc_upbnd_genie_mle),
                )

        plot_roc(errors, stationary, saveas + "-" + kwargs_str)


def plot_roc(
    errors: dict[float, dict[str, ROC]], stationary: bool, saveas: str
) -> None:
    """Plots ROC curves.

    Args:
        errors: False negative ratios and false positive ratios.
        saveas: Output prefix.

    Returns:
        Saves figures.
    """
    alg_name = {"ocse": "oCSE", "lasso": "lasso", "gLR_bnd": "upper bound"}
    if stationary:
        variable_str = (
            ", $r_o =$"  # r_o (target spec+_rad) is varied if stationary = True
        )
    else:
        variable_str = ", $v_o =$"  # v_o is varied if stationary = False
    color = ("b", "orange", "g")
    plt.figure()
    for idx, scale in enumerate(errors):  # scale is scale_factor or spec_rad
        for alg in errors[scale]:
            if alg == "ocse":
                symbol = "-o"
            elif alg == "lasso":
                symbol = "--x"
            elif alg == "gLR_bnd":
                symbol = "-."
            else:
                raise ValueError("Unknown algorithm.")
            plt.plot(
                errors[scale][alg].pfa,
                errors[scale][alg].pdet,
                symbol,
                color=color[idx],
                label=alg_name[alg] + variable_str + str(scale),
            )
    plt.legend(loc="best")
    plt.xlabel("false positive rate", fontsize=16)
    plt.ylabel("true positive rate", fontsize=16)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.savefig(saveas + ".pdf", bbox_inches="tight")


def gen_lin_gaussian(
    num_times: int, adj_mat: np.ndarray, obs_noise: float = 0.0
) -> np.ndarray:
    """Generate linear Gaussian dynamics.

    Args:
        num_times: Number of times.
        adj_mat: Adjacency matrix.
        obs_noise: Observation noise variance.

    Returns:
        T-by-n array, where T and n are the numbers of times and genes.
    """
    num_genes = adj_mat.shape[0]
    data = np.empty((num_times, num_genes))
    driving_noise = np.random.randn(num_times, num_genes)
    data[0, :] = np.zeros((num_genes))  # Start with zero state
    for i in range(1, num_times):
        data[i, :] = data[i - 1, :].dot(adj_mat) + driving_noise[i, :]
    return data + np.sqrt(obs_noise) * np.random.randn(num_times, num_genes)


def get_errors(decision: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    """Get inference errors.

    The false negative ratio is defined as
    <false negative> / <condition positive>
    and the false positive ratio is defined as
    <false positive> / <condition negative>
    according to Sun�Taylor�Bollt.

    Args:
        decision: Decision array.
        truth: Ground truth array.

    Returns:
        False negative, positive, false positive, and negative.
    """
    fn_counter = 0
    fp_counter = 0
    for i in range(decision.shape[0]):
        for j in range(decision.shape[1]):
            if truth[i, j] and not decision[i, j]:
                fn_counter += 1
            if not truth[i, j] and decision[i, j]:
                fp_counter += 1
    positive = int(np.sum(abs(np.sign(truth))))
    negative = int(np.multiply(*decision.shape)) - positive
    return fn_counter, positive, fp_counter, negative


def filter_kwargs(kwargs: dict[str, Any], func: Callable) -> dict[str, Any]:
    """Filter keyword arguments for a function."""
    return {
        key: value
        for key, value in kwargs.items()
        if key in inspect.getfullargspec(func).args
    }


def script1():
    """Plots performance of algorithms and the MLROC bound
     for small networks -- 10 genes over 20 observation times
     starting with the stationary initial distribution and
     under scaled with specified spectral radius.
     Both without and with observation noise.   Produces one graph
     for each.

    Args:
        obs_noise: Observation noise.

    Returns:
        Saves figure.
    """
    for obs_noise in range(2):
        recreate_plot_stb(
            saveas=config.config["paths"]["plots_dir"] + "ocse-lasso-mlroc",
            stationary=True,
            spec_rad_arr=[0.9, 0.6, 0.2],
            num_genes=10,
            lasso=[0.6, 0.3, 0.2, 0.1, 0.06],
            alpha=[0.1, 0.3, 0.5, 0.7, 0.9],
            num_times=20,
            # Increase to 100 or more for production run.
            sims=config.config["roc_algs_vs_mlroc"]["n_sims"],
            prob_conn=0.2,
            obs_noise=obs_noise,
        )


def script2():
    """Plots performance of algorithms and the MLROC bound
     for large network -- 20 genes over 200 observation times
     starting with the zero initial distribution and specified
     scale factors.  No observation noise.   Produces one graph.

    Returns:
        Saves figure.
    """
    recreate_plot_stb(
        saveas=config.config["paths"]["plots_dir"] + "ocse-lasso-mlroc",
        stationary=False,
        scale_factor_arr=[0.15, 0.1, 0.05],
        num_genes=20,
        lasso=[0.6, 0.3, 0.2, 0.1, 0.06],
        alpha=[0.1, 0.3, 0.5, 0.7, 0.9],
        num_times=200,
        # Increase to 100 or more for production run.
        sims=config.config["roc_algs_vs_mlroc"]["n_sims"],
        prob_conn=0.1,
        obs_noise=0,
    )


def main():
    plt.style.use("ggplot")
    script1()
    script2()


if __name__ == "__main__":
    main()
