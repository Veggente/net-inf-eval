"""Calculates Bhattacharyya coefficients for a network."""

import matplotlib.pyplot as plt
import numpy as np

from net_inf_eval import sampcomp


def er_bc() -> tuple[float, float]:
    """Calculates Bhattacharyya coefficient for 2x2 ER graphs.

    Returns:
        Positive and negative BCs from one ER graph.
    """
    rs = np.random.default_rng(None)
    signed_er = rs.integers(-1, 2, (2, 2))
    return bc_4_perturbation(signed_er, (0, 1))


def spectral_radius(mat: np.ndarray) -> float:
    """Spectral radius of a matrix.

    Args:
        mat: The matrix.

    Returns:
        Spectral radius.
    """
    return max(abs(np.linalg.eig(mat)[0]))


def perturb(mat: np.ndarray, pos: tuple[int, int], val: int, rho: float) -> np.ndarray:
    """Perturbs the integer matrix by one entry and scales.

    Args:
        mat: The matrix to be perturbed.
        pos: Position of perturbation.
        val: Value of perturbation.
        rho: Target spectral radius if possible.

    Returns:
        Perturbed matrix.
    """
    pert_mat = mat.copy()
    pert_mat[pos] = val
    original_rho = spectral_radius(pert_mat)
    # The calculated spectral radius can be positive even if it should
    # be exactly zero due to computational error.  Here we only
    # normalize the matrix if its original spectral radius is greater
    # than the target rho.
    if original_rho > rho:
        pert_mat = pert_mat / original_rho * rho
    return pert_mat


def bc_4_perturbation(
    signed_graph: np.ndarray,
    pos: tuple[int, int],
    rho: float,
    num_trans: int,
    stationary: bool = True,
    obs_noise: float = 0,
) -> tuple[float, float]:
    """Calculates BC for single-entry perturbation.

    Args:
        signed_graph: A matrix.
        pos: Perturbation position.
        rho: Target spectral radius.
        num_trans: Number of transitions.
        stationary: Starts in stationary distribution.
        obs_noise: Observation noise variance.

    Returns:
        Positive and negative BCs.
    """
    pos_mat = perturb(signed_graph, pos, 1, rho)
    neg_mat = perturb(signed_graph, pos, -1, rho)
    zero_mat = perturb(signed_graph, pos, 0, rho)
    num_genes = signed_graph.shape[0]
    if stationary:
        initial_pos, _ = sampcomp.asymptotic_cov_mat(
            np.identity(num_genes),
            pos_mat,
            1,
            20,
        )
        initial_neg, _ = sampcomp.asymptotic_cov_mat(
            np.identity(num_genes),
            neg_mat,
            1,
            20,
        )
        initial_zero, _ = sampcomp.asymptotic_cov_mat(
            np.identity(num_genes),
            zero_mat,
            1,
            20,
        )
    else:
        initial_pos = np.identity(num_genes)
        initial_neg = np.identity(num_genes)
        initial_zero = np.identity(num_genes)
    cov_mat_pos = sampcomp.gen_cov_mat(
        pos_mat, 0, 1, num_trans + 1, 1, False, obs_noise, initial=initial_pos
    )
    cov_mat_neg = sampcomp.gen_cov_mat(
        neg_mat, 0, 1, num_trans + 1, 1, False, obs_noise, initial=initial_neg
    )
    cov_mat_zero = sampcomp.gen_cov_mat(
        zero_mat, 0, 1, num_trans + 1, 1, False, obs_noise, initial=initial_zero
    )
    bc_pos = sampcomp.bhatta_coeff(cov_mat_zero, cov_mat_pos)
    bc_neg = sampcomp.bhatta_coeff(cov_mat_zero, cov_mat_neg)
    return bc_pos, bc_neg


def er_2x2_avg(
    prob_conn: float, num_trans: int, auto_reg: bool, rho: float, stationary: bool
):
    """Calculates average ER BC for 2x2 matrices.

    Args:
        prob_conn: Probability of connection in ER graphs.
        num_trans: Number of temporal transitions.
        auto_reg: Perturbs autoregulation.
        rho: Target spectral radius.
        stationary: Starts in stationary distribution.

    Returns:
        Prints sorted difference and average BCs.
    """
    bc_dict = {}
    probability = 1
    for i in range(-1, 2):
        probability_i = probability * ternary_prob(i, prob_conn)
        for j in range(-1, 2):
            probability_j = probability_i * ternary_prob(j, prob_conn)
            for k in range(-1, 2):
                probability_k = probability_j * ternary_prob(k, prob_conn)
                if auto_reg:
                    mat = np.array([[0, i], [j, k]])
                    pos = (0, 0)
                else:
                    mat = np.array([[i, 0], [j, k]])
                    pos = (0, 1)
                this_bc = (
                    np.array(
                        bc_4_perturbation(
                            mat, pos, rho, num_trans, stationary=stationary
                        )
                    )
                    * probability_k
                )
                bc_dict[i, j, k] = this_bc
                print(i, j, k, this_bc)
    for key in bc_dict:
        print(key, bc_dict[key][0] - bc_dict[tuple(-digit for digit in key)][1])
    print(np.sum([val for key, val in bc_dict.items()], axis=0))


def ternary_prob(i, prob_conn):
    return prob_conn / 2 * (abs(i)) + (1 - prob_conn) * (1 - abs(i))


def er_3x3_avg(
    prob_conn: float, num_trans: int, auto_reg: bool, rho: float, stationary: bool
):
    """Calculates average ER BC for 3x3 matrices.

    Args:
        prob_conn: Probability of connection in ER graphs.
        num_trans: Number of temporal transitions.
        auto_reg: Perturbs autoregulation.
        rho: Target spectral radius.
        stationary: Starts in stationary distribution.

    Returns:
        Prints sorted difference and average BCs.
    """
    bc_dict = {}
    prob_conn = 0.9
    prob = [0 for _ in range(8)]
    for i in range(-1, 2):
        prob[0] = ternary_prob(i, prob_conn)
        for j in range(-1, 2):
            prob[1] = prob[0] * ternary_prob(j, prob_conn)
            for k in range(-1, 2):
                prob[2] = prob[1] * ternary_prob(k, prob_conn)
                for l in range(-1, 2):  # noqa: E741
                    prob[3] = prob[2] * ternary_prob(l, prob_conn)
                    for m in range(-1, 2):
                        prob[4] = prob[3] * ternary_prob(m, prob_conn)
                        for n in range(-1, 2):
                            prob[5] = prob[4] * ternary_prob(n, prob_conn)
                            for o in range(-1, 2):
                                prob[6] = prob[5] * ternary_prob(o, prob_conn)
                                for p in range(-1, 2):
                                    prob[7] = prob[6] * ternary_prob(p, prob_conn)
                                    if auto_reg:
                                        mat = np.array(
                                            [[0, i, j], [k, l, m], [n, o, p]]
                                        )
                                        pos = (0, 0)
                                    else:
                                        mat = np.array(
                                            [[i, 0, j], [k, l, m], [n, o, p]]
                                        )
                                        pos = (0, 1)
                                    this_bc = (
                                        np.array(
                                            bc_4_perturbation(
                                                mat,
                                                pos,
                                                rho,
                                                num_trans,
                                                stationary=stationary,
                                            )
                                        )
                                        * prob[7]
                                    )
                                    if np.isnan(this_bc).any():
                                        print(mat, pos, this_bc)
                                        print(0, spectral_radius(mat))
                                        pert_mat = mat.copy()
                                        pert_mat[pos] = 1
                                        print(1, spectral_radius(pert_mat))
                                        pert_mat[pos] = -1
                                        print(-1, spectral_radius(pert_mat))
                                    else:
                                        bc_dict[i, j, k, l, m, n, o, p] = this_bc
    max_abs_diff = 0
    for key in bc_dict:
        this_diff = bc_dict[key][0] - bc_dict[tuple(-digit for digit in key)][1]
        if abs(this_diff) > max_abs_diff:
            max_abs_diff = abs(this_diff)
    print(max_abs_diff)


def plot_ternary_bc_bound():
    bc_list = np.array([0.5, 0.7, 0.9])
    p10 = np.linspace(0, 1, 101)
    plt.figure()
    for bc in bc_list:
        p01_lb = 2 / 3 * (bc - np.sqrt(3 / 2 * p10 - 1 / 2)) ** 2
        plt.plot(p10, p01_lb, label=str(bc))
    plt.show()


if __name__ == "__main__":
    plot_ternary_bc_bound()
