"""Calculations related to the Bhattacharyya bounds."""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
'''
plt.style.use("ggplot")


def half_norm(vec: List[float]) -> float:
    """Half norm.

    Args:
        vec: Vector.

    Returns:
        The half norm of vec.
    """
    return np.sum(np.sqrt(vec)) ** 2


def check_anti_triangle():
    """Checks the anti-triangle inequality for the half norm.

    A.K.A. the reverse Minkowski inequality.
    """
    dim = 20
    for _ in range(10000):
        vec1 = np.random.rand(dim)
        vec2 = np.random.rand(dim)
        norm_of_sum = half_norm(vec1 + vec2)
        sum_of_norm = half_norm(vec1) + half_norm(vec2)
        if norm_of_sum < sum_of_norm:
            print("Anti-triangle inequality is false.")
            print(vec1, vec2)
            return False
    print("No counter-examples found.")
    return True


def err_bounds_dirichlet():  # pylint: disable=too-many-locals
    """Plots error bounds for Dirichlet experiments."""
    bc_sq_on_marginal_list = []
    avg_bc_sq_list = []
    avg_bc_list = []
    lb_comp_list = []
    lb_mix_list = []
    avg_error_list = []
    num_vals = 10
    for _ in range(1000):
        coin_prob = np.random.rand()
        dist = {}
        for i in range(2):
            for j in {"a", "b"}:
                # Dirichlet distribution.
                dist[i, j] = np.random.exponential(size=num_vals)
                dist[i, j] = dist[i, j] / sum(dist[i, j])
        bc_sq_on_marginal = rho_sq(
            coin_prob, dist[0, "a"], dist[0, "b"], dist[1, "a"], dist[1, "b"]
        )
        avg_bc_sq = rho_sq_bar(
            coin_prob, dist[0, "a"], dist[0, "b"], dist[1, "a"], dist[1, "b"]
        )
        avg_bc = rho_bar(
            coin_prob, dist[0, "a"], dist[0, "b"], dist[1, "a"], dist[1, "b"]
        )
        lb_comp = (
            1
            / 2
            * (
                1
                - coin_prob * np.sqrt(1 - half_norm(dist[0, "a"] * dist[1, "a"]))
                - (1 - coin_prob) * np.sqrt(1 - half_norm(dist[0, "b"] * dist[1, "b"]))
            )
        )
        lb_mix = 1 / 2 * (1 - np.sqrt(1 - bc_sq_on_marginal))
        avg_error = (
            1
            / 2
            * sum(
                np.minimum(
                    mix_prob(coin_prob, dist[0, "a"], dist[0, "b"]),
                    mix_prob(coin_prob, dist[1, "a"], dist[1, "b"]),
                )
            )
        )
        bc_sq_on_marginal_list.append(bc_sq_on_marginal)
        avg_bc_sq_list.append(avg_bc_sq)
        avg_bc_list.append(avg_bc)
        lb_comp_list.append(lb_comp)
        lb_mix_list.append(lb_mix)
        avg_error_list.append(avg_error)
    curves = np.array(
        [
            [i, j, k]
            for i, j, k in sorted(zip(lb_mix_list, avg_error_list, lb_comp_list))
        ]
    )
    plt.plot(curves[:, 0], label=r"$\mathsf{LB}_{\mathrm{direct}}$")
    plt.plot(curves[:, 1], label=r"$p_e^{\mathrm{opt}}$")
    plt.plot(curves[:, 2], label=r"$\mathsf{LB}_{\text{side-info}}$")
    plt.legend(loc="best")
    plt.xlabel("trial")
    plt.ylabel("error probability")
    plt.savefig("conjecture-d2-n10.eps")


def check_conjecture_special():
    """Checks the validity of the conjecture in a special case."""
    term_list = []
    for _ in range(1000):
        pdf = {}
        for i in range(2):
            for s in ["a", "b"]:  # pylint: disable=invalid-name
                pdf[i, s] = np.random.rand()
        terms = [
            np.sqrt(
                (pdf[0, "a"] + pdf[0, "b"])
                * (pdf[1, "a"] + pdf[1, "b"])
                * (2 - pdf[0, "a"] - pdf[0, "b"])
                * (2 - pdf[1, "a"] - pdf[1, "b"])
            ),
            -(pdf[0, "a"] - pdf[0, "b"]) * (pdf[1, "a"] - pdf[1, "b"]),
            -2
            * np.sqrt(
                pdf[0, "a"] * pdf[1, "a"] * (1 - pdf[0, "a"]) * (1 - pdf[1, "a"])
            ),
            -2
            * np.sqrt(
                pdf[0, "b"] * pdf[1, "b"] * (1 - pdf[0, "b"]) * (1 - pdf[1, "b"])
            ),
        ]
        term_list.append(terms)
        if sum(terms) < 0:
            print("Counter example found.")
            print(pdf, terms)
            return False
        if terms[1] < -0.5:
            print(pdf, terms)
    print("No counter examples found.")
    curves = np.array([[i, j, k, l] for i, j, k, l in sorted(term_list)])
    plt.plot(curves[:, 0], label="first term")
    plt.plot(curves[:, 1], label="second term")
    plt.plot(curves[:, 2], label="third term")
    plt.plot(curves[:, 3], label="fourth term")
    plt.legend(loc="upper left")
    plt.xlabel("trial")
    plt.ylabel("terms in summation")
    plt.savefig("conjecture-special.eps")
    return True


def check_conjecture_higher_dim(  # pylint: disable=too-many-branches
    num_vals, num_switch_vals, no_small_terms=False, plot=False
):  # pylint: disable=too-many-locals
    """Checks conjecture for higher dimensional cases."""
    bc_sq_on_marginal_list = []
    avg_bc_sq_list = []
    small_term_dict = {}
    for _ in range(10000):
        coin_prob = np.random.rand(num_switch_vals)
        coin_prob = coin_prob / sum(coin_prob)
        dist = {}
        marginal = []
        avg_bc_sq = 0
        for i in range(2):
            marginal.append(0)
            for j in range(num_switch_vals):
                dist[i, j] = np.random.rand(num_vals)
                dist[i, j] = dist[i, j] / sum(dist[i, j])
                marginal[i] += coin_prob[j] * dist[i, j]
        for j in range(num_switch_vals):
            avg_bc_sq += coin_prob[j] * half_norm(dist[0, j] * dist[1, j])
        marginal = np.array(marginal)
        bc_sq_on_marginal = half_norm(marginal[0] * marginal[1])
        bc_sq_on_marginal_list.append(bc_sq_on_marginal)
        avg_bc_sq_list.append(avg_bc_sq)
        if bc_sq_on_marginal < avg_bc_sq:
            print("Found counter example.")
            print(dist, bc_sq_on_marginal, avg_bc_sq)
            return False
        avg_pmf = {}
        for i in range(num_vals - 1):
            if (1, i) not in avg_pmf:
                avg_pmf[0, i] = get_avg_pmf(coin_prob, dist, 0, i)
                avg_pmf[1, i] = get_avg_pmf(coin_prob, dist, 1, i)
            for j in range(i + 1, num_vals):
                if (i, j) not in small_term_dict:
                    small_term_dict[i, j] = []
                if (1, j) not in avg_pmf:
                    avg_pmf[0, j] = get_avg_pmf(coin_prob, dist, 0, j)
                    avg_pmf[1, j] = get_avg_pmf(coin_prob, dist, 1, j)
                small_term = 1 / (num_vals - 1) * (
                    sum(
                        coin_prob
                        * np.array([dist[0, k][i] for k in range(num_switch_vals)])
                        * (
                            avg_pmf[1, i]
                            - np.array([dist[1, k][i] for k in range(num_switch_vals)])
                        )
                    )
                    + sum(
                        coin_prob
                        * np.array([dist[0, k][j] for k in range(num_switch_vals)])
                        * (
                            avg_pmf[1, j]
                            - np.array([dist[1, k][j] for k in range(num_switch_vals)])
                        )
                    )
                ) + 2 * (
                    np.sqrt(
                        avg_pmf[0, i] * avg_pmf[1, i] * avg_pmf[0, j] * avg_pmf[1, j]
                    )
                    - np.inner(
                        coin_prob,
                        np.sqrt(
                            [
                                dist[0, k][i]
                                * dist[1, k][i]
                                * dist[0, k][j]
                                * dist[1, k][j]
                                for k in range(num_switch_vals)
                            ]
                        ),
                    )
                )
                small_term_dict[i, j].append(small_term)
    print("No counter examples found.")
    if not plot:
        return True
    if no_small_terms:
        curves = np.array(
            [
                [i, j]
                for i, j in sorted(
                    zip(
                        avg_bc_sq_list,
                        bc_sq_on_marginal_list,
                    )
                )
            ]
        )
    else:
        curves = np.array(
            [
                [i, j, k, l, m]
                for i, j, k, l, m in sorted(
                    zip(
                        avg_bc_sq_list,
                        bc_sq_on_marginal_list,
                        small_term_dict[0, 1],
                        small_term_dict[0, 2],
                        small_term_dict[1, 2],
                    )
                )
            ]
        )
        plt.plot(curves[:, 2], label="small term (1, 2)")
        plt.plot(curves[:, 3], label="small term (1, 3)")
        plt.plot(curves[:, 4], label="small term (2, 3)")
    plt.plot(curves[:, 1], color="C3", label=r"$\mathsf{BC}^2$ (on marginals)")
    plt.plot(curves[:, 0], color="C4", label=r"$\overline{\mathsf{BC}^2}$ (on joint)")
    plt.legend(loc="best")
    plt.xlabel("trial")
    plt.ylabel("Bhattacharyya coefficient")
    plt.savefig(
        "conjecture-d{}-n{}-nst{}.eps".format(
            num_switch_vals, num_vals, no_small_terms
        )
    )
    return True


def get_avg_pmf(coin_prob, dist, hyp, val):
    """Gets average pmf."""
    return np.inner(coin_prob, [dist[hyp, i][val] for i in range(len(coin_prob))])


def plot_bc2_det():
    """Plots determinant of BC^2."""
    f = g = np.linspace(0.01, 0.99, 99)  # pylint: disable=invalid-name
    f, g = np.meshgrid(f, g)  # pylint: disable=invalid-name
    detbc2 = (
        1 / 4 * (1 / f + 1 / (1 - f)) * (1 / g + 1 / (1 - g))
        - 4
        - 2
        * (np.sqrt((1 - g) / g) - np.sqrt(g / (1 - g)))
        * (np.sqrt((1 - f) / f) - np.sqrt(f / (1 - f)))
        - 1 / 4 * ((1 - g) / g + g / (1 - g) - 2) * ((1 - f) / f + f / (1 - f) - 2)
    )
    print(np.min(detbc2))
    fig = plt.figure()
    ax = fig.gca(projection="3d")  # pylint: disable=invalid-name
    ax.plot_surface(f, g, detbc2)
    plt.show()


def rho_sq(
    coin_prob: float,
    dist1: np.ndarray,
    dist2: np.ndarray,
    dist3: np.ndarray,
    dist4: np.ndarray,
) -> float:
    """Calculates BC for marginal distributions.

    Args:
        coin_prob: Head probability of coin.
        dist1: Distribution under H0, head.
        dist2: Distribution under H0, tail.
        dist3: Distribution under H1, head.
        dist4: Distribution under H1, tail.

    Returns:
        BC between two mixture hypotheses.
    """
    return half_norm(
        (coin_prob * dist1 + (1 - coin_prob) * dist2)
        * (coin_prob * dist3 + (1 - coin_prob) * dist4)
    )


def rho_sq_bar(
    coin_prob: float,
    dist1: np.ndarray,
    dist2: np.ndarray,
    dist3: np.ndarray,
    dist4: np.ndarray,
) -> float:
    """Calculates average BC^2.

    Args:
        coin_prob: Head probability of coin.
        dist1: Distribution under H0, head.
        dist2: Distribution under H0, tail.
        dist3: Distribution under H1, head.
        dist4: Distribution under H1, tail.

    Returns:
        Average BC^2.
    """
    return coin_prob * half_norm(dist1 * dist3) + (1 - coin_prob) * half_norm(
        dist2 * dist4
    )


def rho_bar(
    coin_prob: float,
    dist1: np.ndarray,
    dist2: np.ndarray,
    dist3: np.ndarray,
    dist4: np.ndarray,
) -> float:
    """Calculates average BC.

    Args:
        coin_prob: Head probability of coin.
        dist1: Distribution under H0, head.
        dist2: Distribution under H0, tail.
        dist3: Distribution under H1, head.
        dist4: Distribution under H1, tail.

    Returns:
        Average BC.
    """
    return (
        coin_prob * half_norm(dist1 * dist3) ** 0.5
        + (1 - coin_prob) * half_norm(dist2 * dist4) ** 0.5
    )


def mix_prob(coin_prob: float, dist_a: List[float], dist_b: List[float]) -> List[float]:
    """Mixes probability distributions.

    Args:
        coin_prob: Coin probability for distribution A.
        dist_a: Distribution A.
        dist_b: Distribution B.

    Returns:
        Mixed distribution.
    """
    return coin_prob * np.array(dist_a) + (1 - coin_prob) * np.array(dist_b)


def auc_bound(rho: float, points: int) -> float:
    """Calculates our new AUC upper bound.

    Args:
        rho: Bhattacharyya coefficient.
        points: Number of points of FPR in numerical estimation.

    Returns:
        New upper bound on AUC.
    """
    return np.mean(new_roc_bound([rho ** 2], np.linspace(0, 1, points))[:-1])


def new_roc_bound(rho_sq_list: List[float], fpr: List[float]) -> np.ndarray:
    """Calculates our new ROC upper bound.

    Args:
        rho_sq_list: Bhattacharyya coefficients.
        fpr: FPR in numerical estimation.

    Returns:
        New upper bound on the ROC curve.
    """
    bound = np.ones(len(fpr))
    for pi in np.linspace(0.01, 0.99, 99):  # pylint: disable=invalid-name
        new_bound = np.mean(
            [new_roc_bound_w_pi(rho_sq, np.array(fpr), pi) for rho_sq in rho_sq_list],
            axis=0,
        )
        bound = np.minimum(bound, new_bound)
    return bound


def new_roc_bound_w_pi(  # pylint: disable=invalid-name
    bc2: float, fpr: np.ndarray, pi: float
) -> np.ndarray:
    """Calculates our new ROC upper bound for parameter pi.

    Args:
        bc2: Bhattacharyya coefficient squared.
        fpr: False positive rates.
        pi: Prior for H1 (i.e., weight for FNR).

    Returns:
        New upper bound on ROC with pi.
    """
    return (
        1 - (1 / 2 * (1 - np.sqrt(1 - 4 * pi * (1 - pi) * bc2)) - (1 - pi) * fpr) / pi
    )


def compare_auc_bounds(points: int) -> None:
    """Compares AUC upper bounds.

    Args:
        points: Number of points of FPR in numerical estimation.

    Returns:
        Plots figure of Shapiro bound and ours.
    """
    rho = np.linspace(0, 1, 101)
    shapiro = 1 - (1 - np.sqrt(1 - rho ** 2)) ** 2 / 2
    ours = [auc_bound(this_rho, points) for this_rho in rho]
    ours_weaker = 1 - rho ** 4 / 6
    plt.figure()
    plt.plot(rho, shapiro, label="Shapiro's bound")
    plt.plot(rho, ours, label="our tighter, numerical bound")
    plt.plot(rho, ours_weaker, label="our looser, analytical bound")
    plt.legend()
    plt.xlabel(r"$\rho$")
    plt.ylabel("upper bound on AUC")
    plt.savefig("auc_bound.eps")


def nonconcavity():
    """Shows a nonconcavity example."""
    pi_list = np.linspace(0, 1, 101)
    f_1 = np.array([1 / 2, 0, 1 / 2])
    f_2 = np.array([0, 1, 0])
    g_1 = np.array([3 / 4, 0, 1 / 4])
    g_2 = f_2
    f = 2 / 3 * f_1 + 1 / 3 * f_2
    g = 2 / 3 * g_1 + 1 / 3 * g_2  # pylint: disable=invalid-name
    lb_mixed = (
        1 / 2 * (1 - np.sqrt(1 - 4 * pi_list * (1 - pi_list) * bhatta_coeff(f, g)))
    )
    lb_component = (
        1
        / 2
        * (
            1
            - 2 / 3 * np.sqrt(1 - 4 * pi_list * (1 - pi_list) * bhatta_coeff(f_1, g_1))
            - 1 / 3 * np.sqrt(1 - 4 * pi_list * (1 - pi_list) * bhatta_coeff(f_2, g_2))
        )
    )
    print(lb_mixed[50], lb_component[50])
    plt.figure()
    plt.plot(pi_list, lb_mixed, label=r"$\mathsf{LB}_m$")
    plt.plot(pi_list, lb_component, label=r"$\mathsf{LB}_c$")
    plt.legend()
    plt.show()


def bhatta_coeff(f: np.ndarray, g: np.ndarray) -> float:  # pylint: disable=invalid-name
    """Calculates the BC for two discrete distributions.

    Args:
        f: Distribution 1.
        g: Distribution 2.

    Returns:
        The BC.
    """
    return sum(np.sqrt(f * g))


if __name__ == "__main__":
    err_bounds_dirichlet()
