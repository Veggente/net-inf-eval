"""Generates ternary Erdos-Renyi networks and also
produces several plots related to the scaling of the networks:
     1- Histogram of spectral radii
     2- Scatterplot showing relation between S and spectral radius
     3- Scatterplots of pairs of spectral radii for weighted adjacency
        matrices differing in one edge
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import Config
from scipy.stats import rv_discrete
from tqdm import tqdm

config = Config(
    "config-net-inf-eval.toml",
    template_path=Path(__file__).parent / "config.toml.template",
)


def erdos_renyi(
    num_genes: int,
    prob_conn: float,
    spec_rad: float = 0.8,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, float]:
    """Initialize an Erdos Renyi network as in Sun–Taylor–Bollt 2015.

    If the spectral radius is positive, the matrix is normalized
    to a spectral radius of spec_rad and the scale shows the
    normalization.  If the spectral radius is zero, the returned
    matrix will have entries of 0, 1, and -1, and the scale is set
    to zero.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        spec_rad: The desired spectral radius.
        rng: Random number generator.

    Returns:
        Adjacency matrix and its scale.
    """
    rng = np.random.default_rng(rng)
    signed_edges = erdos_renyi_ternary(num_genes, prob_conn, rng)
    return scale_by_spec_rad(signed_edges, spec_rad)


def asymptotic_cov_mat(
    initial: np.ndarray, adj_mat: np.ndarray, sigma_sq: float, num_iter: int
) -> tuple[np.ndarray, float]:
    """Gets the asymptotic covariance matrix iteratively.

    Args:
        initial: Initial covariance matrix.
        adj_mat: Adjacency matrix.
        sigma_sq: Total biological variance.
        num_iter: Max number of iterations.

    Returns:
        Limiting covariance matrix and norm of the last difference.
    """
    last_cov_mat = initial
    for i in range(num_iter):
        new_cov_mat = adj_mat.T.dot(last_cov_mat).dot(adj_mat) + sigma_sq * np.identity(
            adj_mat.shape[0]
        )
        difference = np.linalg.norm(new_cov_mat - last_cov_mat)
        if difference < 0.00001:
            break
        last_cov_mat = new_cov_mat * 1.0
    return last_cov_mat, difference


def erdos_renyi_ternary(
    num_genes: int, prob_conn: float, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Generate ternary valued ER graph.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        rng: Random number generator.

    Returns:
        Adjacency matrix.
    """
    rng = np.random.default_rng(rng)
    signed_edge_dist = rv_discrete(
        values=([-1, 0, 1], [prob_conn / 2, 1 - prob_conn, prob_conn / 2])
    )
    return signed_edge_dist.rvs(size=(num_genes, num_genes), random_state=rng)


def scale_by_spec_rad(mat: np.ndarray, rho: float = 0.8) -> tuple[np.ndarray, float]:
    """Scales matrix by spectral radius.

    Args:
        mat: Matrix.
        rho: Desired spectral radius.

    Returns:
        Scaled matrix and its scale.
    """
    original_spec_rad = max(abs(np.linalg.eigvals(mat)))
    if original_spec_rad > 0.5:  # Was originally smaller
        return mat / original_spec_rad * rho, rho / original_spec_rad
    else:
        return scale_by_s(mat, 1 / (1 - rho * rho))


def scale_by_s(mat: np.ndarray, S: float) -> tuple[np.ndarray, float]:
    """Generate scaled ternary valued ER graph.

    Args:
        mat: A square matrix.
        S: Target ratio of average variance to the variation noise.

    Returns:
        mat_out and Q such that mat_out is a scaled version of mat such that:
        Q = A.T * mat_out * A  + I, and
        Trace(Q)/(row length of Q) = S.
    """
    if S <= 1:
        print("Warning: S<=1 -- can't scale")
    n = np.shape(mat)[1]
    Q = np.identity(n)
    Q_old = np.identity(n)
    identity = np.identity(n)
    x = 1.0
    eta = 0.05 / np.sqrt(n)
    while np.abs(np.trace(Q_old) / n - S) > 0.001 or np.sum(np.abs(Q - Q_old)) > 0.001:
        Q_old = Q * 1.0
        Q = x * mat.T.dot(Q).dot(mat) + identity
        x = max(x - eta * (np.trace(Q) / n - S), 0.1 * x)
    return mat * np.sqrt(x), np.sqrt(x)


def spec_rad_sample(
    num_genes: int, prob_conn: float, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Generate a ternary erdos_renyi_graph and output its spectral radius

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        rng: Random number generator.

        Returns:
            np.ndarray with two numbers:  the spectral radius with
            [0][1] entry set to zero and the spectral radius with
            [0][1] entry set to one.
    """
    rng = np.random.default_rng(rng)
    A = erdos_renyi_ternary(num_genes, prob_conn, rng)
    return max(abs(np.linalg.eigvals(A)))


def spec_rad_dist_plot(
    num_genes: int,
    prob_conn: float,
    sim: int,
    rng: np.random.Generator | int | None = None,
):
    """Generates a scatter plot of num_pairs of spec_rad
    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection
        sim: Number of spectral radii to put into histogram

        Returns:
             Produces a scatterplot
    """
    rng = np.random.default_rng(rng)
    spec_radii = np.zeros(sim)
    for i in range(sim):
        spec_radii[i] = spec_rad_sample(num_genes, prob_conn, rng)
    plt.figure()
    plt.rcParams.update({"font.size": 16})
    plt.hist(
        spec_radii,
        bins=50,
        density=False,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("spectral radius")
    plt.savefig(
        config.config["paths"]["plots_dir"]
        + "spec-rad-dist-n"
        + str(num_genes)
        + "-p"
        + str(prob_conn)
        + ".pdf",
        bbox_inches="tight",
    )
    print(
        "Fraction with spectral radius near zero =",
        np.sum(np.abs(spec_radii) < 0.5) / sim,
    )


def spec_rad_vs_s_plot(
    num_genes: int,
    prob_conn: float,
    num_pairs: int,
    rng: np.random.Generator | int | None = None,
):
    """Generate scatterplot of (S,spectral_radius) pairs

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        num_pairs: Number of points in scatterplot.
        rng: Random number generator.
        Returns:
            Produces a scatterplot.
    """
    rng = np.random.default_rng(rng)
    plt.figure()
    plt.rcParams.update({"font.size": 16})
    values = np.zeros((2, num_pairs))
    for i in tqdm(range(num_pairs)):
        S = rng.uniform(1.1, 4)
        A = erdos_renyi_ternary(num_genes, prob_conn, rng)
        A, _ = scale_by_s(A, S)
        values[:, i] = [S, max(abs(np.linalg.eigvals(A)))]
    plt.scatter(values[0, :], values[1, :])

    def f(s):
        return np.sqrt(1.0 - 1.0 / s)

    x = np.linspace(1.1, 4, 100)
    y = f(x)
    plt.plot(x, y)
    plt.ylabel("spectral radius")
    plt.xlabel("S")
    plt.savefig(
        config.config["paths"]["plots_dir"]
        + "rho-vs-s-n"
        + str(num_genes)
        + "-p"
        + str(prob_conn)
        + ".pdf",
        bbox_inches="tight",
    )


def spec_rad_pair(
    num_genes: int, prob_conn: float, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Generate an erdos_renyi_graph and output spectral radius pair

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        rng: Random number generator.

        Returns:
            np.ndarray with two numbers:  the spectral radius with
            [0][1] entry set to zero and the spectral radius with
            [0][1] entry set to one.
    """
    rng = np.random.default_rng(rng)
    A = erdos_renyi_ternary(num_genes, prob_conn, rng)
    A[0][1] = 0.0
    r0 = max(abs(np.linalg.eigvals(A)))
    A[0][1] = 1.0
    r1 = max(abs(np.linalg.eigvals(A)))
    return np.array([r0, r1])


def spec_rad_pair_plot(
    num_genes: int,
    prob_conn: float,
    num_pairs: int,
    rng: np.random.Generator | int | None = None,
):
    """Generates a scatter plot of num_pairs of spec_rad_pair's
    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        num_pairs: Number of pairs to plot.
        rng: Random number generator.

        Returns:
             Produces a scatterplot.
    """
    rng = np.random.default_rng(rng)
    values = np.zeros((2, num_pairs))
    for i in tqdm(range(num_pairs)):
        values[:, i] = spec_rad_pair(num_genes, prob_conn, rng)
    plt.figure()
    plt.rcParams.update({"font.size": 16})
    plt.scatter(values[0, :], values[1, :])
    plt.plot([values.min(), values.max()], [values.min(), values.max()])
    plt.xlabel("spec rad with edge absent")
    plt.ylabel("spec rad with edge present")
    plt.savefig(
        config.config["paths"]["plots_dir"]
        + "spec-rad-pairs-n"
        + str(num_genes)
        + "-p"
        + str(prob_conn)
        + ".pdf",
        bbox_inches="tight",
    )
    print(
        "Number with spectral radius near zero =", np.sum(np.abs(values) < 0.00000001)
    )


def main():
    plt.style.use("ggplot")
    rng = np.random.default_rng(42)
    for setting in config.config["ternary_er_nets"]["histogram"]:
        spec_rad_dist_plot(
            setting["num_genes"], setting["prob_conn"], setting["sim"], rng
        )

    for setting in config.config["ternary_er_nets"]["spectral_radius_vs_S"]:
        spec_rad_vs_s_plot(
            setting["num_genes"], setting["prob_conn"], setting["num_pairs"], rng
        )

    for setting in config.config["ternary_er_nets"]["single_edge_change"]:
        spec_rad_pair_plot(
            setting["num_genes"], setting["prob_conn"], setting["num_pairs"], rng
        )


if __name__ == "__main__":
    main()
