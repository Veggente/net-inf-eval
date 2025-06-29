"""Calculates ROC bounds for causal network inference with all-but-one side
information.

One bound is the MLROC based on likelihood ratio samples. Others are based on
Bhattacharyya coefficient or KL diverence.   For small network size (10 genes
and 20 observation times, plugin method can be used for all three -- see
script1(). That starts with stationary distribution and can have observation
noise.

For large network size (eg. 20 genes and 200 observation times) we need to
consider intial state zero and constant scale factor and no observation noise.
Then we can compare MLROC and KL based bounds -- see script2().

script3 and script4 use the zeroA version of MLROC or KL bounds in the case of
zero intial condition with no observation noise and compare the computation
using the plugin method to rollout method.   There is excellent agreement
giving confidence of correct implementation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import Config
from mleroc.estimators import MLE
from mleroc.roc import ROC
from scipy.linalg import eigh
from scipy.stats import multivariate_normal

from net_inf_eval.ternary_er_nets import asymptotic_cov_mat
from net_inf_eval.ternary_er_nets import erdos_renyi_ternary
from net_inf_eval.ternary_er_nets import scale_by_spec_rad

config = Config(
    "config-net-inf-eval.toml",
    template_path=Path(__file__).parent / "config.toml.template",
)


def big_covar_matrix(
    A: np.ndarray, T: int, Q: np.ndarray, sigma_sq: float, obs_noise: float = 0.0
) -> np.ndarray:
    """Generates (T+1)n x (T+1)n or Tn x Tn covariance matrix.

    Args:
        A: square matrix
        T: observations go from t=0 to t=T
        Q: covariance matrix of X(0), same shape as A
        sigma_sq : individual noise variance
        obs_noise: observation noise variance

    Returns:
        Covariance matrix of [Y(0), . . . , Y(T)] assuming
        X(t+1) = X(t) A + W(t)   Y(t) = X(t) + Z(t)
           Cov(X(0))=Q  Cov(W(t))=sigma_sq*I  Cov(Z(t))=obs_noise*I
           if Q is nonsingular.   If Q is singular (e.g. all zero matrix)
           the function will return covariance matrix of
           [Y(1), . . . , Y(T)]

    """

    n = np.shape(A)[1]
    sigma = np.zeros((T + 1) * (T + 1) * n * n).reshape(T + 1, T + 1, n, n)
    # Will set sigma[s,t,:,:] to equal cov(X(s),X(t)).
    # Later will flatten sigma to a (T+1)n x (T+1)n covariance matrix.
    sigma[0, 0, :, :] = Q
    # So 1\leq t \leq T.  Fill diagonal blocks first.
    for t in range(1, T + 1):
        sigma[t, t, :, :] = A.T.dot(sigma[t - 1, t - 1, :, :]).dot(
            A
        ) + sigma_sq * np.identity(n)
    for t in range(T + 1):
        for s in range(t + 1, T + 1):
            sigma[t, s, :, :] = sigma[t, s - 1, :, :].dot(A)
            sigma[s, t, :, :] = A.T.dot(sigma[s - 1, t, :, :])
    sigma = sigma.transpose(0, 2, 1, 3).reshape((T + 1) * n, (T + 1) * n)
    sigma = sigma + obs_noise * np.identity((T + 1) * n)

    eigval, eigvec = eigh(Q)
    if eigval[0] > 0.000001:
        return sigma
    else:  # Q is singular so we leave out first T rows and columns.
        return sigma[n:, n:]


def lr_sample(
    num_genes: int,
    prob_conn: float,
    sigma_sq: float,
    stationary: bool,
    scale_factor: float,
    spec_rad: float,
    num_times: int,
    obs_noise: float = 0,
    b: int = 0,
) -> float:
    """Generate a sample or likelihood ratio assuming genie and stationary
    initial covariance matrix.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        sigma_sq: variance in state equations.
        stationary: True if start in stationary distribution, False if start
            from 0.
        scale_factor: magnitude of weights, used only if stationary = False.
        spec_rad: spectral radius, used only if stationary = True.
        num_times: one plus number of observation times, starting at zero.
        obs_noise: variance of observation noise.
        b: activation of link 1->0 should be 0 or 1.

        Returns:
            Sample of likelihood ratio
    """
    if stationary:
        A0 = erdos_renyi_ternary(num_genes, prob_conn)
        A1 = A0.copy()
        A0[1][0] = 0.0
        A1[1][0] = 1.0  # By symmetry we are assuming R[1,0] = 1
        mean = np.zeros((num_times + 1) * num_genes)
        A0, _ = scale_by_spec_rad(A0, spec_rad)
        A1, _ = scale_by_spec_rad(A1, spec_rad)
        Q0, _ = asymptotic_cov_mat(np.identity(num_genes), A0, sigma_sq, 20)
        Q1, _ = asymptotic_cov_mat(np.identity(num_genes), A1, sigma_sq, 20)
        Sigma0 = big_covar_matrix(A0, num_times, Q0, sigma_sq, obs_noise)
        Sigma1 = big_covar_matrix(A1, num_times, Q1, sigma_sq, obs_noise)
        mvn0 = multivariate_normal(mean=mean, cov=Sigma0)
        mvn1 = multivariate_normal(mean=mean, cov=Sigma1)
        if b == 0:
            Y = mvn0.rvs()
        else:
            Y = mvn1.rvs()
        denominator = mvn0.pdf(Y)
        if denominator == 0:
            return 1000000.0
        return max(min(1000000.0, mvn1.pdf(Y) / denominator), 0.000001)
    else:
        myNormal = multivariate_normal(mean=np.zeros(num_genes), cov=np.eye(num_genes))
        A = scale_factor * erdos_renyi_ternary(num_genes, prob_conn)
        A[1, 0] = scale_factor * b  # By symmetry we are assuming R[1,0] = 1.
        X = np.zeros(num_genes)
        SUM = 0.0

        for t in range(num_times):
            # Only need W[0] and previous state to update sum.
            W = myNormal.rvs()
            SUM += scale_factor * X[1] * (W[0] + scale_factor * (b - 0.5) * X[1])
            X = np.dot(X, A) + W  # Update state.
        if SUM > 30:
            return 10000
        if SUM < -30:
            return 0.0001
        return np.exp(SUM)


def lr_sample_zero_a(
    num_genes: int,
    prob_conn: float,
    sigma_sq: float,
    scale_factor: float,
    num_times: int,
    b: int = 0,
) -> float:
    """Generate a spample or likelihood ratio assuming genie and initial
       state zero and observation noise zero.  Using direct method with
       big covariance matrix.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        sigma_sq: variance in state equations
        scale_factor: magnitude of nonzero edge weights
        num_times: one plus number of observation times
        b: activation of link 1->0 should be 0 or 1

        Returns:
            Sample of likelihood ratio
    """
    A0 = erdos_renyi_ternary(num_genes, prob_conn) * scale_factor
    A1 = A0.copy()
    A0[1][0] = 0.0
    A1[1][0] = 1.0 * scale_factor
    obs_noise = 0.0
    mean = np.zeros(num_times * num_genes)  # No longer num_times+1.
    Q_null = np.zeros((num_genes, num_genes))
    Sigma0 = big_covar_matrix(A0, num_times, Q_null, sigma_sq, obs_noise)
    Sigma1 = big_covar_matrix(A1, num_times, Q_null, sigma_sq, obs_noise)
    mvn0 = multivariate_normal(mean=mean, cov=Sigma0)
    mvn1 = multivariate_normal(mean=mean, cov=Sigma1)
    if b == 0:
        Y = mvn0.rvs()
    else:
        Y = mvn1.rvs()
    denominator = mvn0.pdf(Y)
    if denominator == 0:
        return 10000.0
    return max(min(10000.0, mvn1.pdf(Y) / denominator), 0.0001)


def roc_upbnd_genie_mle(
    num_genes: int,
    prob_conn: float,
    stationary: bool,
    scale_factor: float,
    spec_rad: float,
    num_times: int,
    num_LR_samples: int = 500,
    obs_noise: float = 0,
    sigma_sq: float = 1.0,
    **kwargs,
) -> ROC:
    """Generate an ROC under conditions that network matrix is available to
    estimator for all other links using the method of generating LR samples
    and using the MLE estimator or ROC.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        stationary: True if start in stationary distribution, False if start
            from 0.
        scale_factor: magnitude of weights, used only if stationary = False.
        spec_rad: spectral radius, used only if stationary = True.
        num_times: one plus number of observation times, starting at zero.
        num_LR_samples: number of LR sample produced, half with link 1->0 on.
        obs_noise: variance of observation noise.
        sigma_sq: variance in state equations.

    Returns an ROC object which is a curve that can be used by caller to plot
    Does not include plt.figure(), plt.legend(), or plt.show().
    """

    LR_samples = np.zeros(num_LR_samples)
    for k in range(num_LR_samples // 2):
        LR_samples[k] = lr_sample(
            num_genes,
            prob_conn,
            sigma_sq,
            stationary,
            scale_factor,
            spec_rad,
            num_times,
            obs_noise,
            b=0,
        )
    for k in range(num_LR_samples // 2, num_LR_samples):
        LR_samples[k] = lr_sample(
            num_genes,
            prob_conn,
            sigma_sq,
            stationary,
            scale_factor,
            spec_rad,
            num_times,
            obs_noise,
            b=1,
        )
    myMLE = MLE(LR_samples)
    return myMLE.roc("MLE")


def roc_upbnd_genie_mle_zero_a(
    num_genes: int,
    prob_conn: float,
    scale_factor: float,
    num_times: int,
    num_LR_samples: int = 500,
    sigma_sq: float = 1.0,
    **kwargs,
) -> ROC:
    """Generate an ROC under conditions that network matrix is available to
    estimator for all other links using the method of generating LR samples
    and using the MLE estimator or ROC.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        scale_factor: magnitude of weights, used only if stationary = False.
        num_times: one plus number of observation times, starting at zero.
        num_LR_samples: number of LR sample produced, half with link 1->0 on.
        sigma_sq: variance in state equations.

    Returns an ROC object which is a curve that can be used by caller to plot.
    Does not include plt.figure(), plt.legend(), or plt.show().
    """
    LR_samples = np.zeros(num_LR_samples)
    for k in range(num_LR_samples // 2):
        LR_samples[k] = lr_sample_zero_a(
            num_genes, prob_conn, sigma_sq, scale_factor, num_times, b=1
        )
    for k in range(num_LR_samples // 2, num_LR_samples):
        LR_samples[k] = lr_sample_zero_a(
            num_genes, prob_conn, sigma_sq, scale_factor, num_times, b=0
        )
    myMLE = MLE(LR_samples)
    return myMLE.roc("MLE")


def bhatta_roc_bnd(BC_samples: np.ndarray) -> ROC:
    """Compute an ROC given an array of squared Bhattacharyya samples

    Args:
        BC_samples: An np.array of values in [0,1] (Bhattacharyya
            coefficients).  These values are assumed to be the BC2
            coefficients from component distributions when the components are
            randomly generated using the underlying mixture weights.

    Returns: An ROC which is an estimated upper bound on true ROC.
    """
    BC_ave = np.average(BC_samples)

    def bc_binary(p, q):
        return np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))

    pfa = np.zeros(101)
    pdet = np.zeros(101)
    for i in range(101):
        pfa[i] = 0.01 * i
        pdet[i] = pfa[i]
        while bc_binary(pdet[i], pfa[i]) > BC_ave and pdet[i] < 0.9999:
            pdet[i] += 0.0001
        i = i + 1
    return ROC(pfa, pdet, "Bhat ROC bnd")


def bhatta_coeff(cov_mat_0, cov_mat_1):
    """Bhattacharyya coefficient."""
    # Use np.linalg.slogdet to avoid overflow.
    logdet = [np.linalg.slogdet(cov_mat)[1] for cov_mat in [cov_mat_0, cov_mat_1]]
    logdet_avg = np.linalg.slogdet((cov_mat_0 + cov_mat_1) / 2)[1]
    return np.exp(sum(logdet) / 4 - logdet_avg / 2)


def bc_sample(
    num_genes: int,
    prob_conn: float,
    sigma_sq: float,
    stationary: bool,
    scale_factor: float,
    spec_rad: float,
    num_times: int,
    obs_noise: float = 0,
    b: int = 0,
) -> float:
    """Generate a BC2 sample for component distribution assuming genie.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        sigma_sq: variance in state equations.
        spec_rad: spectral radius.
        num_times: one plus number of observation times.
        obs_noise: variance of observation noise.
        b: activation of link 1->0 should be 0 or 1.

    Returns:
        Sample of BC squared.
    """
    if stationary:
        A0 = erdos_renyi_ternary(num_genes, prob_conn)
        A1 = A0.copy()
        A0[1][0] = 0.0
        A1[1][0] = 1.0
        A0, _ = scale_by_spec_rad(A0, spec_rad)
        A1, _ = scale_by_spec_rad(A1, spec_rad)
        Q0, _ = asymptotic_cov_mat(np.identity(num_genes), A0, sigma_sq, 20)
        Q1, _ = asymptotic_cov_mat(np.identity(num_genes), A1, sigma_sq, 20)
        Sigma0 = big_covar_matrix(A0, num_times, Q0, sigma_sq, obs_noise)
        Sigma1 = big_covar_matrix(A1, num_times, Q1, sigma_sq, obs_noise)
        return bhatta_coeff(Sigma0, Sigma1)

    else:
        A0 = erdos_renyi_ternary(num_genes, prob_conn)
        A1 = A0.copy()
        A0[1][0] = 0.0
        A1[1][0] = 1.0
        A0 = A0 * scale_factor
        A1 = A1 * scale_factor
        Q_null = np.zeros((num_genes, num_genes))
        Q0, _ = asymptotic_cov_mat(np.identity(num_genes), A0, sigma_sq, 20)
        Q1, _ = asymptotic_cov_mat(np.identity(num_genes), A1, sigma_sq, 20)
        Sigma0 = big_covar_matrix(A0, num_times, Q_null, sigma_sq, obs_noise)
        Sigma1 = big_covar_matrix(A1, num_times, Q_null, sigma_sq, obs_noise)
        return bhatta_coeff(
            Sigma0[num_genes:, num_genes:], Sigma1[num_genes:, num_genes:]
        )


def roc_upbnd_genie_bc(
    num_genes: int,
    prob_conn: float,
    stationary: bool,
    scale_factor: float,
    spec_rad: float,
    num_times: int,
    num_BC_samples: int = 500,
    obs_noise: float = 0,
    sigma_sq: float = 1.0,
    **kwargs,
) -> ROC:
    """Generate an ROC bound under conditions that network matrix is available
    to estimator for all other links using the method of generating BC2samples
    and using the stronger Bhattacharyya bound for mixture distributions.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        stationary: True if start in stationary distribution, False if start
            from 0.
        scale_factor: magnitude of weights, used only if stationary = False.
        spec_rad: spectral radius, used only if stationary = True.
        num_times: one plus number of observation times, starting at zero.
        num_BC_samples: number of BC2 sample produced, half with link 1->0 on.
        obs_noise: variance of observation noise.
        sigma_sq: variance in state equations.

    Returns an ROC curve that can be used by caller to plot. Does not include
    plt.figure(), plt.legend(), or plt.show().
    """

    BC_samples = np.zeros(num_BC_samples)
    for k in range(num_BC_samples // 2):
        BC_samples[k] = bc_sample(
            num_genes,
            prob_conn,
            sigma_sq,
            stationary,
            scale_factor,
            spec_rad,
            num_times,
            obs_noise,
            b=0,
        )
    for k in range(num_BC_samples // 2, num_BC_samples):
        BC_samples[k] = bc_sample(
            num_genes,
            prob_conn,
            sigma_sq,
            stationary,
            scale_factor,
            spec_rad,
            num_times,
            obs_noise,
            b=1,
        )
    return bhatta_roc_bnd(BC_samples)


def kl_roc_bnd(
    KL1_in: np.ndarray,
    KL0_in: np.ndarray,
) -> ROC:
    """Compute an ROC given one or more KL divergence values as a float or
    np.ndarray.

    Does not assume symmetry. KL1 should be d_KL(P_1 || P_0) values and KL0
    should be d_KL(P_0 || P_1).

    Args:
        KL1_in: Samples of KL divergence (each will be averaged).
        KL0_in: Samples of KL divergence (each will be averaged).

    Returns: An ROC which is an upper bound of true ROC.
    """

    def kl_binary(p: float, q: float) -> float:
        """Compute the KL divergence between Bernoulli(p) and Bernoulli(q)
        Args p and q assumed to be strictly between 0 and 1.
        """
        return (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)),)

    KL1 = np.average(KL1_in)
    KL0 = np.average(KL0_in)
    pdet = np.zeros(99)
    pfa = np.zeros(99)
    for i in range(99):
        pfa[i] = 0.01 * i + 0.001
        pdet[i] = pfa[i]
        while (
            kl_binary(pdet[i], pfa[i]) < KL1
            and kl_binary(pfa[i], pdet[i]) < KL0
            and pdet[i] < 0.9999
        ):
            pdet[i] += 0.0001
        i = i + 1
    return ROC(pfa, pdet, "KL ROC bnd")


def kl_sample(
    num_genes: int,
    prob_conn: float,
    sigma_sq: float,
    stationary: bool,
    scale_factor: float,
    spec_rad: float,
    num_times: int,
    obs_noise: float = 0,
) -> float:
    """Generate a sample of d_KL(P_1||P_0) and d_KL(P_1 || P_0) assuming genie.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        sigma_sq: variance in state equations.
        stationary: True if start in stationary distribution, False if start
            from 0.
        scale_factor: magnitude of weights, used only if stationary = False.
        spec_rad: spectral radius, used only if stationary = True.
        num_times: one plus number of observation times, starting at zero.
        obs_noise: variance of observation noise (assumed to be zero if
            stationary = false).

        Returns:
            A pair of KL distances.
    """

    def kl_divergence_mean_zero_gaussians(sigma1, sigma2) -> np.ndarray:
        """Computes the KL divergence between two mean-zero multivariate
        Gaussian distributions.

        Args:
            sigma1: Covariance matrix of the first Gaussian distribution.
            sigma2: Covariance matrix of the second Gaussian distribution.

        Returns:
            The KL divergence between the two Gaussian distributions.
        """
        if (
            sigma1.shape != sigma2.shape
            or sigma1.shape[0] != sigma1.shape[1]
            or sigma2.shape[0] != sigma2.shape[1]
        ):
            raise ValueError(
                "Input covariance matrices must be square and have the same dimensions."
            )

        # Check if covariance matrices are positive definite.
        try:
            np.linalg.cholesky(sigma1)
            np.linalg.cholesky(sigma2)
        except np.linalg.LinAlgError:
            raise ValueError("Input covariance matrices must be positive definite.")

        # Calculate the log determinant of the covariance matrices.
        log_det_sigma1 = np.log(np.linalg.det(sigma1))
        log_det_sigma2 = np.log(np.linalg.det(sigma2))

        # Calculate the trace of sigma1 * sigma2^-1.
        try:
            trace_term = np.trace(np.linalg.solve(sigma2, sigma1))
        except np.linalg.LinAlgError:
            raise ValueError("sigma2 must be invertible")

        # Calculate the KL divergence.
        kl_divergence = 0.5 * (
            log_det_sigma2 - log_det_sigma1 + trace_term - sigma1.shape[0]
        )

        return kl_divergence

    A0 = erdos_renyi_ternary(num_genes, prob_conn)
    A1 = A0.copy()
    A0[1][0] = 0.0
    A1[1][0] = 1.0  # By symmetry we are assuming R[1,0] = 1.
    if stationary:
        A0, _ = scale_by_spec_rad(A0, spec_rad)
        A1, _ = scale_by_spec_rad(A1, spec_rad)
        Q0, _ = asymptotic_cov_mat(np.identity(num_genes), A0, sigma_sq, 20)
        Q1, _ = asymptotic_cov_mat(np.identity(num_genes), A1, sigma_sq, 20)
        Sigma0 = big_covar_matrix(A0, num_times, Q0, sigma_sq, obs_noise)
        Sigma1 = big_covar_matrix(A1, num_times, Q1, sigma_sq, obs_noise)
        return np.array(
            [
                kl_divergence_mean_zero_gaussians(Sigma1, Sigma0),
                kl_divergence_mean_zero_gaussians(Sigma0, Sigma1),
            ]
        )

    else:  # stationary = False   Using direct method.
        A0 = A0 * scale_factor
        A1 = A1 * scale_factor
        SUM10 = 0.0  # 10 variables are for D_KL(P1||P0)  P1<->A1, P0<->A0.
        SUM01 = 0.0  # 01 variables are for D_KL(P1||P0).
        eye = np.identity(num_genes)
        Q_10 = np.zeros((num_genes, num_genes))
        Q_01 = np.zeros((num_genes, num_genes))
        for t in range(num_times - 1):
            # Cov(X[t+1]) under A1.
            Q_10 = np.dot(A1.T, np.dot(Q_10, A1)) + eye
            # Cov(X[t+1]) under A0.
            Q_01 = np.dot(A0.T, np.dot(Q_01, A0)) + eye
            SUM10 += Q_10[1, 1]
            SUM01 += Q_01[1, 1]
        return scale_factor * scale_factor * np.array([SUM10, SUM01]) / 2


def roc_upbnd_genie_kl(
    num_genes: int,
    prob_conn: float,
    stationary: bool,
    scale_factor: float,
    spec_rad: float,
    num_times: int,
    num_KL_samples: int = 500,
    obs_noise: float = 0,
    sigma_sq: float = 1.0,
    **kwargs,
) -> ROC:
    """Generate an ROC bound under conditions that network matrix is available
    to estimator for all other links using the method of generating KL samples
    and using the stronger KL bound for mixture distributions.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        stationary: True if start in stationary distribution, False if start
            from 0.
        scale_factor: magnitude of weights, used only if stationary = False.
        spec_rad: spectral radius, used only if stationary = True.
        num_times: one plus number of observation times, starting at zero.
        num_KL_samples: number of KL sample produced, half with link 1->0 on.
        obs_noise: variance of observation noise.
        sigma_sq: variance in state equations.

    Returns an ROC curve that can be used by caller to plot. Does not include
    plt.figure(), plt.legend(), or plt.show().
    """

    KL_sums = np.zeros(2)
    for k in range(num_KL_samples):
        KL_sums = KL_sums + kl_sample(
            num_genes,
            prob_conn,
            sigma_sq,
            stationary,
            scale_factor,
            spec_rad,
            num_times,
            obs_noise,
        )
    return kl_roc_bnd(KL_sums[0] / num_KL_samples, KL_sums[1] / num_KL_samples)


def kl_sample_zero_a(
    num_genes: int,
    prob_conn: float,
    sigma_sq: float,
    stationary: bool,
    scale_factor: float,
    spec_rad: float,
    num_times: int,
    obs_noise: float = 0,
) -> float:
    """Generate a sample of d_KL(P_1||P_0) and d_KL(P_1 || P_0) assuming
    genie.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        sigma_sq: variance in state equations.
        stationary: True if start in stationary distribution, False if start
            from 0.
        scale_factor: magnitude of weights, used only if stationary = False.
        spec_rad: spectral radius, used only if stationary = True.
        num_times: one plus number of observation times, starting at zero.
        obs_noise: variance of observation noise (assumed to be zero if
            stationary = false).

        Returns:
            A pair of KL distances.
    """

    def kl_divergence_mean_zero_gaussians(sigma1, sigma2) -> np.ndarray:
        """Computes the KL divergence between two mean-zero multivariate
        Gaussian distributions.

        Args:
            sigma1: Covariance matrix of the first Gaussian distribution.
            sigma2: Covariance matrix of the second Gaussian distribution.

        Returns:
            The KL divergence between the two Gaussian distributions.
        """
        # Check if covariance matrices are square and have the same
        # dimensions.
        if (
            sigma1.shape != sigma2.shape
            or sigma1.shape[0] != sigma1.shape[1]
            or sigma2.shape[0] != sigma2.shape[1]
        ):
            raise ValueError(
                "Input covariance matrices must be square and have the same dimensions."
            )

        # Check if covariance matrices are positive definite.
        try:
            np.linalg.cholesky(sigma1)
            np.linalg.cholesky(sigma2)
        except np.linalg.LinAlgError:
            raise ValueError("Input covariance matrices must be positive definite.")

        # Calculate the log determinant of the covariance matrices.
        log_det_sigma1 = np.log(np.linalg.det(sigma1))
        log_det_sigma2 = np.log(np.linalg.det(sigma2))

        # Calculate the trace of sigma1 * sigma2^-1.
        try:
            trace_term = np.trace(np.linalg.solve(sigma2, sigma1))
        except np.linalg.LinAlgError:
            raise ValueError("sigma2 must be invertible")

        # Calculate the KL divergence.
        kl_divergence = 0.5 * (
            log_det_sigma2 - log_det_sigma1 + trace_term - sigma1.shape[0]
        )

        return kl_divergence

    A0 = erdos_renyi_ternary(num_genes, prob_conn)
    A1 = A0.copy()
    A0[1][0] = 0.0
    A1[1][0] = 1.0  # By symmetry we are assuming R[1,0] = 1.
    if stationary:
        A0, _ = scale_by_spec_rad(A0, spec_rad)
        A1, _ = scale_by_spec_rad(A1, spec_rad)
        Q0, _ = asymptotic_cov_mat(np.identity(num_genes), A0, sigma_sq, 20)
        Q1, _ = asymptotic_cov_mat(np.identity(num_genes), A1, sigma_sq, 20)
        Sigma0 = big_covar_matrix(A0, num_times, Q0, sigma_sq, obs_noise)
        Sigma1 = big_covar_matrix(A1, num_times, Q1, sigma_sq, obs_noise)
        return np.array(
            [
                kl_divergence_mean_zero_gaussians(Sigma1, Sigma0),
                kl_divergence_mean_zero_gaussians(Sigma0, Sigma1),
            ]
        )

    else:  # stationary = False   Using direct method.
        A0 = A0 * scale_factor
        A1 = A1 * scale_factor
        Q_null = np.zeros((num_genes, num_genes))
        Sigma0 = big_covar_matrix(A0, num_times, Q_null, sigma_sq, obs_noise)
        Sigma1 = big_covar_matrix(A1, num_times, Q_null, sigma_sq, obs_noise)
        return np.array(
            [
                kl_divergence_mean_zero_gaussians(
                    Sigma1[num_genes:, num_genes:], Sigma0[num_genes:, num_genes:]
                ),
                kl_divergence_mean_zero_gaussians(
                    Sigma0[num_genes:, num_genes:], Sigma1[num_genes:, num_genes:]
                ),
            ]
        )


def roc_upbnd_genie_kl_zero_a(
    num_genes: int,
    prob_conn: float,
    stationary: bool,
    scale_factor: float,
    spec_rad: float,
    num_times: int,
    num_KL_samples: int = 500,
    obs_noise: float = 0,
    sigma_sq: float = 1.0,
    **kwargs,
) -> ROC:
    """Generate an ROC bound under conditions that network matrix is available
    to estimator for all other links using the method of generating KL samples
    and using the stronger KL bound for mixture distributions.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        sigma_sq: variance in state equations.
        spec_rad: spectral radius.
        T: duration (T+1 observations).
        num_BC2_samples: number of BC2 sample produced, half with link 1->0
            on.
        obs_noise: variance of observation noise.

    Returns an ROC curve that can be used by caller to plot. Does not include
    plt.figure(), plt.legend(), or plt.show()
    """

    if stationary:
        print("Warning -- ROC_upbnd_genieKL only works for stationary = False")
    KL_sums = np.zeros(2)
    for k in range(num_KL_samples):
        KL_sums = KL_sums + kl_sample_zero_a(
            num_genes,
            prob_conn,
            sigma_sq,
            stationary,
            scale_factor,
            spec_rad,
            num_times,
        )
    return kl_roc_bnd(KL_sums[0] / num_KL_samples, KL_sums[1] / num_KL_samples)


def script1():
    """Comparison of ROC_upbnd_genieROC, ROC_upbnd_genieBC(), ROC_upbnd_genieKL(),
    for stationary network version.

    Could use in paper.

    Shows ROC, BC, KL bounds for case of stationary = TRUE with observation
    noise.

    Returns nothing but produces a plot with ROC bounds for several spec_rad
    values.
    """

    num_genes = 10
    prob_conn = 0.2
    T = 20
    # Should increase to 1000 later -- takes 8 minutes.
    num_samples = config.config["roc_mlroc_vs_info_bnds"]["n_sims"]
    stationary = True  # Not fixed for stationary = False.
    scale_factor = 10  # Not used because stationary = True.
    spec_rad_list = [0.6, 0.3, 0.1]
    color = ("b", "orange", "g")
    for obs_noise in range(2):
        plt.figure()  # Needs to precede creation of labeled curves.

        for j in range(len(spec_rad_list)):

            ROC = roc_upbnd_genie_bc(
                num_genes,
                prob_conn,
                stationary,
                scale_factor,
                spec_rad_list[j],
                T,
                num_samples,
                obs_noise,
            )
            plt.plot(
                ROC.pfa,
                ROC.pdet,
                color[j],
                linestyle="-.",
                label="BC up bnd, $r_0$ =" + str(spec_rad_list[j]),
            )

            ROC = roc_upbnd_genie_kl(
                num_genes,
                prob_conn,
                stationary,
                scale_factor,
                spec_rad_list[j],
                T,
                num_samples,
                obs_noise,
            )
            plt.plot(
                ROC.pfa,
                ROC.pdet,
                color[j],
                linestyle="--",
                label="KL up bnd, $r_0$ =" + str(spec_rad_list[j]),
            )

            ROC = roc_upbnd_genie_mle(
                num_genes,
                prob_conn,
                stationary,
                scale_factor,
                spec_rad_list[j],
                T,
                num_samples,
                obs_noise,
            )
            plt.plot(
                ROC.pfa,
                ROC.pdet,
                color[j],
                linestyle="-",
                label="MLROC up bnd, $r_0$ =" + str(spec_rad_list[j]),
            )

        plt.legend()
        plt.xlabel("false positive rate", fontsize=16)
        plt.ylabel("true positive rate", fontsize=16)
        plt.savefig(
            config.config["paths"]["plots_dir"]
            + "bnd_compare-stationary_"
            + str(stationary)
            + "-Obs_noise_"
            + str(obs_noise)
            + "-num_genes_"
            + str(num_genes)
            + "-num_times_"
            + str(T)
            + "p_conn="
            + str(prob_conn)
            + "-num_samples_"
            + str(num_samples)
            + ".pdf",
            bbox_inches="tight",
        )


def script2():
    """Comparison of ROC_upbnd_genieROC and ROC_upbnd_genieKL() for zero
    initial condition.

    Could use in paper.

    Compares ROC_MLE with upper bounds implied by KL divergence for large
    networks (e.g. num_genes = 20, num_times = 200) Only works for stationary
    = False and no observation noise.

    Returns nothing but produces a plot with ROC bounds for several spec_rad
    values.
    """

    num_genes = 20
    prob_conn = 0.1
    T = 200
    # Should increase to 1000 later.  Takes 2 minutes.
    num_samples = config.config["roc_mlroc_vs_info_bnds"]["n_sims"]
    stationary = False  # Not fixed for stationary = False.
    # Not used because stationary = True.
    scale_factor_list = [0.15, 0.1, 0.05]
    spec_rad = 10  # Not used.
    obs_noise = 0
    plt.figure()  # Needs to precede creation of labeled curves.
    color = ("b", "orange", "g")

    for j in range(len(scale_factor_list)):
        ROC = roc_upbnd_genie_kl(
            num_genes,
            prob_conn,
            stationary,
            scale_factor_list[j],
            spec_rad,
            T,
            num_samples,
            obs_noise,
        )
        plt.plot(
            ROC.pfa,
            ROC.pdet,
            color[j],
            linestyle="-.",
            label="KL up bnd, $v_0$ =" + str(scale_factor_list[j]),
        )

        ROC = roc_upbnd_genie_mle(
            num_genes,
            prob_conn,
            stationary,
            scale_factor_list[j],
            spec_rad,
            T,
            num_samples,
            obs_noise,
        )
        plt.plot(
            ROC.pfa,
            ROC.pdet,
            color[j],
            linestyle="-",
            label="MLROC up bnd, $v_0$ =" + str(scale_factor_list[j]),
        )

    plt.legend()
    plt.xlabel("false positive rate", fontsize=16)
    plt.ylabel("true positive rate", fontsize=16)
    plt.savefig(
        config.config["paths"]["plots_dir"]
        + "bnd_compare-stationary_"
        + str(stationary)
        + "-Obs_noise_"
        + str(obs_noise)
        + "-num_genes_"
        + str(num_genes)
        + "-num_times_"
        + str(T)
        + "p_conn="
        + str(prob_conn)
        + "-num_samples_"
        + str(num_samples)
        + ".pdf",
        bbox_inches="tight",
    )


def script3_mleroc():
    """Comparing ROCs for zero case produced by plug in and roll out method.

    It is for the stationary =False -- so zero initial condition and no
    observation noise. We find very good agreement.   The innovation method is
    much faster and is the one used in ROC_upbnd_genieMLE if stationary =
    False.

    Not for paper but nice to know the plugin and innovation methods agree for
    ROC stationary = FALSE case.
    """
    num_genes = 10
    prob_conn = 0.2
    T = 20
    num_LR_samples = 1000  # Should increase to 1000 later.
    stationary = False
    scale_factor_list = [0.2, 0.35, 0.5]
    spec_rad = 10  # Not used because stationary = False.
    plt.figure()  # Needs to precede creation of labeled curves.

    for scale_factor in scale_factor_list:
        ROC = roc_upbnd_genie_mle(
            num_genes, prob_conn, stationary, scale_factor, spec_rad, T, num_LR_samples
        )
        plt.plot(
            ROC.pfa,
            ROC.pdet,
            linestyle="-.",
            label="upper bound rollout, $v_0$ =" + str(scale_factor),
        )

    for scale_factor in scale_factor_list:
        ROC = roc_upbnd_genie_mle_zero_a(
            num_genes, prob_conn, scale_factor, T, num_LR_samples
        )
        plt.plot(
            ROC.pfa,
            ROC.pdet,
            linestyle="-.",
            label="upper bound plugin, $v_0$ =" + str(scale_factor),
        )

    plt.legend()
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("Two way calculation of ROC upper bounds version zero")


def script3_kl():
    """Comparing ROCs based on KL bound for zero case produced by plug in and
    innovation method.

    It is for the stationary = False -- so zero initial condition and no
    observation noise. We find very good agreement.   The roll out method is
    much faster (but not as fast as the rollout method for LR samples) and is
    the one used in ROC_upbnd_genieBC if stationary = False.

    Won't use in paper but nice to know the plugin and rollout methods agree
    for KL stationary = FALSE case.
    """

    num_genes = 10
    prob_conn = 0.2
    T = 20
    num_samples = 1000  # Should increase to 1000 later.
    stationary = False  # Not fixed for stationary = False.
    scale_factor_list = [0.1, 0.2, 0.3]  # Not used because stationary = True.
    spec_rad = 10  # Not used.
    obs_noise = 0.0
    plt.figure()  # Needs to precede creation of labeled curves.
    color = ("b", "orange", "g")

    for j in range(len(scale_factor_list)):
        ROC = roc_upbnd_genie_kl_zero_a(
            num_genes,
            prob_conn,
            stationary,
            scale_factor_list[j],
            spec_rad,
            T,
            num_samples,
            obs_noise,
        )
        plt.plot(
            ROC.pfa,
            ROC.pdet,
            color[j],
            linestyle="-",
            label="KL up bnd plugin, $v_0$ =" + str(scale_factor_list[j]),
        )

        ROC = roc_upbnd_genie_kl(
            num_genes,
            prob_conn,
            stationary,
            scale_factor_list[j],
            spec_rad,
            T,
            num_samples,
            obs_noise,
        )
        plt.plot(
            ROC.pfa,
            ROC.pdet,
            color[j],
            linestyle="-.",
            label="KL up bnd direct, $v_0$ =" + str(scale_factor_list[j]),
        )

    plt.legend()
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title(r"OC upper bounds based on KL two ways,   obs_noise=" + str(obs_noise))
    plt.savefig(
        config.config["paths"]["plots_dir"]
        + "bnd_compare:stationary="
        + str(stationary)
        + "Obs_noise="
        + str(obs_noise)
        + "num_genes="
        + str(num_genes)
        + "p_conn="
        + str(prob_conn)
        + "num_times="
        + str(T)
        + "num_samples="
        + str(num_samples)
        + ".pdf"
    )


def main():
    plt.style.use("ggplot")
    script1()
    script2()


if __name__ == "__main__":
    main()
