"""Plots ROC and ROC bounds based on BC and KL divergence for
the case of detecting (signal plus noise, signal) pairs vs
(noise, signal) pairs.   The curve depends only on the distribution
of the norm of the signal.   For one example the signal has a constant
norm (so it becomes a binormal detection problem.)  In the other the signal
is a two state Markov chain.    Used for two figures in appendix of paper.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import Config
from mleroc.estimators import MLE
from mleroc.roc import ROC
from scipy.stats import norm

config = Config(
    "config-net-inf-eval.toml",
    template_path=Path(__file__).parent / "config.toml.template",
)


def binormal_roc(mu: float) -> ROC:
    """Generate ROC for binormal scalars same variance mean difference mu"""
    fpr = np.zeros(1000)
    tpr = np.zeros(1000)
    for i in range(1000):
        gamma = -5 + i / 100.0
        fpr[i] = 1 - norm.cdf(gamma)
        tpr[i] = 1 - norm.cdf(gamma - mu)
    return ROC(fpr, tpr, "binormal ROC")


def bhatta_roc_bnd(BC_samples: np.ndarray) -> ROC:
    """Compute an ROC given an array of squared Bhattacharyya samples

     Args: An np.array of values in [0,1] (Bhattacharyya coefficients)
       These values are assumed to be the BC2 coefficients from component
       distributions when the components are randomly generated using the
       underlying mixture weights.

    Returns: An ROC which is an estimated upper bound on true ROC
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


def kl_binary(p: float, q: float) -> float:
    """Compute the larger of the KL divergence between Bernoulli(p) and Bernoulli(q) and
    the divergence between Bernoulli(q) and Bernoulli(p).
    Args p and q assumed to be strictly between 0 and 1
    """
    return max(
        p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)),
        q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p)),
    )


def kl_roc_bnd1(KL_in: float) -> ROC:
    """Compute an ROC given ONE KL divergence value.   Assumes symmetry -- uses
        the maxiumum of the two binary divergences.

    Args: KL_in (Bhattacharyya coefficient)

    Returns: An ROC which is an upper bound of true ROC
    """
    pdet = np.zeros(99)
    pfa = np.zeros(99)
    for i in range(99):
        pfa[i] = 0.01 * i + 0.001
        pdet[i] = pfa[i]
        while kl_binary(pdet[i], pfa[i]) < KL_in and pdet[i] < 0.9999:
            pdet[i] += 0.0001
        i = i + 1
    return ROC(pfa, pdet, "KL ROC bnd")


def simulate_markov_norm2(p, A: float, length: int, p_0=0.5) -> float:
    """
    Simulates a two-state discrete time Markov process with states 0 and 1.

    The initial state is A with probability p_0
    The transition probability matrix is:
         [ [1-p,    p],
           [  p,  1-p] ]

    Args:
        p (float): Crossover probability of transitioning to the other state.
        A: the "on" state or maximum signal amplitute
        length (int): Number of transitions to simulate, i.e. signal duration


    Returns:
        L2 norm squared of the random sequence.
    """
    if random.random() < p_0:
        state = 0
        sum = 0.0
    else:
        state = 1
        sum = 1.0
    for _ in range(length - 1):
        r = random.random()  # Generate a random number between 0 and 1
        # Update state
        if state == 0:
            state = 1 if r < p else 0
        else:  # current_state == 1
            state = 0 if r < p else 1
        if state == 1:
            sum = sum + 1
    return sum * A * A


def markov_many_trials(p, A, n) -> tuple[np.array, float, float]:
    """Returns an array of LR samples, estimate of BC, estimate of KL divergence
    Note: It returns the estimate of BC -- not an estimate of BC^2.  So it

    Args:
          p (float): Crossover probability of transitioning to the other state.
          A
          num_steps (int): Number of transitions to simulate.

    """

    num_trials = 10000
    BC_sum = 0.0
    KL_sum = 0.0
    LR_samples = np.zeros(num_trials)
    for i in range(num_trials):
        S2 = simulate_markov_norm2(p, A, n)
        Z = norm.rvs()
        # print("S2=", S2, " Z=", Z, "np.sqrt(S2)*Z - S2/2 =", np.sqrt(S2)*Z - S2/2)
        if random.random() < 0.5:  # Generate sample for H_0
            LR_samples[i] = np.exp(np.sqrt(S2) * Z - S2 / 2)
        else:  # Generate sample for H_0
            LR_samples[i] = np.exp(np.sqrt(S2) * Z + S2 / 2)
        BC_sum += np.exp(-S2 / 8)
        KL_sum += S2 / 2

    myMLE = MLE(LR_samples)
    return (myMLE.roc("Markov signal"), BC_sum / num_trials, KL_sum / num_trials)


def script1():
    """Produces ROC for binormal, the KL bound and the BC bound"""

    plt.figure()  # Needs to precede creation of labeled curves
    plt.grid(True)
    mu = np.array([2.0, 1.0, 0.5])
    color = ["b", "orange", "g"]
    for j in range(3):
        ROC_binorm = binormal_roc(mu[j])
        ROC_KL = kl_roc_bnd1(0.5 * mu[j] * mu[j])
        ROC_BC = bhatta_roc_bnd(np.exp(-mu[j] * mu[j] / 8))

        plt.plot(
            ROC_binorm.pfa,
            ROC_binorm.pdet,
            color[j],
            linestyle="-",
            label=r"$\mu$=" + str(mu[j]) + " ROC",
        )
        plt.plot(
            ROC_KL.pfa,
            ROC_KL.pdet,
            color[j],
            linestyle="-.",
            label=r"$\mu$=" + str(mu[j]) + " KL bound",
        )
        plt.plot(
            ROC_BC.pfa,
            ROC_BC.pdet,
            color[j],
            linestyle="--",
            label=r"$\mu$=" + str(mu[j]) + " BC bound",
        )

    plt.legend()
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.savefig(config.config["paths"]["plots_dir"] + "roc_binormal.pdf")


def script2():
    """Produces ROC for Markov, the KL bound and the BC bound"""

    plt.figure()  # Needs to precede creation of labeled curves
    plt.grid(True)
    A = np.array([1.0, 0.8, 0.4])
    n = 10
    p = np.array([0.5, 0.1, 0.001])
    color = ["b", "orange", "g"]
    for j in range(3):
        ROC_Markov, BC, KL = markov_many_trials(p[j], A[j], n)
        print("p=", p[j], "A=", A[j], "BC =", BC, "KL =", KL)
        ROC_BC = bhatta_roc_bnd(BC)
        ROC_KL = kl_roc_bnd1(KL)

        plt.plot(
            ROC_Markov.pfa,
            ROC_Markov.pdet,
            color[j],
            linestyle="-",
            label=r"$A$=" + str(A[j]) + " $p$=" + str(p[j]) + " ROC",
        )
        plt.plot(
            ROC_KL.pfa,
            ROC_KL.pdet,
            color[j],
            linestyle="-.",
            label=r"$A$=" + str(A[j]) + " $p$=" + str(p[j]) + " KL bound",
        )
        plt.plot(
            ROC_BC.pfa,
            ROC_BC.pdet,
            color[j],
            linestyle="--",
            label=r"$A$=" + str(A[j]) + " $p$=" + str(p[j]) + " BC bound",
        )

    plt.legend()
    plt.xlabel("probability of false alarm")
    plt.ylabel("probability of detection")
    plt.savefig(config.config["paths"]["plots_dir"] + "roc_markov.pdf")


def main():
    plt.style.use("ggplot")
    # Enable LaTeX rendering (optional, requires LaTeX installation).
    plt.rcParams["text.usetex"] = True
    script1()
    script2()


if __name__ == "__main__":
    main()
