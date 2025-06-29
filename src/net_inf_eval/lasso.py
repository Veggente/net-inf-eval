"""The lasso algorithm."""

import numpy as np
from sklearn.linear_model import Lasso


def lasso_grn(
    data_cell: list[np.ndarray], alpha: float
) -> tuple[list[list[int]], list[list[int]]]:
    """Lasso for GRN.

    Args:
        data_cell: A list of gene expression matrices.  The ith
            matrix is T_i-by-n where T_i is the number of sample
            times in experiment i, and n is the number of genes.
        alpha: l1 regularizer coefficient.

    Returns:
        Parents and signs.
    """
    sensing_mat = np.concatenate([traj[:-1, :] for traj in data_cell])
    num_genes = data_cell[0].shape[1]
    parents = []
    signs = []
    for i in range(num_genes):
        reg = Lasso(alpha=alpha)
        target = np.concatenate([traj[1:, i] for traj in data_cell])
        reg.fit(sensing_mat, target)
        parents.append([idx for idx, val in enumerate(reg.coef_) if val])
        signs.append([int(np.sign(val)) for val in reg.coef_ if val])
    return parents, signs
