import numpy as np
import scipy

import pyttb as ttb

def hosvd(
    input_tensor,
    ranks,
    verbosity: float = 1,
    dimorder = None,
    sequential: bool = True,
    
):
    """Compute sequentially-truncated higher-order SVD (Tucker).

    Computes a Tucker decomposition with relative error
    specified by ranks.

    Parameters
    ----------
    input_tensor: Tensor to factor
    ranks: Specify ranks to consider rather than computing
    verbosity: Print level
    dimorder: Order to loop through dimensions
    sequential: Use sequentially-truncated version
    

    Example
    -------
    >>> data = np.array([[29, 39.], [63., 85.]])
    >>> disable_printing = -1
    >>> tensorInstance = ttb.tensor().from_data(data)
    >>> result = hosvd(tensorInstance, verbosity=disable_printing)
    """
    # In tucker als this is N
    d = input_tensor.ndims

    
    if len(ranks) != d:
        raise ValueError(
            f"Ranks must be a list of length tensor ndims. Ndims: {d} but got "
            f"ranks: {ranks}."
        )

    if not dimorder:
        dimorder = list(range(d))
    else:
        if not isinstance(dimorder, list):
            raise ValueError("Dimorder must be a list")
        elif tuple(range(d)) != tuple(sorted(dimorder)):
            raise ValueError(
                "Dimorder must be a list or permutation of range(tensor.ndims)"
            )

    normxsqr = (input_tensor**2).collapse()

    # Main Loop
    factor_matrices = [np.empty(1)] * d
    # Copy input tensor, shrinks every step for sequential
    Y = ttb.tensor.from_tensor_type(input_tensor)

    for k in dimorder:
        # Compute Gram matrix
        Yk = ttb.tenmat.from_tensor_type(Y, np.array([k])).double()
        Z = np.dot(Yk, Yk.transpose())

        # Compute eigenvalue decomposition
        D, V = scipy.linalg.eigh(Z)
        pi = np.argsort(-D, kind="quicksort")
        eigvec = D[pi]

        # Extract factor matrix b picking leading eigenvectors of V
        # NOTE: Plus 1 in pi slice for inclusive range to match MATLAB
        factor_matrices[k] = V[:, pi[0 : ranks[k] + 1]]

        # Shrink!
        if sequential:
            Y = Y.ttm(factor_matrices[k].transpose(), k)
    # Extract final core
    if sequential:
        G = Y
    else:
        G = Y.ttm(factor_matrices, transpose=True)

    result = ttb.ttensor.from_data(G, factor_matrices)

    diffnormsqr = ((input_tensor - result.full()) ** 2).collapse()
    relnorm = np.sqrt(diffnormsqr / normxsqr)
        
    return result, relnorm
