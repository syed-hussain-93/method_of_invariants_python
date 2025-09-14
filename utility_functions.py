import sympy as sp
import numpy as np
from functools import reduce


def create_rotation_matrix(theta):
    """Create a 2D rotation matrix about the origin by an angle θ be denoted """

    # Rotation matrix
    r = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)],
        [sp.sin(theta), sp.cos(theta)]
    ])

    return r


def create_reflection_matrix(theta):
    """Create 2D reflection matrix about a line L through the origin which makes an angle θ with the x-axis"""

    # Rotation matrix
    r = sp.Matrix([
        [sp.cos(2 * theta), sp.sin(2 * theta)],
        [sp.sin(2 * theta), -sp.cos(2 * theta)]
    ])

    return r


def tensor(*reps):
    """
    Takes in multiple lists of representation matrices and returns
    a new list where each entry is the Kronecker product of the
    corresponding matrices from the input lists.

    Example:
    reps = [ [A1, A2, A3], [B1, B2, B3] ]
    returns: [kron(A1, B1), kron(A2, B2), kron(A3, B3)]
    """
    # Ensure all reps are the same length
    n = len(reps[0])
    if not all(len(r) == n for r in reps):
        raise ValueError("All representation lists must have equal length")

    result = []
    for elem in zip(*reps):
        elem: list = [e if isinstance(e, sp.MatrixBase) else sp.ImmutableMatrix([[e]]) for e in elem]
        kron_product = reduce(sp.KroneckerProduct, elem).doit()

        result.append(kron_product)

    return result


def direct_sum(*reps):
    """
    Takes in multiple lists of matrices (representations),
    and returns a list of block diagonal matrices formed from
    corresponding elements of each list.
    """
    # Ensure all reps are same length
    n = len(reps[0])
    if not all(len(r) == n for r in reps):
        raise ValueError("All representation lists must have equal length")

    result = []
    for elem in zip(*reps):
        elem: list = [e if isinstance(e, sp.MatrixBase) else sp.ImmutableMatrix([[e]]) for e in elem]
        block_sum = sp.Matrix(reduce(sp.BlockDiagMatrix, elem).doit())
        result.append(block_sum)
    return result


def character_of_representation(representation: list) -> list:
    return [sp.Trace(g).expand(complex=True).simplify() if isinstance(g, sp.MatrixBase) else g for g in representation]


def irrep_decomposition(representation: list, irreducible_representations: dict):
    chars_rep = character_of_representation(representation)
    ir_vec = {}
    for ir_name, ir in irreducible_representations.items():
        chars_ir = character_of_representation(ir)
        decomp = (sum([sp.conjugate(i) * j for i, j in zip(chars_ir, chars_rep)]) / len(representation)).simplify()
        if decomp != 0:
            ir_vec[ir_name] = decomp
    return ir_vec


def convert_rep_to_matrix_type(rep):
    rep = [R if isinstance(R, sp.MatrixBase) else sp.Matrix([R]) for R in rep]
    return rep


def projection_operator(irrep: list, representation: list, i: int, j: int):
    irrep = convert_rep_to_matrix_type(irrep)
    d_mu = irrep[0].shape[0]
    d_G = len(irrep)

    if d_mu == 1:
        terms = [sp.conjugate(irrep[R])[0] * representation[R] for R in range(d_G)]
        # P = sp.Rational(d_mu, d_G) * sum(terms, sp.Matrix.zeros(*representation[0].shape))
    else:
        terms = [sp.conjugate(irrep[R][i, j]) * representation[R] for R in range(d_G)]
    P = sp.Rational(d_mu, d_G) * sum(terms, sp.Matrix.zeros(*representation[0].shape)).expand(complex=True)
    return P


def is_faithful(rep: list[sp.MatrixBase]):
    for i, gi in enumerate(rep):
        for j, gj in enumerate(rep):
            if i != j and gi == gj:
                return False
    return True


def ismember(a: list | np.ndarray, b: list | np.ndarray):
    """
    Replicates MATLAB's ismember for 2D arrays using NumPy.

    Parameters
    ----------
    a : np.ndarray
        Array of elements to test.
    b : np.ndarray
        Array in which to look for elements of a.

    Returns
    -------
    rows : np.ndarray
        Row indices in `b` where elements of `a` are found.
    cols : np.ndarray
        Column indices in `b` where elements of `a` are found.
    mask : np.ndarray
        Boolean array, same shape as `a`, True if element is in `b`.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # flatten b for search
    b_flat = b.ravel()

    # searchsorted trick (works like MATLAB ismember)
    sorter = np.argsort(b_flat)
    idx = sorter[np.searchsorted(b_flat, a.ravel(), sorter=sorter)]

    # check membership
    mask = b_flat[idx] == a.ravel()

    # get row and col indices
    rows, cols = np.unravel_index(idx, b.shape)

    # mask non-members
    rows = np.where(mask, rows, -1)
    cols = np.where(mask, cols, -1)

    return rows.reshape(a.shape), cols.reshape(a.shape)
