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


def rotation_matrix_3D(axis, theta):
    """
    Create a 3D rotation matrix for rotation about an arbitrary axis.

    Parameters
    ----------
    axis : list or tuple of length 3
        Axis of rotation [x, y, z].
    theta : sympy expression or symbol
        Rotation angle.

    Returns
    -------
    sp.Matrix
        3x3 rotation matrix.
    """
    # Ensure axis is sympy Matrix
    axis = sp.Matrix(axis).normalized()
    x, y, z = axis

    cos_t = sp.cos(theta)
    sin_t = sp.sin(theta)
    one_minus_cos = 1 - cos_t

    R = sp.Matrix([
        [cos_t + x ** 2 * one_minus_cos, x * y * one_minus_cos - z * sin_t, x * z * one_minus_cos + y * sin_t],
        [y * x * one_minus_cos + z * sin_t, cos_t + y ** 2 * one_minus_cos, y * z * one_minus_cos - x * sin_t],
        [z * x * one_minus_cos - y * sin_t, z * y * one_minus_cos + x * sin_t, cos_t + z ** 2 * one_minus_cos]
    ])
    return sp.cancel(R.expand(complex=False))


# def tensor(*reps):
#     """
#     Takes in multiple lists of representation matrices and returns
#     a new list where each entry is the Kronecker product of the
#     corresponding matrices from the input lists.
#
#     If any element is 1D (scalar or 1x1 matrix), it is multiplied
#     instead of taking the Kronecker product.
#     """
#     # Ensure all reps are the same length
#     n = len(reps[0])
#     if not all(len(r) == n for r in reps):
#         raise ValueError("All representation lists must have equal length")
#
#     result = []
#     for elem in zip(*reps):
#         processed = []
#         scalar_factor = 1
#
#         for e in elem:
#             if isinstance(e, sp.MatrixBase):
#                 if e.shape == (1, 1):  # treat 1x1 matrix as scalar
#                     scalar_factor *= e[0, 0]
#                 else:
#                     processed.append(e)
#             else:
#                 # plain scalar§
#                 scalar_factor = scalar_factor * sp.sympify(e)
#
#         if not processed:  # all were scalars
#             kron_product = scalar_factor
#         else:
#             kron_product = reduce(lambda a, b: sp.KroneckerProduct(a, b).doit(), processed)
#             kron_product = scalar_factor * kron_product
#
#         result.append(kron_product)
#     result = [sp.cancel(R.expand(complex=True)) for R in result]
#     return result


def tensor(*reps):
    """
    Takes in multiple lists of representation matrices and returns
    a new list where each entry is the Kronecker product of the
    corresponding matrices from the input lists.

    If any element is 1D (scalar or 1x1 matrix), it is multiplied
    instead of taking the Kronecker product.
    """
    n = len(reps[0])
    if not all(len(r) == n for r in reps):
        raise ValueError("All representation lists must have equal length")

    result = []
    for elem in zip(*reps):
        scalar_factor = sp.Integer(1)
        matrices = []

        for e in elem:
            if isinstance(e, sp.MatrixBase):
                if e.shape == (1, 1):  # scalar in matrix form
                    scalar_factor *= e[0, 0]
                else:
                    matrices.append(e)
            else:
                scalar_factor *= sp.sympify(e)

        if not matrices:  # only scalars
            kron_product = scalar_factor
        else:
            kron_product = reduce(sp.kronecker_product, matrices)
            if scalar_factor != 1:
                kron_product = scalar_factor * kron_product

        result.append(sp.cancel(kron_product.expand(complex=True)))

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
    chars = []
    for g in representation:
        if isinstance(g, sp.MatrixBase):
            # direct trace is already simplified enough in most cases
            tr = sp.Trace(g).doit()
            # lightweight simplification
            chars.append(sp.simplify(tr))
        else:
            chars.append(g)
    return chars


def irrep_decomposition(representation: list, irreducible_representations: dict):
    n = len(representation)
    chars_rep = sp.Matrix(character_of_representation(representation))

    # Precompute irreps characters
    chars_irreps = {
        name: sp.Matrix(character_of_representation(ir))
        for name, ir in irreducible_representations.items()
    }

    ir_vec = {}
    for ir_name, chars_ir in chars_irreps.items():
        # inner product ⟨χ_ir, χ_rep⟩
        decomp = (chars_ir.conjugate().dot(chars_rep) / n)
        if not decomp.is_zero:
            ir_vec[ir_name] = sp.cancel(decomp)

    return ir_vec


def convert_rep_to_matrix_type(rep):
    rep = [R if isinstance(R, sp.MatrixBase) else sp.Matrix([R]) for R in rep]
    return rep


# def projection_operator(irrep: list, representation: list, i: int, j: int):
#     irrep = convert_rep_to_matrix_type(irrep)
#     d_mu = irrep[0].shape[0]
#     d_G = len(irrep)
#
#     if d_mu == 1:
#         terms = [sp.conjugate(irrep[R])[0] * representation[R] for R in range(d_G)]
#         # P = sp.Rational(d_mu, d_G) * sum(terms, sp.Matrix.zeros(*representation[0].shape))
#     else:
#         terms = [sp.conjugate(irrep[R][i, j]) * representation[R] for R in range(d_G)]
#     P = sp.Rational(d_mu, d_G) * sum(terms, sp.Matrix.zeros(*representation[0].shape)).expand(complex=True)
#     return P

def projection_operator(irrep: list, representation: list, i: int, j: int):
    irrep = convert_rep_to_matrix_type(irrep)
    d_mu = irrep[0].shape[0]
    d_G = len(irrep)

    P = sp.zeros(*representation[0].shape)  # direct accumulator

    if d_mu == 1:
        for R in range(d_G):
            P += sp.conjugate(irrep[R][0, 0]) * representation[R]
    else:
        for R in range(d_G):
            P += sp.conjugate(irrep[R][i, j]) * representation[R]

    return sp.Rational(d_mu, d_G) * P


# def is_faithful(rep: list[sp.MatrixBase]):
#     for i, gi in enumerate(rep):
#         for j, gj in enumerate(rep):
#             if i != j and gi == gj:
#                 return False
#     return True

def is_faithful(rep: list[sp.MatrixBase]) -> bool:
    """
    Check if a representation is faithful by verifying
    that all matrices are unique (injective mapping).
    """
    # Convert to immutable form so SymPy can compare structurally
    unique_mats = set(map(sp.ImmutableMatrix, rep))
    return len(unique_mats) == len(rep)


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


def match_subgroup_to_group(full_group, subgroup):
    """
    Find mapping from subgroup elements to full group elements.
    Handles complex matrices properly.

    Args:
        full_group: Group object with .get_faithful_representation() method
        subgroup: Subgroup object with .get_faithful_representation() method

    Returns:
        dict: mapping[subgroup_index] = full_group_index
    """
    full_group_matrices = full_group.faithful_representation
    subgroup_matrices = subgroup.faithful_representation

    mapping = {}

    for i, subgroup_mat in enumerate(subgroup_matrices):
        # Create signature for subgroup matrix - handle complex elements
        subgroup_trace = sp.expand(sp.simplify(subgroup_mat.trace()))
        subgroup_det = sp.expand(sp.simplify(subgroup_mat.det()))

        # Additional invariant: sum of squares of all matrix elements
        subgroup_frobenius_sq = sp.expand(sp.simplify(sum(element.conjugate() * element
                                                          for element in subgroup_mat)))

        # Find matching full group matrix
        for j, full_group_mat in enumerate(full_group_matrices):
            if j in mapping.values():  # Skip already matched
                continue

            full_group_trace = sp.expand(sp.simplify(full_group_mat.trace()))
            full_group_det = sp.expand(sp.simplify(full_group_mat.det()))
            full_group_frobenius_sq = sp.expand(sp.simplify(sum(element.conjugate() * element
                                                                for element in full_group_mat)))

            # Check if invariants match (handling complex expressions)
            if (_complex_equal(subgroup_trace, full_group_trace) and
                    _complex_equal(subgroup_det, full_group_det) and
                    _complex_equal(subgroup_frobenius_sq, full_group_frobenius_sq)):
                mapping[i] = j
                break

    if len(mapping) != len(subgroup_matrices):
        raise ValueError(f"Could only match {len(mapping)}/{len(subgroup_matrices)} elements")

    return mapping


def _complex_equal(expr1, expr2, tolerance=1e-12):
    """
    Check if two potentially complex SymPy expressions are equal.
    """
    diff = sp.expand(sp.simplify(expr1 - expr2))

    # If difference is zero, they're equal
    if diff == 0:
        return True

    # For complex expressions, check both real and imaginary parts
    try:
        real_diff = sp.expand(sp.simplify(sp.re(diff)))
        imag_diff = sp.expand(sp.simplify(sp.im(diff)))

        return real_diff == 0 and imag_diff == 0
    except:
        # Fallback: try direct simplification
        return sp.simplify(diff) == 0
