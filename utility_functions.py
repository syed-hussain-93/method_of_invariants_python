import sympy as sp

from representation_matrices import BaseGroup


def create_rotation_matrix(n):
    """Create a 2D rotation matrix about the origin by an angle θ be denoted """

    # Create 2D representation
    theta = 2 * sp.pi / n

    # Rotation matrix
    r = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)],
        [sp.sin(theta), sp.cos(theta)]
    ])

    return r


def create_reflection_matrix(n):
    """Create 2D reflection matrix about a line L through the origin which makes an angle θ with the x-axis"""

    # Create 2D representation
    theta = 2 * sp.pi / n

    # Rotation matrix
    r = sp.Matrix([
        [sp.cos(2 * theta), sp.sin(2 * theta)],
        [sp.sin(2 * theta), -sp.cos(2 * theta)]
    ])

    return r
