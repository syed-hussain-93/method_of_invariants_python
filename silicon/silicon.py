import sympy as sp

from base_connectivity import BaseConnectivity
from representation_matrices_explicit import (
    GroupTdO, CosetReps, GroupOfK, GroupD4C4vD2d, GroupD3C3v, GroupD2C2v, GroupOh, GroupD4h, GroupD3d
)
from lattice import DiamondCubicLattice
import numpy as np


class SiliconConnectivity(BaseConnectivity):

    def __init__(self, with_spin):
        super().__init__()
        self.with_spin = with_spin
        if self.with_spin:
            is_double_group = True
        else:
            is_double_group = False

        self.group = GroupOh(is_double_group=is_double_group)
        self.lattice = DiamondCubicLattice()

        group_G = GroupOh(is_double_group=is_double_group)
        group_X = GroupD4h(is_double_group=is_double_group)
        group_L = GroupD3d(is_double_group=is_double_group)
        group_W = GroupD4C4vD2d(name='D2d', is_double_group=is_double_group)

        self.group_of_G = GroupOfK(name='G',
                                   point=self.lattice.get_reciprocal_vector([0, 0, 0]),
                                   group=group_G)
        self.group_of_X = GroupOfK(name='X',
                                   point=self.lattice.get_reciprocal_vector([0, sp.Rational(1, 2), sp.Rational(1, 2)]),
                                   group=group_X)
        self.group_of_L = GroupOfK(name='L',
                                   point=self.lattice.get_reciprocal_vector(
                                       [sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 2)]),
                                   group=group_L)
        self.group_of_W = GroupOfK(name='W',
                                   point=self.lattice.get_reciprocal_vector(
                                       [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4)]),
                                   group=group_W)
        # self.group_of_K = GroupOfK(name='K',
        #                            point=self.lattice.get_reciprocal_vector(
        #                                [sp.Rational(3, 8), sp.Rational(3, 8), sp.Rational(3, 4)]),
        #                            group=GroupD3d(is_double_group=is_double_group))
        # self.group_of_U = GroupOfK(name='U',
        #                            point=self.lattice.get_reciprocal_vector(
        #                                [sp.Rational(1, 4), sp.Rational(5, 8), sp.Rational(5, 8)]),
        #                            group=GroupD2C2v(name='C2v', is_double_group=is_double_group))

        # self._site_symmetry_group = None
        self.site_symmetry_group = None
        self.coset_reps_dict = None
        self.tau_vectors = None
        self.coset_reps_ssg = None

    def run_setup(self, wyckoff_position: str):
        if wyckoff_position == '8a':
            self.site_symmetry_group = GroupTdO(name='Td',
                                                is_double_group=self.with_spin,
                                                )

            self.coset_reps_ssg = CosetReps(symbols=['E', 'I'], full_group=self.group)

            self.tau_vectors = {
                0: self.lattice.get_lattice_vector([0, 0, 0]),
                1: self.lattice.get_lattice_vector([sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 4)])
            }

        else:
            raise ValueError('No setup for this wyckoff posistion')

    def generate_band_representation(self, orbital_irrep_name: str, group_of_k: GroupOfK, zak_basis: bool = True):

        def calculate_phase(R, t_bra_idx):
            gR = np.array(self.group.get_xyz_representation())[self.subgroup_indices_in_full_group(group_of_k.group)][
                     R] * group_of_k.point - group_of_k.point
            tau_bra = self.tau_vectors[t_bra_idx]
            phase = sp.exp(sp.I * (gR.dot(tau_bra)))
            return phase

        def calculate_zak_phase(k_point, tau_bra_idx, tau_ket_idx):
            tau_ket = self.tau_vectors[tau_ket_idx]
            tau_bra = self.tau_vectors[tau_bra_idx]
            phase = sp.exp(sp.I * (k_point.dot(tau_bra - tau_ket)))
            return phase

        ssg_coset_mapping, ssg_element_mapping = self.group_to_subgroup_mapping(
            self.site_symmetry_group,
            coset_reps=self.coset_reps_ssg,
            from_subgroup=group_of_k.group
        )

        d_G = ssg_element_mapping.shape[1]
        m_tau = ssg_element_mapping.shape[0]  # ssg_coset_mapping.shape[0],
        orbital_irrep = self.group.irreducible_representations[orbital_irrep_name]
        d_mu = orbital_irrep[0].shape[0] if isinstance(orbital_irrep[0], sp.MatrixBase) or isinstance(orbital_irrep[0],
                                                                                                      np.ndarray) else 1

        d_H = m_tau * d_mu

        site_orbital_rep = []
        for Rk in range(d_G):
            H = sp.MutableDenseMatrix.zeros(d_H, d_H)
            for t_bra in range(m_tau):
                t_ket = ssg_coset_mapping[t_bra, Rk]
                element_in_ssg = ssg_element_mapping[t_bra, Rk]
                orbital_rep = orbital_irrep[element_in_ssg] * calculate_phase(Rk, t_bra)
                if zak_basis:
                    zak_phase = calculate_zak_phase(group_of_k.point, tau_bra_idx=t_bra, tau_ket_idx=t_ket)
                    orbital_rep *= zak_phase
                orbital_rep = orbital_rep if isinstance(orbital_rep, sp.MatrixBase) else sp.Matrix([orbital_rep])

                self.set_block(H, t_bra, t_ket, orbital_rep, orbital_rep.shape[0])

            site_orbital_rep.append(H)

        return site_orbital_rep
