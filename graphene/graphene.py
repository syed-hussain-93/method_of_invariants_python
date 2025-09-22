from representation_matrices_explicit import (
    GroupD6h, GroupD6D3hC6v, BaseGroup, GroupD2h, CosetReps, GroupOfK, Generators, SiteSymmetryGroup
)
import sympy as sp
from utility_functions import ismember, projection_operator, tensor
import numpy as np
from lattice import Hexagonal2DLattice

from base_connectivity import BaseConnectivity


class GrapheneHamiltonian(BaseConnectivity):

    def __init__(self, with_spin: bool):
        super().__init__()
        self.with_spin = with_spin
        if self.with_spin:
            self.is_double_group = True
        else:
            self.is_double_group = False

        self.group = GroupD6h(
            generators=[
                Generators(axis=(0, 0, 1), angle=2 * sp.pi / 6, name='C6'),
                Generators(axis=(0, 1, 0), angle=2 * sp.pi / 2, name='C2'),
                Generators(matrix=-sp.Matrix.eye(3), name='I')
            ],
            is_double_group=self.is_double_group
        )
        self.lattice = Hexagonal2DLattice()

        self.group_of_G = GroupOfK(
            name='G',
            point=self.lattice.get_reciprocal_vector([0, 0, 0]),
            group=self.group
        )
        self.group_of_K = GroupOfK(
            name='K',
            point=self.lattice.get_reciprocal_vector([sp.Rational(1, 3), sp.Rational(1, 3), 0]),
            group=GroupD6D3hC6v(
                name='D3h',
                generators=[
                    Generators(axis=(0, 0, 1), angle=2 * sp.pi / 6, name='C6'),
                    Generators(axis=(1, 0, 0), angle=2 * sp.pi / 2, name='C2')
                ],
                is_double_group=self.is_double_group
            )
        )
        self.group_of_M = GroupOfK(
            name='M',
            point=self.lattice.get_reciprocal_vector([-sp.Rational(1, 2), sp.Rational(1, 2), 0]),
            group=GroupD2h(
                generators=[
                    Generators(axis=(0, 0, 1), angle=2 * sp.pi / 2, name='C2'),
                    Generators(axis=(0, 1, 0), angle=2 * sp.pi / 2, name='C2'),
                    Generators(matrix=-sp.Matrix.eye(3), name='I')
                ],
                is_double_group=self.is_double_group
            )
        )

        # self._site_symmetry_group = None
        self.site_symmetry_group = None
        self.coset_reps_dict = None
        self.tau_vectors = None
        self.coset_reps_ssg = None

    # @property
    # def site_symmetry_group(self):
    #     return GroupD6D3hC6v(name='d3h', is_double_group=self.with_spin, edge_x_orientation=False)
    #
    # @site_symmetry_group.setter
    # def site_symmetry_group(self, value):
    #     self._site_symmetry_group = value

    def run_setup(self, wyckoff_position: str):
        if wyckoff_position == '2b':

            site_symmetry_group = GroupD6D3hC6v(
                name='D3h',
                generators=[
                    Generators(axis=(0, 0, 1), angle=2 * sp.pi / 6, name='C6'),
                    Generators(axis=(0, 1, 0), angle=2 * sp.pi / 2, name='C2')
                ],
                is_double_group=self.is_double_group
            )

            coset_reps_ssg = CosetReps(symbols=['E', 'I'], full_group=self.group)

            tau_vectors = {
                0: sp.Rational(1, 3) * self.lattice.a1 + sp.Rational(2, 3) * self.lattice.a2,
                1: sp.Rational(2, 3) * self.lattice.a1 + sp.Rational(1, 3) * self.lattice.a2
            }
            self.site_symmetry_group = SiteSymmetryGroup(
                wyckoff_position=wyckoff_position,
                group=site_symmetry_group,
                coset_reps=coset_reps_ssg,
                tau_vectors=tau_vectors
            )

        else:
            raise ValueError('No setup for this wyckoff posistion')

    def generate_shell_vectors(self, shell_number: int, t_bra: int, t_ket: int):
        """
        t_bra/t_ket defines between what two wyckoff positions we want the shell vector for

        pass as integer which corresponds to the coset rep wrt ssg
        """

        xyz_full_group = self.group.get_xyz_representation()

        def unique_matrices(mats):
            # convert each Matrix to a tuple of its entries
            seen = set()
            uniq = []
            for m in mats:
                key = tuple(sp.simplify(x) for x in m)  # ensures (x, y, z) tuple
                if key not in seen:
                    seen.add(key)
                    uniq.append(m)
            return uniq

        if shell_number == 0:
            shell_n = [0, 0, 0]
            tau_ket_loc = self.tau_vectors[t_ket]
            tau_bra_loc = self.tau_vectors[t_ket]

        elif shell_number == 1:
            shell_n = [0, 0, 0]
            tau_ket_loc = self.tau_vectors[t_ket]
            tau_bra_loc = self.tau_vectors[t_bra]

        elif shell_number == 2:
            shell_n = [1, 0, 0]
            tau_ket_loc = self.tau_vectors[t_ket]
            tau_bra_loc = self.tau_vectors[t_ket]
        else:
            raise NotImplementedError('shell number not implemented')

        rho = tau_ket_loc - tau_bra_loc + self.lattice.get_lattice_vector(n=shell_n)

        R_on_rho = [R_xyz * rho for R_xyz in xyz_full_group]
        shell_vectors = unique_matrices(R_on_rho)

        return shell_vectors

    def permutation_representation(self, p):
        P = sp.zeros(len(p))
        for i, j in enumerate(p):
            P[j, i] = 1
        return P

    def generate_shell_representation(self, shell_number: int):
        if shell_number == 0:
            t_bra, t_ket = 0, 0
        elif shell_number == 1:
            t_bra, t_ket = 0, 1
        elif shell_number == 2:
            t_bra, t_ket = 0, 0
        else:
            raise NotImplementedError('shell not implemented')

        xyz_full_group = self.group.get_xyz_representation()

        shell_vectors = self.generate_shell_vectors(shell_number, t_bra, t_ket)
        permutations = np.array(
            [[shell_vectors.index(R_xyz * rho) for R_xyz in xyz_full_group] for rho in shell_vectors]).T
        permutation_rep = [self.permutation_representation(p) for p in permutations]

        return permutation_rep

    def reshape_hamiltonian(self, trial_vector, d_rho, d_bra, d_ket):
        blocks = []
        for i in range(d_rho):
            start = i * d_bra * d_ket
            end = (i + 1) * d_bra * d_ket
            block_vec = trial_vector[start:end, :]
            block_mat = block_vec.reshape(d_bra, d_ket)
            blocks.append(block_mat)
        return blocks

    def project_invariant_hamiltonian(self, projection_operator, trial_vector, d_rho, d_bra, d_ket):

        projected = (projection_operator * trial_vector).expand()

        return self.reshape_hamiltonian(projected, d_rho, d_bra, d_ket)

    def projection_operator_hamiltonian(self, shell_representation, site_orbital_representation_bra,
                                        site_orbital_representation_ket, trial_vector=None):

        from utility_functions import tensor
        d_rho = shell_representation[0].shape[0]
        bra_orb = [R.conjugate() for R in site_orbital_representation_bra]
        ket_orb = site_orbital_representation_ket

        projection_T = tensor(shell_representation, bra_orb, ket_orb)

        projection_T = sp.Rational(1, len(ket_orb) / d_rho) * sum(projection_T, sp.Matrix.zeros(*projection_T[0].shape))

        return projection_T

    def generate_site_orbital_representation(self, orbital_irrep_name: str):
        """
        orbital irrep name in full group

        :param orbital_irrep_name:
        :return:
        """

        ssg_coset_mapping, ssg_element_mapping = self.group_to_subgroup_mapping(self.site_symmetry_group,
                                                                                coset_reps=self.coset_reps_ssg)

        d_G = len(self.group.faithful_representation)
        m_tau = len(self.coset_reps_ssg.symbols)  # ssg_coset_mapping.shape[0],
        orbital_irrep = self.group.irreducible_representations[orbital_irrep_name]
        d_mu = orbital_irrep[0].shape[0] if isinstance(orbital_irrep[0], sp.MatrixBase) else 1

        d_H = m_tau * d_mu

        site_orbital_rep = []
        for R in range(d_G):
            H = sp.MutableDenseMatrix.zeros(d_H, d_H)
            for t_bra in range(m_tau):
                t_ket = ssg_coset_mapping[t_bra, R]
                element_in_ssg = ssg_element_mapping[t_bra, R]
                orbital_rep = orbital_irrep[element_in_ssg]
                orbital_rep = orbital_rep if isinstance(orbital_rep, sp.MatrixBase) else sp.Matrix([orbital_rep])

                self.set_block(H, t_bra, t_ket, orbital_rep, orbital_rep.shape[0])

            site_orbital_rep.append(H)

        return site_orbital_rep

    def generate_site_orbital_representation_2(self, orbital_irrep_name: str):
        """
        orbital irrep name in full group

        :param orbital_irrep_name:
        :return:
        """
        ssg_mapping = self.group_to_subgroup_mapping(self.site_symmetry_group, coset_reps=self.coset_reps_ssg)

        ssg_coset_mapping, ssg_element_mapping = self.group_to_subgroup_mapping(self.site_symmetry_group,
                                                                                coset_reps=self.coset_reps_ssg)

        d_G = len(ssg_mapping)
        # d_G = len(self.group.faithful_representation)
        m_tau = len(self.coset_reps_ssg.symbols)  # ssg_coset_mapping.shape[0],
        orbital_irrep = self.group.irreducible_representations[orbital_irrep_name]
        d_mu = orbital_irrep[0].shape[0] if isinstance(orbital_irrep[0], sp.MatrixBase) else 1

        d_H = m_tau * d_mu

        site_orbital_rep = []
        for R in range(d_G):
            H = sp.MutableDenseMatrix.zeros(d_H, d_H)

            for t_bra in range(m_tau):
                t_ket = ssg_coset_mapping[t_bra, R]
                element_in_ssg = ssg_element_mapping[t_bra, R]
                orbital_rep = orbital_irrep[element_in_ssg]
                orbital_rep = orbital_rep if isinstance(orbital_rep, sp.MatrixBase) else sp.Matrix([orbital_rep])

                self.set_block(H, t_bra, t_ket, orbital_rep, orbital_rep.shape[0])

            site_orbital_rep.append(H)

        return site_orbital_rep

    def _construct_projection_operator(self, shell_number: int, bra_orbital_irrep_name: str,
                                       ket_orbital_irrep_name: str, location_trial: dict | None = None):

        shell_representation = self.generate_shell_representation(shell_number)
        bra_orbital = self.generate_site_orbital_representation(bra_orbital_irrep_name)
        ket_orbital = self.generate_site_orbital_representation(ket_orbital_irrep_name)

        # projection = self.projection_operator_hamiltonian(shell_representation, bra_orbital, ket_orbital)
        representation = tensor(shell_representation, [R.conjugate() for R in bra_orbital], ket_orbital)
        projection = projection_operator(self.group.irreducible_representations['g1p'], representation, 0, 0)
        if location_trial:
            n = projection.shape[0]
            trial_vector = sp.Matrix.zeros(n, 1)
            for loc, val in location_trial.items():
                i, j, k = loc
                d_ket = ket_orbital[0].shape[0]
                d_bra = bra_orbital[0].shape[0]
                idx = i * (d_bra * d_ket) + j * (d_bra * d_ket) + k
                trial_vector[idx] = val

            d_rho = shell_representation[0].shape[0]
            d_ket = ket_orbital[0].shape[0]
            d_bra = bra_orbital[0].shape[0]
            result = self.project_invariant_hamiltonian(projection, trial_vector, d_rho=d_rho, d_bra=d_bra, d_ket=d_ket)
            return result
        else:
            return projection

    def construct_projection_operator(self, shell_number: int, bra_orbital_irrep_name: str,
                                      ket_orbital_irrep_name: str):
        return self._construct_projection_operator(shell_number, bra_orbital_irrep_name, ket_orbital_irrep_name)

    def generate_invariant_form(self, shell_number: int, bra_orbital_irrep_name: str,
                                ket_orbital_irrep_name: str, location_trial: dict[tuple, float]):

        """

        :param shell_number:
        :param bra_orbital_irrep_name:
        :param ket_orbital_irrep_name:
        :param location_trial:
        :return:
        """

        return self._construct_projection_operator(shell_number, bra_orbital_irrep_name, ket_orbital_irrep_name,
                                                   location_trial)

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

    def generate_band_representation_2(self, orbital_irrep_name: str, group_of_k: GroupOfK, zak_basis: bool = True):

        group_of_k_idx_in_full_group = self.subgroup_indices_in_full_group(group_of_k.group)

        def calculate_phase(R, t_bra_idx):
            gR = np.array(self.group.get_xyz_representation())[group_of_k_idx_in_full_group][
                     R] * group_of_k.point - group_of_k.point
            tau_bra = self.tau_vectors[t_bra_idx]
            phase = sp.exp(sp.I * (gR.dot(tau_bra)))
            return phase

        def calculate_zak_phase(k_point, tau_bra_idx, tau_ket_idx):
            tau_ket = self.tau_vectors[tau_ket_idx]
            tau_bra = self.tau_vectors[tau_bra_idx]
            phase = sp.exp(sp.I * (k_point.dot(tau_bra - tau_ket)))
            return phase

        ssg_coset_mapping = self.group_to_subgroup_mapping_3(self.site_symmetry_group,
                                                             coset_reps_ssg=self.coset_reps_ssg,
                                                             from_subgroup=group_of_k.group)

        d_G = len(ssg_coset_mapping)
        m_tau = len(self.coset_reps_ssg.symbols)  # ssg_coset_mapping.shape[0],
        orbital_irrep = self.group.irreducible_representations[orbital_irrep_name]
        d_mu = orbital_irrep[0].shape[0] if isinstance(orbital_irrep[0], sp.MatrixBase) or isinstance(orbital_irrep[0],
                                                                                                      np.ndarray) else 1

        d_H = m_tau * d_mu

        site_orbital_rep = []
        for Rk, Rk_mapped in ssg_coset_mapping.items():
            H = sp.MutableDenseMatrix.zeros(d_H, d_H)
            for t_bra, t_ket, Rt in Rk_mapped:
                orbital_rep = orbital_irrep[Rt] * calculate_phase(Rk, t_bra)
                if zak_basis:
                    zak_phase = calculate_zak_phase(group_of_k.point, tau_bra_idx=t_bra, tau_ket_idx=t_ket)
                    orbital_rep *= zak_phase
                orbital_rep = orbital_rep if isinstance(orbital_rep, sp.MatrixBase) else sp.Matrix([orbital_rep])
                self.set_block(H, t_bra, t_ket, orbital_rep, orbital_rep.shape[0])

            site_orbital_rep.append(H)

        return site_orbital_rep

    def projection_operator_k_point(self, group_k: GroupOfK, band_rep: list,
                                    irrep_in_decomp_name: str):

        mu = group_k.group.irreducible_representations[irrep_in_decomp_name]
        d_mu = mu[0].shape[0] if isinstance(mu[0], sp.MatrixBase) or isinstance(mu[0], np.ndarray) else 1
        results = {}
        for i in range(d_mu):
            for j in range(d_mu):
                results[i, j] = projection_operator(mu, band_rep, i, j)

        return results


if __name__ == '__main__':
    graphene = GrapheneHamiltonian(with_spin=True)
    graphene.run_setup(wyckoff_position='2b')
    # graphene.group_to_subgroup_mapping(graphene.site_symmetry_group, graphene.coset_reps_ssg)
    graphene.generate_band_representation_2('g2m', graphene.group_of_G)
    print('run finished')
