from representation_matrices_explicit import GroupD6h, GroupD6D3hC6v, BaseGroup
import sympy as sp
from dataclasses import dataclass
from utility_functions import ismember
import numpy as np


class Lattice:
    def __init__(self):
        self._a = sp.Symbol('a', real=True, positive=True)

        # Real-space lattice vectors
        self._a1 = None
        self._a2 = None
        self._a3 = None

        # Reciprocal lattice vectors
        self._b1 = None
        self._b2 = None
        self._b3 = None

    @property
    def a(self):
        return self._a

    # ----- Real-space lattice vectors -----
    @property
    def a1(self):
        if self._a1 is None:
            self._a1 = self._get_a1()
        return self._a1

    @a1.setter
    def a1(self, value):
        self._a1 = value

    @property
    def a2(self):
        if self._a2 is None:
            self._a2 = self._get_a2()
        return self._a2

    @a2.setter
    def a2(self, value):
        self._a2 = value

    @property
    def a3(self):
        if self._a3 is None:
            self._a3 = self._get_a3()
        return self._a3

    @a3.setter
    def a3(self, value):
        self._a3 = value

    # ----- Reciprocal lattice vectors -----
    @property
    def b1(self):
        if self._b1 is None:
            self._b1 = self._get_b1()
        return self._b1

    @property
    def b2(self):
        if self._b2 is None:
            self._b2 = self._get_b2()
        return self._b2

    @property
    def b3(self):
        if self._b3 is None:
            self._b3 = self._get_b3()
        return self._b3

    def _get_a1(self):
        raise NotImplementedError("Define lattice vector a1")

    def _get_a2(self):
        raise NotImplementedError("Define lattice vector a2")

    def _get_a3(self):
        raise NotImplementedError("Define lattice vector a2")

    def _get_b1(self):
        return 2 * sp.pi * self.a2.cross(self.a3) / self._volume()

    def _get_b2(self):
        return 2 * sp.pi * self.a3.cross(self.a1) / self._volume()

    def _get_b3(self):
        return 2 * sp.pi * self.a1.cross(self.a2) / self._volume()

    def _volume(self):
        return self.a1.dot(self.a2.cross(self.a3))

    def get_lattice_vector(self, n: list[int]):
        """Compute lattice vector from real-space basis."""
        if len(n) != 3:
            raise ValueError("Expected a list of 3 integers")
        return n[0] * self.a1 + n[1] * self.a2 + n[2] * self.a3

    def get_reciprocal_vector(self, m: list[int]):
        """Compute reciprocal lattice vector from (m1, m2, m3)."""
        if len(m) != 3:
            raise ValueError("Expected a list of 3 integers")
        return m[0] * self.b1 + m[1] * self.b2 + m[2] * self.b3


class Hexagonal2DLattice(Lattice):

    def _get_a1(self):
        a1 = (self.a * sp.sqrt(3) / 2) * (sp.Matrix([1, 0, 0]) - sp.sqrt(3) * sp.Matrix([0, 1, 0]))
        return a1

    def _get_a2(self):
        a2 = (self.a * sp.sqrt(3) / 2) * (sp.Matrix([1, 0, 0]) + sp.sqrt(3) * sp.Matrix([0, 1, 0]))
        return a2

    def _get_a3(self):
        return sp.Matrix([0, 0, 1])


class CosetReps:
    symbols: list[str]
    full_group: BaseGroup

    def __init__(self, symbols: list[str], full_group: BaseGroup):
        self.symbols = symbols
        self.full_group = full_group
        self.indices = [self.full_group.elements.index(x) for x in self.symbols]
        # [i for i, x in enumerate(full_group.elements) if x in self.symbols]
        self.faithful_representation = [self.full_group.faithful_representation()[i] for i in self.indices]
        self.xyz_representation = [self.full_group.xyz_representation()[i] for i in self.indices]


class GrapheneHamiltonian:

    def __init__(self, with_spin):
        self.with_spin = with_spin
        if self.with_spin:
            self.group = GroupD6h(is_double_group=True)
        else:
            self.group = GroupD6h()

        self.lattice = Hexagonal2DLattice()

        # self._site_symmetry_group = None
        self.site_symmetry_group = None
        self.coset_reps_dict = None
        self.tau_vectors = None

    # @property
    # def site_symmetry_group(self):
    #     return GroupD6D3hC6v(name='d3h', is_double_group=self.with_spin, edge_x_orientation=False)
    #
    # @site_symmetry_group.setter
    # def site_symmetry_group(self, value):
    #     self._site_symmetry_group = value

    def run_setup(self, wyckoff_position: str):
        if wyckoff_position == '2b':
            self.site_symmetry_group = GroupD6D3hC6v(name='d3h', is_double_group=self.with_spin,
                                                     edge_x_orientation=False)

            self.coset_reps_ssg = CosetReps(symbols=['E', 'I'], full_group=self.group)

            self.tau_vectors = {
                0: sp.Rational(1, 3) * self.lattice.a1 + sp.Rational(2, 3) * self.lattice.a2,
                1: sp.Rational(2, 3) * self.lattice.a1 + sp.Rational(1, 3) * self.lattice.a2
            }

        else:
            raise ValueError('No setup for this wyckoff posistion')

    def full_group_to_subgroup_mapping(self, sub_group, coset_reps: CosetReps):

        full_group = self.group
        full_group_faithful_rep = full_group.faithful_representation()
        sub_group_faithful_rep = sub_group.faithful_representation()

        sub_group_index_in_full_group = [full_group_faithful_rep.index(x) for x in sub_group_faithful_rep]
        sub_group_in_full_group = [full_group_faithful_rep[i] for i in sub_group_index_in_full_group]

        coset_reps_f = coset_reps.faithful_representation

        left_coset_sub_group = [[full_group_faithful_rep.index(R_row * R_sg) for R_sg in sub_group_in_full_group] for
                                R_row in coset_reps_f]
        right_coset_full_group = [[full_group_faithful_rep.index(R * R_col) for R in full_group_faithful_rep] for R_col
                                  in coset_reps_f]

        coset_pairs_mapping, element_mapping = ismember(right_coset_full_group, left_coset_sub_group)

        return coset_pairs_mapping, np.array(sub_group_index_in_full_group)[element_mapping]

    def generate_shell_vectors(self, shell_number: int, t_bra: int, t_ket: int):
        """
        t_bra/t_ket defines between what two wyckoff positions we want the shell vector for

        pass as integer which corresponds to the coset rep wrt ssg
        """

        xyz_full_group = self.group.xyz_representation()

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

        xyz_full_group = self.group.xyz_representation()

        shell_vectors = self.generate_shell_vectors(shell_number, t_bra, t_ket)
        permutations = np.array(
            [[shell_vectors.index(R_xyz * rho) for R_xyz in xyz_full_group] for rho in shell_vectors]).T
        permutation_rep = [self.permutation_representation(p) for p in permutations]

        return permutation_rep

    def project_invariant_hamiltonian(self, projection_operator, trial_vector, d_rho, d_bra, d_ket):

        projected = (projection_operator * trial_vector).expand()

        blocks = []
        for i in range(d_rho):
            start = i * d_bra * d_ket
            end = (i + 1) * d_bra * d_ket
            block_vec = projected[start:end, :]
            block_mat = block_vec.reshape(d_bra, d_ket)
            blocks.append(block_mat)
        return blocks

    def projection_operator(self, shell_representation, site_orbital_representation_bra,
                            site_orbital_representation_ket, trial_vector=None):

        from utility_functions import tensor
        d_rho = shell_representation[0].shape[0]
        bra_orb = [R.conjugate() for R in site_orbital_representation_bra]
        ket_orb = site_orbital_representation_ket

        projection_T = tensor(shell_representation, bra_orb, ket_orb)

        projection_T = sp.Rational(1, len(ket_orb) / d_rho) * sum(projection_T, sp.Matrix.zeros(*projection_T[0].shape))

        return projection_T

    # --- helpers to get/set blocks ---

    def get_block(M, i, j, block_size):
        a = block_size
        return M[i * a:(i + 1) * a, j * a:(j + 1) * a]

    def set_block(self, M, i, j, block, block_size):
        a = block_size
        if block.shape != (a, a):
            raise ValueError(f"block must be {a}x{a}, got {block.shape}")
        M[i * a:(i + 1) * a, j * a:(j + 1) * a] = block

    def generate_site_orbital_representation(self, orbital_irrep_name: str):
        """
        orbital irrep name in full group

        :param orbital_irrep_name:
        :return:
        """
        ssg_coset_mapping, ssg_element_mapping = self.full_group_to_subgroup_mapping(self.site_symmetry_group,
                                                                                     coset_reps=self.coset_reps_ssg)

        d_G = len(self.group.faithful_representation())
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

        projection = self.projection_operator(shell_representation, bra_orbital, ket_orbital)

        if location_trial:
            n = projection.shape[0]
            trial_vector = sp.Matrix.zeros(n, 1)
            for loc, val in location_trial.items():
                trial_vector[sum(loc)] = val

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

        return self._construct_projection_operator(shell_number, bra_orbital_irrep_name, ket_orbital_irrep_name,
                                                   location_trial)


if __name__ == '__main__':
    graphene = GrapheneHamiltonian(with_spin=False)
    coset_rep = CosetReps(symbols=['E', 'I'], full_group=graphene.group)
    sub_group = GroupD6D3hC6v(name='d3h', edge_x_orientation=True)
    graphene.full_group_to_subgroup_mapping(sub_group, coset_rep)
    print('run finished')
