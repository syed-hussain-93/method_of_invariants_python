import sympy as sp
from utility_functions import (
    create_reflection_matrix, create_rotation_matrix, tensor, character_of_representation, irrep_decomposition,
    direct_sum, is_faithful, rotation_matrix_3D
)

from abc import ABC

from collections import deque


class GroupGenerator:
    def __init__(self, generators, identity=None, names=None, N=100):
        """
        Args:
            generators: list of sympy objects (matrices or scalars).
            identity: identity element (default: eye for matrices, 1 for scalars).
            names: optional list of string names for generators (default: ["g1","g2",...]).
            N: maximum allowed group size (safety).
        """
        self.generators = [sp.Matrix(g) if isinstance(g, (list, tuple, sp.MatrixBase)) else sp.sympify(g)
                           for g in generators]

        if identity is None:
            if isinstance(self.generators[0], sp.MatrixBase):
                identity = sp.eye(self.generators[0].shape[0])
            else:
                identity = sp.Integer(1)
        self.identity = identity

        self.names = names if names is not None else [f"g{i}" for i in range(len(generators))]
        self.N = N

        # Will be filled after enumeration
        self.elements = []
        self.word_map = {}  # frozen -> tuple of generator indices
        self.pretty_map = {}  # frozen -> pretty string
        self._frozen_to_matrix = {}  # frozen -> normal matrix/scalar

    def _freeze(self, g):
        """Return hashable version for dict keys."""
        if isinstance(g, sp.MatrixBase):
            return sp.ImmutableMatrix(g)
        return sp.sympify(g)

    def generate(self):
        """Enumerate the group elements and record generator words."""
        frozen_id = self._freeze(self.identity)
        elements = {frozen_id: ()}
        self._frozen_to_matrix[frozen_id] = self.identity
        frontier = deque([frozen_id])

        while frontier:
            current_frozen = frontier.popleft()
            current_word = elements[current_frozen]
            current_mat = self._frozen_to_matrix[current_frozen]

            for idx, gen in enumerate(self.generators):
                # FIX: multiply on the right, append idx to word (left-to-right order)
                new_mat = current_mat * gen
                new_word = current_word + (idx,)
                new_frozen = self._freeze(new_mat)

                if new_frozen not in elements:
                    elements[new_frozen] = new_word
                    self._frozen_to_matrix[new_frozen] = new_mat
                    frontier.append(new_frozen)

                    if len(elements) > self.N:
                        raise ValueError("Exceeded maximum group size N")

        # Save results
        self.elements = [sp.Matrix(m) if isinstance(m, sp.MatrixBase) else m
                         for m in self._frozen_to_matrix.values()]
        self.word_map = elements
        self.pretty_map = {f: self._pretty(w) for f, w in elements.items()}

        return self.elements

    def _pretty(self, word):
        """Convert tuple of generator indices into human-readable word string."""
        if not word:
            return "e"
        out = []
        count = 1
        for i in range(len(word)):
            if i + 1 < len(word) and word[i] == word[i + 1]:
                count += 1
            else:
                name = self.names[word[i]]
                if count > 1:
                    out.append(f"{name}^{count}")
                else:
                    out.append(name)
                count = 1
        return " * ".join(out)

    def build_in_representation(self, new_generators):
        """
        Rebuild group elements in another representation using stored words.

        Args:
            new_generators: list of matrices or scalars for the new rep.
        Returns:
            dict mapping word tuple -> new representation element (Matrix or scalar).
        """
        new_gens = [sp.Matrix(g) if isinstance(g, (list, tuple, sp.MatrixBase)) else sp.sympify(g)
                    for g in new_generators]
        rep_elements = {}
        for word in self.word_map.values():
            # choose identity for this rep
            if isinstance(new_gens[0], sp.MatrixBase):
                g = sp.eye(new_gens[0].shape[0])
            else:
                g = sp.Integer(1)

            # FIX: multiply on the right, follow word order
            for idx in word:
                g = g * new_gens[idx]
            rep_elements[word] = g
        return rep_elements

    def show_words(self):
        """Print all elements with their word representations."""
        for frozen, word in self.word_map.items():
            mat = self._frozen_to_matrix[frozen]
            print(f"{self._pretty(word)}:\n{mat}\n")


class BaseGroup(ABC):

    def __init__(self):
        self.elements = None
        self.irreducible_representations = None

    def _elements(self):
        pass

    def _irreducible_representations(self):
        pass

    def faithful_representation(self):
        pass

    def xyz_representation(self):
        pass

    def character_table(self):
        ch_table = {}
        for name, ir in self.irreducible_representations.items():
            ch_table[name] = character_of_representation(ir)
        return ch_table


class SU_N:

    @staticmethod
    def J_operators(j):

        """
        The operators are with respect to a basis which ranges  from <j -j| to <j j| in increments of m=1.
        For instance, in the spin-1/2 case, the first basis vector is spin-down, and the second is spin-up.
        This is somewhat contrary to Physics convention.

        Factors of Planck's constant have been omitted from all operators, and
        the raising and lowering operators (J+ and J-) have an implicit square
        root.  This is to keep the results exactly representable.  To correct for
        this, something like the following is necessary:

        Jx = hbar * Jx;                  and same for Jy and Jz
        Jminus = hbar * Jminus .^ 0.5;   and same for Jplus

        :param j:
        :return:
        """
        if not isinstance(j, sp.Rational):
            j = sp.Rational(j)
        mvalues = [j - i for i in range(int(2 * j) + 1)]
        cardinality = len(mvalues)

        # Initialize matrices
        Jminus = sp.Matrix.zeros(cardinality)
        Jplus = sp.Matrix.zeros(cardinality)

        # Jminus (lowering operator)
        for i in range(cardinality - 1):
            m = mvalues[i]
            value = sp.sqrt(j * (j + 1) - m * (m - 1))
            Jminus[i + 1, i] = value

        # Jplus (raising operator)
        for i in range(1, cardinality):
            m = mvalues[i]
            value = sp.sqrt(j * (j + 1) - m * (m + 1))
            Jplus[i - 1, i] = value

        # Jx, Jy, Jz
        Jx = (Jplus + Jminus) / 2
        Jy = (Jplus - Jminus) / (2 * sp.I)
        Jz = sp.diag(*mvalues)

        return {'jx': Jx, 'jy': Jy, 'jz': Jz, 'j_plus': Jplus, 'j_minus': Jminus}

    def representation(self, j, axis_angle_representation):
        j_operators = self.J_operators(j)
        jx, jy, jz = j_operators['jx'], j_operators['jy'], j_operators['jz']
        reps = []

        if not isinstance(axis_angle_representation[0], list):
            axis_angle_representation = [axis_angle_representation]

        for aa in axis_angle_representation:
            elem = (-sp.I * (jx * aa[0] + jy * aa[1] + jz * aa[2])).exp()
            reps.append(elem)
        return reps

    @staticmethod
    def axis_angle(direction: list, angle):
        axis = sp.Matrix(direction)
        unit_axis = axis / axis.norm()
        # return sp.simplify(angle * unit_axis)
        return list(sp.Matrix(direction) / sp.Matrix(direction).norm() * angle)


class GroupD6D3hC6v(BaseGroup):

    def __init__(self, name: str, edge_x_orientation: bool = False, is_double_group: bool = False):
        super().__init__()
        self.name = name.lower()
        self.is_double = is_double_group
        self.edge_x_orientation = edge_x_orientation

        self.elements = self._elements()
        self.irreducible_representations = self._irreducible_representations()

    def _elements(self):
        if self.edge_x_orientation:
            edge = 'x'
        else:
            edge = 'y'
        e1 = ['E', 'C3', 'C3^2']
        e2 = [f"C'2({edge})*{e}" for e in e1]
        e3 = [f'C2*{e}' for e in e1]
        e4 = [f'C2*C3*{e}' for e in e2]
        elements = e1 + e2 + e3 + e4

        if self.is_double:
            elements += [f'E_bar*{e}' for e in elements]

        elements = [e.replace('*E', '').replace('E*', '') for e in elements]

        return elements

    def _irreducible_representations(self):
        ir1 = [1, 1, 1, 1, 1, 1]
        ir2 = [1, 1, 1, -1, -1, -1]
        ir3 = [1, 1, 1, 1, 1, 1]
        ir4 = [1, 1, 1, -1, -1, -1]

        c3 = create_rotation_matrix(2 * sp.pi / 3)
        if self.edge_x_orientation:
            c2p = create_reflection_matrix(0)
        else:
            # c2p = create_reflection_matrix(sp.pi / 6)
            c2p = create_reflection_matrix(sp.pi / 2)  # y edge

        c2_1 = 1
        c2_2 = -1
        c2 = create_rotation_matrix(sp.pi)

        ir5 = [
            create_rotation_matrix(0),
            c3,
            c3 ** 2,
            c2p,
            c2p * c3,
            c2p * c3 ** 2,
            c2,
            c2 * c3,
            c2 * c3 ** 2,
            c2 * c3 * c2p,
            c2 * c3 * c2p * c3,
            c2 * c3 * c2p * c3 ** 2,
        ]

        ir1 += [c2_1 * i for i in ir1]
        ir2 += [c2_1 * i for i in ir2]
        ir3 += [c2_2 * i for i in ir3]
        ir4 += [c2_2 * i for i in ir4]
        ir6 = tensor(ir3, ir5)

        irrep_names = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6']
        irreps = [ir1, ir2, ir3, ir4, ir5, ir6]
        if self.is_double:
            irrep_names += ['g7', 'g8', 'g9']
            irreps = self.get_double_group_irreps(irreps)

        d6_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}
        return d6_irreps

    def get_double_group_irreps(self, single_group_irreps: list):
        # double group irreps of D6

        j = 5 / 2
        su_n_rep = SU_N()
        e_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=0)
        c3_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi / 3)

        if self.edge_x_orientation:
            c2_p_aa = su_n_rep.axis_angle(direction=[1, 0, 0], angle=sp.pi)
        else:
            # c2p = create_reflection_matrix(sp.pi / 6)
            c2_p_aa = su_n_rep.axis_angle(direction=[0, 1, 0], angle=sp.pi)

        # c2_p_aa = su_n_rep.axis_angle(direction=[sp.cos(sp.pi / 6), sp.sin(sp.pi / 6), 0], angle=sp.pi)

        c2_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=sp.pi)

        axis_angle_reps = [
            e_aa,
            c3_aa,
            c2_p_aa,
            c2_aa
        ]
        gen_su_n = su_n_rep.representation(j, axis_angle_representation=axis_angle_reps)
        e, c3, c2p, c2 = tuple(gen_su_n)
        single_group_su_n_rep = [
            e,
            c3,
            c3 ** 2,
            c2p,
            c2p * c3,
            c2p * c3 * c3,
            c2,
            c2 * c3,
            c2 * c3 * c3,
            c2 * c3 * c2p,
            c2 * c3 * c2p * c3,
            c2 * c3 * c2p * c3 * c3,
        ]

        e_bar_axis_angle = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_axis_angle)[0]

        double_group_su_n_rep = single_group_su_n_rep.copy()
        for sg in single_group_su_n_rep:
            double_group_su_n_rep.append(e_bar * sg)

        P = sp.Matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        g7 = []
        g8 = []
        g9 = []

        for elem in double_group_su_n_rep:
            blk_diag = sp.expand_complex(P.inv() * elem * P)
            g7.append(blk_diag[4:6, 4:6])
            g8.append(blk_diag[0:2, 0:2])
            g9.append(blk_diag[2:4, 2:4])

        double_group_irreps = [ir * 2 for ir in single_group_irreps] + [g7, g8, g9]

        return double_group_irreps

    def character_table(self):
        ch_table = {}
        for name, ir in self.irreducible_representations.items():
            ch_table[name] = character_of_representation(ir)
        return ch_table

    def faithful_representation(self):
        if self.is_double:
            if self.name == 'd6':
                rep = direct_sum(self.irreducible_representations['g7'], self.irreducible_representations['g2'])
            elif self.name == 'c6v':
                rep = direct_sum(self.irreducible_representations['g7'], self.irreducible_representations['g1'])
            elif self.name == 'd3h':
                rep = direct_sum(self.irreducible_representations['g7'], self.irreducible_representations['g4'])
        else:
            if self.name == 'd6':
                rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g2'])
            elif self.name == 'c6v':
                rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g1'])
            elif self.name == 'd3h':
                rep = direct_sum(self.irreducible_representations['g6'], self.irreducible_representations['g4'])

        if not is_faithful(rep):
            raise ValueError('Representation not faithful')
        return rep

    def xyz_representation(self):
        if self.name == 'd6':
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g2'])
        elif self.name == 'c6v':
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g1'])
        elif self.name == 'd3h':
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g4'])
        else:
            raise ValueError('name not provided')
        return rep


class GroupD6h(BaseGroup):

    def __init__(self, is_double_group: bool = False, edge_x_orientation: bool = False):
        super().__init__()
        self.is_double = is_double_group
        self.edge_x_orientation = edge_x_orientation
        self.group_d6 = GroupD6D3hC6v(name='d6', is_double_group=is_double_group, edge_x_orientation=edge_x_orientation)

        self.elements = self._elements()
        self.irreducible_representations = self._irreducible_representations()

    def _elements(self) -> list[str]:
        elements = self.group_d6.elements

        elements += [f'I*{e}' for e in elements]

        elems = [e.replace('*E', '').replace('E*', '') for e in elements]

        return elems

    def _irreducible_representations(self):
        # ir1 = [1, 1, 1, 1, 1, 1]
        # ir2 = [1, 1, 1, -1, -1, -1]
        # ir3 = [1, 1, 1, 1, 1, 1]
        # ir4 = [1, 1, 1, -1, -1, -1]
        #
        # c3 = create_rotation_matrix(2 * sp.pi / 3)
        # if self.edge_x_orientation:
        #     c2p = create_reflection_matrix(0)
        # else:
        #     # c2p = create_reflection_matrix(sp.pi / 6)
        #     c2p = create_reflection_matrix(sp.pi / 2)  # y edge
        #
        # ir5 = [
        #     create_rotation_matrix(0),
        #     c3,
        #     c3 ** 2,
        #     c2p,
        #     c2p * c3,
        #     c2p * c3 ** 2
        # ]
        #
        # c2_1 = 1
        # c2_2 = -1
        # c2_rot = create_rotation_matrix(sp.pi)
        #
        # ir1 += [c2_1 * i for i in ir1]
        # ir2 += [c2_1 * i for i in ir2]
        # ir3 += [c2_2 * i for i in ir3]
        # ir4 += [c2_2 * i for i in ir4]
        # ir5 += [c2_rot * i for i in ir5]
        # ir6 = tensor(ir3, ir5)
        # d6_irreps = [ir1, ir2, ir3, ir4, ir5, ir6]
        d6_irreps = list(self.group_d6.irreducible_representations.values())

        irrep_names_p = ['g1p', 'g2p', 'g3p', 'g4p', 'g5p', 'g6p']
        irrep_names_m = ['g1m', 'g2m', 'g3m', 'g4m', 'g5m', 'g6m']

        if self.is_double:
            irrep_names_p += ['g7p', 'g8p', 'g9p']
            irrep_names_m += ['g7m', 'g8m', 'g9m']
            # d6_irreps = self.get_double_group_irreps(d6_irreps)

        irreps_p = []
        irreps_m = []
        c2_1 = 1
        c2_2 = -1
        for ir in d6_irreps:
            irreps_p.append(ir + [c2_1 * i for i in ir])
            irreps_m.append(ir + [c2_2 * i for i in ir])

        d6h_irreps = {name: ir for name, ir in zip(irrep_names_p + irrep_names_m, irreps_p + irreps_m)}
        return d6h_irreps

    # def get_double_group_irreps(self, single_group_irreps: list):
    #     # double group irreps of D6
    #
    #     j = 5 / 2
    #     su_n_rep = SU_N()
    #     e_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=0)
    #     c3_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi / 3)
    #
    #     if self.edge_x_orientation:
    #         c2_p_aa = su_n_rep.axis_angle(direction=[1, 0, 0], angle=sp.pi)
    #     else:
    #         # c2p = create_reflection_matrix(sp.pi / 6)
    #         c2_p_aa = su_n_rep.axis_angle(direction=[0, 1, 0], angle=sp.pi)
    #     # c2_p_aa = su_n_rep.axis_angle(direction=[sp.cos(sp.pi / 6), sp.sin(sp.pi / 6), 0], angle=sp.pi)
    #
    #     c2_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=sp.pi)
    #
    #     axis_angle_reps = [e_aa, c3_aa, c2_p_aa, c2_aa]
    #     gen_su_n = su_n_rep.representation(j, axis_angle_representation=axis_angle_reps)
    #     e, c3, c2_p, c2 = tuple(gen_su_n)
    #     single_group_su_n_rep = [e, c3, c3 ** 2, c2_p, c2_p * c3, c2_p * c3 ** 2]
    #     single_group_su_n_rep += [c2 * g for g in single_group_su_n_rep]
    #
    #     e_bar_axis_angle = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi)
    #     e_bar = su_n_rep.representation(j, e_bar_axis_angle)[0]
    #
    #     double_group_su_n_rep = single_group_su_n_rep.copy()
    #     for sg in single_group_su_n_rep:
    #         double_group_su_n_rep.append(e_bar * sg)
    #
    #     P = sp.Matrix([
    #         [1, 0, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 1, 0, 0],
    #         [0, 1, 0, 0, 0, 0]
    #     ])
    #
    #     g7 = []
    #     g8 = []
    #     g9 = []
    #
    #     for elem in double_group_su_n_rep:
    #         blk_diag = sp.expand_complex(P.inv() * elem * P)
    #         g7.append(blk_diag[4:6, 4:6])
    #         g8.append(blk_diag[0:2, 0:2])
    #         g9.append(blk_diag[2:4, 2:4])
    #
    #     double_group_irreps = [ir * 2 for ir in single_group_irreps] + [g7, g8, g9]
    #
    #     return double_group_irreps

    def character_table(self):
        ch_table = {}
        for name, ir in self.irreducible_representations.items():
            ch_table[name] = character_of_representation(ir)
        return ch_table

    def faithful_representation(self):
        if self.is_double:
            rep = direct_sum(self.irreducible_representations['g7p'], self.irreducible_representations['g2m'])
        else:
            rep = direct_sum(self.irreducible_representations['g5m'], self.irreducible_representations['g2m'])
        if not is_faithful(rep):
            raise ValueError('Representation not faithful')

        return rep

    def xyz_representation(self):
        rep = direct_sum(self.irreducible_representations['g5m'], self.irreducible_representations['g2m'])
        return rep


class GroupD2C2v(BaseGroup):

    def __init__(self, name: str, is_double: bool = False):
        super().__init__()
        self.name = name
        self.is_double = is_double
        self.elements = self._elements()
        self.irreducible_representations = self._irreducible_representations()

    def _elements(self):
        elements = ['E', 'C2(z)', "C'2(y)", "C''2(x)"]

        if self.is_double:
            elements += [f'E_bar*{e}' for e in elements]

        elements = [e.replace('*E', '').replace('E*', '') for e in elements]
        return elements

    def _irreducible_representations(self):
        ir1 = [1, 1, 1, 1]
        ir2 = [1, -1, 1, -1]
        ir3 = [1, 1, -1, -1]
        ir4 = [1, -1, -1, 1]

        irrep_names = ['g1', 'g2', 'g3', 'g4']
        irreps = [ir1, ir2, ir3, ir4]
        if self.is_double:
            irrep_names += ['g5']
            irreps = self.get_double_group_irreps(irreps)

        d2_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}

        return d2_irreps

    def get_double_group_irreps(self, single_group_irreps: list):

        j = 1 / 2
        su_n_rep = SU_N()

        e_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=0)
        c2_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=sp.pi)
        c2p_aa = su_n_rep.axis_angle(direction=[0, 1, 0], angle=sp.pi)
        c2pp_aa = su_n_rep.axis_angle(direction=[1, 0, 0], angle=sp.pi)

        axis_angle_reps = [e_aa, c2_aa, c2p_aa, c2pp_aa]
        gen_su_n = su_n_rep.representation(j, axis_angle_representation=axis_angle_reps)
        e, c2, c2p, c2pp = tuple(gen_su_n)

        single_group_su_n_rep = [e, c2, c2p, c2pp]
        e_bar_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_aa)[0]

        double_group_su_n_rep = single_group_su_n_rep.copy()
        for sg in single_group_su_n_rep:
            double_group_su_n_rep.append(e_bar * sg)

        double_group_irreps = [ir * 2 for ir in single_group_irreps] + double_group_su_n_rep
        return double_group_irreps

    def faithful_representation(self):
        if self.is_double:
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g3'])
        else:
            rep = direct_sum(self.irreducible_representations['g4'], self.irreducible_representations['g2'],
                             self.irreducible_representations['g3'])
        if not is_faithful(rep):
            raise ValueError('Representation not faithful')
        return rep

    def xyz_representation(self):
        if self.name == 'd2':
            rep = direct_sum(self.irreducible_representations['g4'], self.irreducible_representations['g2'],
                             self.irreducible_representations['g3'])
        elif self.name == 'c2v':
            rep = direct_sum(self.irreducible_representations['g2'], self.irreducible_representations['g4'],
                             self.irreducible_representations['g1'])
        else:
            raise ValueError('name not provided')

        return rep


class GroupD2h(BaseGroup):

    def __init__(self, is_double_group: bool = False):
        super().__init__()
        self.is_double_group = is_double_group
        self.elements = self._elements()
        self.irreducible_representations = self._irreducible_representations()

    def _elements(self):
        elements = ['E', 'C2(z)', "C'2(y)", "C''2(x)"]

        if self.is_double_group:
            elements += [f'E_bar*{e}' for e in elements]

        elements += [f'I*{e}' for e in elements]
        elements = [e.replace('*E', '').replace('E*', '') for e in elements]
        return elements

    def _irreducible_representations(self):
        ir1 = [1, 1, 1, 1]
        ir2 = [1, -1, 1, -1]
        ir3 = [1, 1, -1, -1]
        ir4 = [1, -1, -1, 1]

        d2_irreps = [ir1, ir2, ir3, ir4]

        irrep_names_p = ['g1p', 'g2p', 'g3p', 'g4p']
        irrep_names_m = ['g1m', 'g2m', 'g3m', 'g4m']

        i1 = 1
        i2 = -1

        if self.is_double_group:
            irrep_names_p += ['g5p']
            irrep_names_m += ['g5m']
            d2_irreps = self.get_double_group_irreps(d2_irreps)

        irreps_p = []
        irreps_m = []
        for ir in d2_irreps:
            irreps_p.append(ir + [i1 * i for i in ir])
            irreps_m.append(ir + [i2 * i for i in ir])

        d2h_irreps = {name: ir for name, ir in zip(irrep_names_p + irrep_names_m, irreps_p + irreps_m)}

        return d2h_irreps

    def get_double_group_irreps(self, single_group_irreps: list):

        j = 1 / 2
        su_n_rep = SU_N()

        e_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=0)
        c2_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=sp.pi)
        c2p_aa = su_n_rep.axis_angle(direction=[0, 1, 0], angle=sp.pi)
        c2pp_aa = su_n_rep.axis_angle(direction=[1, 0, 0], angle=sp.pi)

        axis_angle_reps = [e_aa, c2_aa, c2p_aa, c2pp_aa]
        gen_su_n = su_n_rep.representation(j, axis_angle_representation=axis_angle_reps)
        e, c2, c2p, c2pp = tuple(gen_su_n)

        single_group_su_n_rep = [e, c2, c2p, c2pp]
        e_bar_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_aa)[0]

        double_group_su_n_rep = single_group_su_n_rep.copy()
        for sg in single_group_su_n_rep:
            double_group_su_n_rep.append(e_bar * sg)

        double_group_irreps = [ir * 2 for ir in single_group_irreps] + [double_group_su_n_rep]
        return double_group_irreps

    def faithful_representation(self):
        if self.is_double_group:
            rep = direct_sum(self.irreducible_representations['g5p'], self.irreducible_representations['g3m'])
        else:
            rep = direct_sum(self.irreducible_representations['g4m'], self.irreducible_representations['g2m'],
                             self.irreducible_representations['g3m'])
        if not is_faithful(rep):
            raise ValueError('Representation not faithful')
        return rep

    def xyz_representation(self):
        rep = direct_sum(self.irreducible_representations['g4m'], self.irreducible_representations['g2m'],
                         self.irreducible_representations['g3m'])
        return rep


class GroupTdO(BaseGroup):

    def __init__(self, name: str, is_double_group: bool):
        super().__init__()
        self.name = name
        self.is_double_group = is_double_group

        e = rotation_matrix_3D((0, 0, 1), 0)
        r = rotation_matrix_3D((0, 0, 1), 2 * sp.pi / 4)  # C4(z)
        s = rotation_matrix_3D((1, 1, 0), 2 * sp.pi / 2)  # C2([110])
        generators = [r, s]

        self.group_generator = GroupGenerator(generators, identity=e, names=['C4(z)', 'C2([110])'])
        self.group_generator.generate()

        self.elements = self._elements()
        self.irreducible_representations = self._irreducible_representations()

    def _elements(self):

        elements = list(self.group_generator.pretty_map.values())
        if self.is_double_group:
            elements += [f'E_bar * {e}' for e in elements]

        elements = [e.replace('* E', '').replace('E *', '') for e in elements]

        return elements

    def _irreducible_representations(self):
        r1, s1 = 1, 1
        r2, s2 = -1, -1
        r3 = sp.Matrix([[0, -1], [1, 0]])
        s3 = sp.Matrix([[1, 0], [0, -1]])

        r4 = rotation_matrix_3D((0, 0, 1), 2 * sp.pi / 4)  # C4(z)
        s4 = rotation_matrix_3D((1, 1, 0), 2 * sp.pi / 2)  # C2([110])

        g1 = list(self.group_generator.build_in_representation([r1, s1]).values())
        g2 = list(self.group_generator.build_in_representation([r2, s2]).values())
        g3 = list(self.group_generator.build_in_representation([r3, s3]).values())
        g4 = list(self.group_generator.build_in_representation([r4, s4]).values())

        g5 = tensor(g4, g2)

        # g4g4 = tensor(g4, g4)

        irrep_names = ['g1', 'g2', 'g3', 'g4', 'g5']
        irreps = [g1, g2, g3, g4, g5]

        if self.is_double_group:
            irrep_names += ['g6', 'g7', 'g8']
            irreps = self.get_double_group_irreps(irreps)

        irreps = [[R.expand(complex=True) if isinstance(R, sp.Matrix) else R for R in ir] for ir in irreps]
        full_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}
        return full_irreps

    def get_SU_N_rep(self, j):
        su_n_rep = SU_N()

        r = rotation_matrix_3D((0, 0, 1), 2 * sp.pi / 4)  # C4(z)
        s = rotation_matrix_3D((1, 1, 0), 2 * sp.pi / 2)  # C2([110])

        r_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi / 4)
        s_aa = su_n_rep.axis_angle(direction=[1, 1, 0], angle=2 * sp.pi / 2)

        axis_angle_reps = [r_aa, s_aa]
        r, s = su_n_rep.representation(j, axis_angle_representation=axis_angle_reps)
        e_bar_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_aa)[0]

        single_group_su_n_rep = list(self.group_generator.build_in_representation([r, s]).values())

        g6 = single_group_su_n_rep.copy()
        for sg in single_group_su_n_rep:
            g6.append(e_bar * sg)

        return g6

    def get_double_group_irreps(self, single_group_irreps: list):

        j = sp.Rational(1, 2)
        g6 = self.get_SU_N_rep(j=j)

        single_group_doubled = [ir * 2 for ir in single_group_irreps]
        # g6 = [R.expand(complex=True).simplify() for R in g6]
        g7 = tensor(g6, single_group_doubled[1])
        g8 = tensor(g6, single_group_doubled[2])
        double_group_su_n_rep = [g6, g7, g8]

        double_group_irreps = single_group_doubled + double_group_su_n_rep

        return double_group_irreps

    def faithful_representation(self):
        if self.is_double_group:
            if self.name == 'Td':
                rep = direct_sum(self.irreducible_representations['g6'], self.irreducible_representations['g2'])
            elif self.name == 'O':
                rep = direct_sum(self.irreducible_representations['g6'], self.irreducible_representations['g2'])
        else:
            if self.name == 'Td':
                rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g1'])
            elif self.name == 'O':
                rep = direct_sum(self.irreducible_representations['g4'], self.irreducible_representations['g2'])

        if not is_faithful(rep):
            raise ValueError('Representation not faithful')
        return rep

    def xyz_representation(self):
        if self.name == 'Td':
            rep = self.irreducible_representations['g5']
        elif self.name == 'O':
            rep = self.irreducible_representations['g4']
        else:
            raise ValueError('name not provided')

        return rep


if __name__ == '__main__':
    Td = GroupTdO(name='Td', is_double_group=True)
    Td.get_SU_N_rep(j=1 / 2)
    # print(d6h.irreducible_representations)
    # print(tensor(d6h.irreducible_representations, d6h.irreducible_representations))
