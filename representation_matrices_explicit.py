import sympy as sp
from utility_functions import (
    create_reflection_matrix, create_rotation_matrix, tensor, character_of_representation, irrep_decomposition,
    direct_sum, is_faithful
)

from abc import ABC, abstractmethod


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
            elem = sp.expand_complex(sp.exp(-sp.I * (jx * aa[0] + jy * aa[1] + jz * aa[2])))
            reps.append(elem)
        return reps

    @staticmethod
    def axis_angle(direction: list, angle):
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


if __name__ == '__main__':
    d2h = GroupD2h(is_double_group=True)
    # print(d6h.irreducible_representations)
    # print(tensor(d6h.irreducible_representations, d6h.irreducible_representations))
