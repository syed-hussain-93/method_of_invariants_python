import itertools
import sympy as sp
from abc import ABC, abstractmethod
from functools import reduce


def create_rotation_matrix(theta):
    """Create a 2D rotation matrix about the origin by an angle theta be denoted """

    # Rotation matrix
    r = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)],
        [sp.sin(theta), sp.cos(theta)]
    ])

    return r


def create_reflection_matrix(theta):
    """Create 2D reflection matrix about a line L through the origin which makes an angle theta with the x-axis"""

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
        kron_product = sp.Matrix(reduce(sp.BlockDiagMatrix, elem).doit())
        result.append(kron_product)
    return result


def convert_to_immutable_matrix(rep: list):
    return [sp.ImmutableMatrix(g if isinstance(g, (list, tuple, sp.MatrixBase)) else [g]) for g in rep]


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


class BaseGroup(ABC):

    def __init__(self, name: str | None = None, is_double_group: bool = False):
        self.is_double_group = is_double_group
        self.name = name

        self.irreducible_representations = None
        self.characters = None
        self.elements = None
        self.order = None
        self.number_of_irreps = None
        self.faithful_representation = None

    @abstractmethod
    def _generate_elements(self) -> list:
        pass

    @abstractmethod
    def _generate_irreducible_representations(self) -> list:
        pass

    def _generate_characters(self) -> list:
        chars = []
        for ir in self.irreducible_representations:
            ir_char = self.character_of_representation(ir)
            chars.append(ir_char)
        return chars

    def get_elements(self):
        if self.elements is None:
            self.elements = self._generate_elements()
        return self.elements

    def get_irreducible_representations(self):
        if self.irreducible_representations is None:
            irreps = self._generate_irreducible_representations()
            self.irreducible_representations = irreps  # [convert_to_immutable_matrix(ir) for ir in irreps]
            self.number_of_irreps = len(self.irreducible_representations)
            self.order = len(self.irreducible_representations[0])
        return self.irreducible_representations

    def get_characters(self):
        if self.characters is None:
            self.characters = self._generate_characters()
        return self.irreducible_representations

    def get_faithful_representation(self):
        if self.faithful_representation is None:
            self.faithful_representation = self._generate_faithful_representation()
        return self.faithful_representation

    @staticmethod
    def character_of_representation(representation: list) -> list:
        return [sp.Trace(g).simplify() if isinstance(g, sp.MatrixBase) else g for g in representation]

    def irrep_decomposition(self, representation: list):
        chars_rep = self.character_of_representation(representation)
        ir_vec = []
        for ir in self.irreducible_representations:
            chars_ir = self.character_of_representation(ir)
            decomp = sum([sp.conjugate(i) * j for i, j in zip(chars_ir, chars_rep)]) / self.order
            ir_vec.append(decomp)
        return ir_vec

    @staticmethod
    def _generate_additional_elements(element_multiply: str, group_elements: list):
        additional_group_elements = group_elements.copy()
        for elem in group_elements:
            additional_group_elements.append(element_multiply + elem)
        return additional_group_elements

    def projection_operator(self, i, j, irrep: list[sp.ImmutableMatrix], transformation: list):
        if not isinstance(irrep[0], sp.MatrixBase):
            irrep = [sp.Matrix([el]) for el in irrep]

        d_mu = irrep[0].shape[0]
        shape = transformation[0].shape
        P_ij = sp.expand_complex(sp.Rational(d_mu, self.order) * sum(
            (g_mu[i - 1, j - 1] * g_T for (g_mu, g_T) in zip(irrep, transformation)), start=sp.zeros(*shape)))
        return P_ij

    @staticmethod
    def is_faithful(rep: list[sp.MatrixBase]):
        for i, gi in enumerate(rep):
            for j, gj in enumerate(rep):
                if i != j and gi == gj:
                    return False
        return True


class GroupC1(BaseGroup):

    def __init__(self, is_double_group: bool = False):
        super().__init__(is_double_group=is_double_group)
        self.get_elements()
        self.get_irreducible_representations()
        self.get_characters()

    def _generate_elements(self):
        single_elems = ['E']
        if self.is_double_group:
            double_elems = self._generate_additional_elements('Ebar', single_elems)
            return double_elems
        return single_elems

    def _generate_irreducible_representations(self) -> list:

        g1 = [1]
        irreps = g1  # {'g1': g1}

        if self.is_double_group:
            irreps = self.get_double_group_irreps(irreps)

        return irreps

    @staticmethod
    def get_double_group_irreps(single_group_irreps: list) -> list:
        single_group_irreps = [ir * 2 for ir in single_group_irreps]
        double_irreps = single_group_irreps + [1, -1]
        return double_irreps


class GroupCi(BaseGroup):

    def __init__(self, is_double_group: bool = False):
        super().__init__(is_double_group=is_double_group)
        self.get_elements()
        self.get_irreducible_representations()
        self.get_characters()

    def _generate_elements(self):
        single_elems = ['E', 'I']
        if self.is_double_group:
            double_elems = self._generate_additional_elements('Ebar', single_elems)
            return double_elems
        return single_elems

    def _generate_irreducible_representations(self):
        irreps = [[1, 1], [1, -1]]
        if self.is_double_group:
            irreps = self.get_double_group_irreps(irreps)
        return irreps

    @staticmethod
    def get_double_group_irreps(single_group_irreps: list) -> list:
        single_group_irreps = [ir * 2 for ir in single_group_irreps]
        double_irreps = single_group_irreps + [[1, 1, -1, -1], [1, -1, -1, 1]]

        return double_irreps


class GroupC2Cs(BaseGroup):

    def __init__(self, name: str, is_double_group: bool = False):
        super().__init__(is_double_group=is_double_group)
        self.name = name
        self.get_elements()
        self.get_irreducible_representations()
        self.get_characters()

    def _generate_elements(self):
        single_elems = ['E']
        if self.name == 'C2':
            single_elems.append('C2')
        elif self.name == 'Cs':
            single_elems.append('sigma')

        if self.is_double_group:
            double_elems = self._generate_additional_elements('Ebar', single_elems)
            return double_elems
        return single_elems

    def _generate_irreducible_representations(self):
        irreps = [[1, 1], [1, -1]]
        if self.is_double_group:
            irreps = self.get_double_group_irreps(irreps)
        return irreps

    @staticmethod
    def get_double_group_irreps(single_group_irreps: list) -> list:
        single_group_irreps = [ir * 2 for ir in single_group_irreps]
        double_irreps = single_group_irreps + [[1, sp.I, -1, -sp.I], [1, -sp.I, -1, sp.I]]

        return double_irreps


class GroupD3C3v(BaseGroup):
    """
    D3

    Single group D3
    1    = e
    2    = C3
    3    = C3^2
    4    = C'2(1)
    5    = C'2(2)
    6    = C'2(3) (two fold about y axis)
    """

    def __init__(self, name: str, is_double_group: bool = False):
        super().__init__(is_double_group=is_double_group)

        self.name = name
        self.get_elements()
        self.get_irreducible_representations()
        self.get_characters()

    def _generate_elements(self):
        base_elements = ['E', 'C3', 'C3^2']
        if self.name == 'D3':
            elems = self._generate_additional_elements("C'2", base_elements)
            return elems
        elif self.name == 'C3v':
            elems = self._generate_additional_elements('sigma_v', base_elements)
            return elems

        if self.is_double_group:
            elems = self._generate_additional_elements('E_bar', base_elements)
            return elems

    def _generate_irreducible_representations(self) -> list:
        g1 = [1, 1, 1, 1, 1, 1]
        g2 = [1, 1, 1, -1, -1, -1]

        # Initialize g3 as a list of 2x2 matrices
        # {x, y} basis for D3h
        e = sp.eye(2)
        c3 = create_rotation_matrix(2 * sp.pi / 3)
        c2_p = create_reflection_matrix(sp.pi / 6)

        g3 = [e, c3, c3 ** 2, c2_p, c2_p * c3, c2_p * c3 ** 2]

        g3 = [sp.ImmutableMatrix(g) for g in g3]

        irreps = [
            g1,
            g2,
            g3
        ]
        if self.is_double_group:
            irreps = self.get_double_group_irreps(irreps)

        return irreps

    @staticmethod
    def get_double_group_irreps(single_group_irreps: list):

        j = 1 / 2
        su_n_rep = SU_N()
        e_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=0)
        c3_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi / 3)
        c2_p_aa = su_n_rep.axis_angle(direction=[sp.cos(sp.pi / 6), sp.sin(sp.pi / 6), 0], angle=sp.pi)

        axis_angle_reps = [e_aa, c3_aa, c2_p_aa]
        gen_su_n = su_n_rep.representation(j, axis_angle_representation=axis_angle_reps)
        e, c3, c2_p = tuple(gen_su_n)
        single_group_su_n_rep = [e, c3, c3 ** 2, c2_p, c2_p * c3, c2_p * c3 ** 2]

        e_bar_axis_angle = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_axis_angle)[0]

        double_group_su_n_rep = single_group_su_n_rep.copy()
        for sg in single_group_su_n_rep:
            double_group_su_n_rep.append(sp.expand_complex(e_bar * sg))

        g4 = double_group_su_n_rep.copy()
        g5 = [1, -1, -1, sp.I, sp.I, sp.I]
        g5 = g5 + [-1 * g for g in g5]

        g6 = [1, -1, -1, -sp.I, -sp.I, -sp.I]
        g6 = g6 + [-1 * g for g in g6]

        double_group_irreps = [ir * 2 for ir in single_group_irreps] + [g4, g5, g6]

        return double_group_irreps


class GroupD6C6vD3h(BaseGroup):
    """
    D6 = C6v = D3h
    D6 = D3 x C2
    C2: E, C_2 = two-fold about z-axis

    Single group D6
    1    = e
    2    = C3
    3    = C3^2
    4    = C'2(1)
    5    = C'2(2)
    6    = C'2(3) (two fold about y axis)
    7-12 = C2 * 1-6

    Double group
    13-24 = E_bar * 1-12

    Mapping D6 -> D3h
    C2 -> sigma_h
    """

    def __init__(self, name: str, is_double_group: bool = False):
        super().__init__(is_double_group=is_double_group)

        self.name = name
        self.get_elements()
        self.get_irreducible_representations()
        self.get_characters()
        self.get_faithful_representation()

    def _generate_elements(self):
        base_elements = ['E', 'C3', 'C3^2']
        elems = None
        if self.name == 'D6':
            d6_elems = self._generate_additional_elements("C'2", base_elements)
            elems = self._generate_additional_elements("C2", d6_elems)
        elif self.name == 'D3h':
            d3h_elems = self._generate_additional_elements("C'2", base_elements)
            elems = self._generate_additional_elements("sigma_h", d3h_elems)
        elif self.name == 'C6v':
            c6v_elems = self._generate_additional_elements('sigma_d', base_elements)
            elems = self._generate_additional_elements('C2', c6v_elems)

        if self.is_double_group:
            elems = self._generate_additional_elements('E_bar', elems)

        return elems

    def _generate_irreducible_representations(self) -> list:
        g1 = [1, 1, 1, 1, 1, 1]
        g2 = [1, 1, 1, -1, -1, -1]
        g3 = [1, 1, 1, 1, 1, 1]
        g4 = [1, 1, 1, -1, -1, -1]

        c2_1 = 1
        c2_2 = -1

        for e1, e2 in zip(g1.copy(), g2.copy()):
            g1.append(c2_1 * e1)
            g2.append(c2_1 * e2)
            g3.append(c2_2 * e1)
            g4.append(c2_2 * e2)

        # Initialize g5 as a list of 2x2 matrices
        # {x, y} basis for D3h
        e = sp.eye(2)
        c3 = create_rotation_matrix(2 * sp.pi / 3)
        c2_p = create_reflection_matrix(sp.pi / 6)

        g5 = [e, c3, c3 ** 2, c2_p, c2_p * c3, c2_p * c3 ** 2]

        # 180-degree rotation
        c2 = create_rotation_matrix(sp.pi)
        for e5 in g5.copy():
            g5.append(c2 * e5)

        g5 = [sp.ImmutableMatrix(g) for g in g5]

        g6 = tensor(g3, g5)

        irreps = [
            g1,
            g2,
            g3,
            g4,
            g5,
            g6
        ]
        if self.is_double_group:
            irreps = self.get_double_group_irreps(irreps)

        return irreps

    @staticmethod
    def get_double_group_irreps(single_group_irreps: list):

        j = 5 / 2
        su_n_rep = SU_N()
        e_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=0)
        c3_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=2 * sp.pi / 3)
        c2_p_aa = su_n_rep.axis_angle(direction=[sp.cos(sp.pi / 6), sp.sin(sp.pi / 6), 0], angle=sp.pi)
        c2_aa = su_n_rep.axis_angle(direction=[0, 0, 1], angle=sp.pi)

        axis_angle_reps = [e_aa, c3_aa, c2_p_aa, c2_aa]
        gen_su_n = su_n_rep.representation(j, axis_angle_representation=axis_angle_reps)
        e, c3, c2_p, c2 = tuple(gen_su_n)
        single_group_su_n_rep = [e, c3, c3 ** 2, c2_p, c2_p * c3, c2_p * c3 ** 2]
        single_group_su_n_rep += [c2 * g for g in single_group_su_n_rep]

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

    def _generate_faithful_representation(self):

        faithful_rep = None

        if self.is_double_group:

            if self.name == 'D6':
                faithful_rep = direct_sum(self.irreducible_representations[6], self.irreducible_representations[1])
            elif self.name == 'C6v':
                faithful_rep = direct_sum(self.irreducible_representations[6], self.irreducible_representations[0])
            elif self.name == 'D3h':
                faithful_rep = direct_sum(self.irreducible_representations[6], self.irreducible_representations[3])
        else:
            if self.name == 'D6':
                faithful_rep = direct_sum(self.irreducible_representations[4], self.irreducible_representations[1])
            elif self.name == 'C6v':
                faithful_rep = direct_sum(self.irreducible_representations[4], self.irreducible_representations[0])
            elif self.name == 'D3h':
                faithful_rep = direct_sum(self.irreducible_representations[4], self.irreducible_representations[3])

        if not self.is_faithful(faithful_rep):
            raise ValueError('Faithful representation defined incorrectly')

        return faithful_rep


class GroupD6h(BaseGroup):

    def __init__(self, name: str | None = None, is_double_group: bool = False):
        super().__init__(is_double_group=is_double_group)

        self.is_double_group = is_double_group
        self.d6 = GroupD6C6vD3h(name='D6', is_double_group=self.is_double_group)
        self.ci = GroupCi()

        self.name = name
        self.get_elements()
        self.get_irreducible_representations()
        self.get_characters()
        self.get_faithful_representation()

    def _generate_elements(self) -> list:
        elements = [''.join(pr) for pr in itertools.product(self.ci.elements, self.d6.elements)]
        return elements

    def _generate_irreducible_representations(self) -> list:

        additional = []
        for ci_ir in self.ci.irreducible_representations:
            for d6_ir in self.d6.irreducible_representations:
                additional_elements_in_ir = []
                for g_ci in ci_ir:
                    additional_elements_in_ir += [g_ci * g_d6 for g_d6 in d6_ir]
                additional.append(additional_elements_in_ir)
        return additional

    def _generate_faithful_representation(self):
        if self.is_double_group:
            faithful_rep = direct_sum(self.irreducible_representations[6], self.irreducible_representations[10])
        else:
            faithful_rep = direct_sum(self.irreducible_representations[11], self.irreducible_representations[9])

        if not self.is_faithful(faithful_rep):
            raise ValueError('Faithful representation defined incorrectly')
        return faithful_rep


class GroupProduct(BaseGroup):

    def __init__(self, group_1: BaseGroup, group_2: BaseGroup, is_double_group: bool = False):
        super().__init__(is_double_group=is_double_group)
        self.group_1 = group_1
        self.group_2 = group_2

        self.get_elements()
        self.get_irreducible_representations()
        self.get_characters()

    def _generate_elements(self) -> list:
        elements = [''.join(pr) for pr in itertools.product(self.group_1.elements, self.group_2.elements)]
        if self.group_1.is_double_group or self.group_2.is_double_group:
            elements = self._generate_additional_elements('E_bar', elements)
        return elements

    def _ir_products(self, ir1, ir2):

        prod_ir = []
        for g1 in ir1:
            if isinstance(g1, sp.MatrixBase):
                if g1.shape[0] > 1:
                    raise NotImplementedError(f'Irrep 1 dimension greater than 1 not implemented')
                else:
                    prod_ir.append([sp.KroneckerProduct(g1, g2).doit() for g2 in ir2])
            else:
                prod_ir.append([g1 * g2 for g2 in ir2])
        return prod_ir

    def _generate_irreducible_representations(self) -> list:
        """
        Not
        :return:
        """

        result = []
        for ir1 in self.group_1.irreducible_representations:
            for ir2 in self.group_2.irreducible_representations:
                ir_prod = self._ir_products(ir1, ir2)
                for irp in ir_prod:
                    result.append(irp)

        return result


if __name__ == '__main__':
    d6h = GroupD6h(is_double_group=True)
    ci = GroupCi(is_double_group=True)
    direct_sum(ci.irreducible_representations)
