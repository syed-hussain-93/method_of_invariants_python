import sympy as sp
import numpy as np
from utility_functions import (
    create_reflection_matrix, create_rotation_matrix, tensor, character_of_representation, irrep_decomposition,
    direct_sum, is_faithful, rotation_matrix_3D
)

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class GroupValidation:
    ok: bool
    order: int
    identity_index: Optional[int]
    inverse_of: List[Optional[int]]
    cayley: List[List[Optional[int]]]
    issues: List[str]


class Generators:
    axis: list | tuple | np.ndarray | None
    angle: Any
    matrix: sp.Matrix | None

    def __init__(self, axis=None, angle=None, name=None, matrix=None):
        self.axis = axis
        self.angle = angle
        self.matrix = matrix
        if self.matrix is None:
            self.matrix = rotation_matrix_3D(self.axis, self.angle)
            self.axis_angle = SU_N.axis_angle(self.axis, self.angle)
        self.name = name  # + str([tuple(axis)]).replace(',', '')


def _mat_eq(A: sp.Matrix, B: sp.Matrix, tol: Optional[float]) -> bool:
    """Robust matrix equality: symbolic exact by default; numeric with tolerance if tol is given."""
    if tol is None:
        eq = A.equals(B)
        if eq is None:
            eq = (A - B).applyfunc(sp.simplify).is_zero_matrix
        return bool(eq)
    D = (A - B).evalf()
    return max(abs(complex(val)) for val in D) <= tol


def validate_group_axioms_from_faithful_rep(rep: List[sp.Matrix],
                                            tol: Optional[float] = None,
                                            check_associativity: bool = False) -> GroupValidation:
    """
    Validate group axioms for the set of matrices `rep` under multiplication.

    Args:
        rep: list of distinct matrices (faithful representation; one per group element).
        tol: if not None, treat numbers within `tol` as equal (useful for float-y reps).
        check_associativity: if True, brute-force check associativity via the Cayley table.

    Returns:
        GroupValidation with details (Cayley table as indices, inverse map, issues if any).
    """
    issues = []
    n = len(rep)
    if n == 0:
        return GroupValidation(False, 0, None, [], [], ["empty representation"])

    # Ensure all are square and invertible (representation must land in GL(d))
    d = rep[0].rows
    for i, G in enumerate(rep):
        if G.rows != G.cols:
            issues.append(f"element {i} is not square")
        if G.rows != d:
            issues.append(f"element {i} has different dimension ({G.rows} vs {d})")
        det = sp.simplify(G.det())
        if tol is None:
            if det == 0:
                issues.append(f"element {i} is not invertible (det=0)")
        else:
            if abs(complex(det.evalf())) <= tol:
                issues.append(f"element {i} is near-singular (|det|<=tol)")

    # Duplicates (faithfulness)
    for i in range(n):
        for j in range(i + 1, n):
            if _mat_eq(rep[i], rep[j], tol):
                issues.append(f"duplicate matrices at indices {i} and {j} (not faithful)")

    # Helper: find index of a product within rep
    def index_of(M: sp.Matrix) -> Optional[int]:
        for k, H in enumerate(rep):
            if _mat_eq(M, H, tol):
                return k
        return None

    # Find identity element (either the literal I or the unique one that acts like it)
    I = sp.eye(d)
    identity_idx = None
    # Try literal identity first
    for i, G in enumerate(rep):
        if _mat_eq(G, I, tol):
            identity_idx = i
            break
    # If not literally there, detect by action
    if identity_idx is None:
        for i, E in enumerate(rep):
            if all(_mat_eq(E * G, G, tol) and _mat_eq(G * E, G, tol) for G in rep):
                identity_idx = i
                break
    if identity_idx is None:
        issues.append("no identity element found")

    # Build Cayley table and check closure
    cayley: List[List[Optional[int]]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            idx = index_of(rep[i] * rep[j])
            cayley[i][j] = idx
            if idx is None:
                issues.append(f"closure fails: element {i} * {j} not in set")

    # Inverses
    inverse_of: List[Optional[int]] = [None] * n
    if identity_idx is not None:
        for i in range(n):
            inv_found = None
            for j in range(n):
                if cayley[i][j] == identity_idx and cayley[j][i] == identity_idx:
                    inv_found = j
                    break
            inverse_of[i] = inv_found
            if inv_found is None:
                issues.append(f"no inverse found for element {i}")

    # Optional: associativity via table (matrix mult is associative, but check if you like)
    if check_associativity:
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    ab = cayley[a][b]
                    bc = cayley[b][c]
                    if ab is None or bc is None:
                        continue
                    left = cayley[ab][c]
                    right = cayley[a][bc]
                    if left != right:
                        issues.append(f"associativity fails at ({a},{b},{c})")
                        # One counterexample is enough to fail
                        a = b = c = n  # break all loops
                        break

    ok = len(issues) == 0
    return GroupValidation(ok, n, identity_idx, inverse_of, cayley, issues)


class BaseGroup(ABC):

    def __init__(self):
        self.elements = None
        self.irreducible_representations = None
        self.is_double_group = None
        self.faithful_representation = None
        self.xyz_representation = None

    def _elements(self):
        pass

    def _irreducible_representations(self):
        pass

    def _faithful_representation(self):
        pass

    # def faithful_representation(self):
    #     """Return the faithful representation depending on self.is_double."""
    #     if self.is_double_group:
    #         if not self.faithful_irreps_double:
    #             raise ValueError("Faithful irreps for double not set.")
    #         irreps = self.faithful_irreps_double
    #     else:
    #         if not self.faithful_irreps_single:
    #             raise ValueError("Faithful irreps for single not set.")
    #         irreps = self.faithful_irreps_single
    #
    #     reps = [self.irreducible_representations[name] for name in irreps]
    #     rep = direct_sum(*reps) if len(reps) > 1 else reps[0]
    #
    #     if not is_faithful(rep):
    #         raise ValueError("Representation not faithful")
    #
    #     return rep

    def get_xyz_representation(self):
        pass

    def character_table(self):
        ch_table = {}
        for name, ir in self.irreducible_representations.items():
            ch_table[name] = character_of_representation(ir)
        return ch_table

    def conjugacy_classes(self, matrices: bool = False):
        """
        G: list of sympy matrices forming a finite group (faithful rep; includes identity).
        Returns: list of lists, each inner list is a conjugacy class (as matrices, deduped).
        """
        G = self.faithful_representation

        def _canon_key(M):
            # Put entries into a hashable, simplified normal form
            M = sp.ImmutableMatrix(M.applyfunc(sp.cancel))
            return tuple(M)

        # Precompute inverses and a lookup from matrix -> index for O(1) membership checks
        inv = [g.inv() for g in G]
        key_to_idx = {_canon_key(g): i for i, g in enumerate(G)}

        remaining = set(range(len(G)))
        classes = []

        while remaining:
            i = remaining.pop()
            h = G[i]
            orbit_keys = set()
            orbit_indices = set()

            for j, g in enumerate(G):
                c = g * h * inv[j]
                k = key_to_idx.get(_canon_key(c))
                if k is not None:
                    orbit_indices.add(k)
                    orbit_keys.add(_canon_key(G[k]))

            # finalize this class
            classes.append([G[k] for k in sorted(orbit_indices)])
            remaining -= orbit_indices
        if not matrices:
            classes = [[np.array(self.elements)[self.faithful_representation.index(R)] for R in conj_class] for
                       conj_class in classes]
        return classes

    def spinor_xyz(self, irrep_names: list[str]):
        irreps = [self.irreducible_representations[ir] for ir in irrep_names]
        representation = direct_sum(*irreps) if len(irreps) > 1 else irreps[0]
        if is_faithful(representation):
            return representation
        else:
            raise ValueError('Not faithful representation')

    def get_spinor_representation(self):
        pass

    def validate(self, tol: Optional[float] = None, check_associativity: bool = False) -> GroupValidation:
        rep = self.faithful_representation
        return validate_group_axioms_from_faithful_rep(rep, tol=tol, check_associativity=check_associativity)


class GroupOfK:

    def __init__(self, name: str, point: sp.Matrix, group: BaseGroup):
        self.name = name
        self.group = group
        self.point = point


class CosetReps:
    symbols: list[str]
    full_group: BaseGroup

    def __init__(self, symbols: list[str], full_group: BaseGroup):
        self.symbols = symbols
        self.full_group = full_group
        self.indices = [self.full_group.elements.index(x) for x in self.symbols]
        # [i for i, x in enumerate(full_group.elements) if x in self.symbols]
        self.faithful_representation = [self.full_group.faithful_representation[i] for i in self.indices]
        self.xyz_representation = [self.full_group.xyz_representation[i] for i in self.indices]


class SiteSymmetryGroup:

    def __init__(self, wyckoff_position: str, group: BaseGroup, coset_reps: CosetReps, tau_vectors: dict):
        self.wyckoff_position = wyckoff_position
        self.group = group
        self.coset_reps = coset_reps


class GroupGenerator:
    def __init__(self, generators, identity=None, N=100, names=None):
        """
        generators: list of sympy matrices (faithful rep)
        identity: identity matrix (default: eye of correct size)
        N: maximum order
        names: optional list of generator names (strings)
        """
        self.generators = generators
        if identity is None:
            n = generators[0].shape[0]
            identity = sp.eye(n)
        self.identity = identity
        self.N = N

        # Default names
        if names is None:
            self.names = [f"g{i}" for i in range(len(generators))]
        else:
            self.names = names

        # Build group elements and store words (as integer sequences)
        self.elements, self.words = self._build_elements()
        self.element_names = [self.word_str(i) for i in range(len(self.elements))]

    def _build_elements(self):
        e = self.identity
        G = self.generators

        L = [e]
        words = [()]  # () means identity

        # first generator
        g = g1 = G[0]
        w = (0,)
        while not g.equals(e):
            L.append(g)
            words.append(w)
            assert len(L) <= self.N
            g = g * g1
            w = w + (0,)

        # remaining generators
        for i in range(1, len(G)):
            C = [e]
            C_words = [()]
            L1 = list(L)
            W1 = list(words)
            more = True
            while more:
                assert len(L) <= self.N
                more = False
                for g, w in list(zip(C, C_words)):
                    for j, s in enumerate(G[:i + 1]):
                        sg = s * g
                        if not any(sg.equals(x) for x in L):
                            C.append(sg)
                            new_word = (j,) + w
                            C_words.append(new_word)
                            for x, w1 in zip(L1, W1):
                                L.append(sg * x)
                                words.append(new_word + w1)
                            more = True
        return L, words

    def apply_irrep(self, irrep_generators):
        """
        irrep_generators: list of matrices/scalars corresponding to self.generators
        Returns: list of matrices in the same order as self.elements
        """
        # Normalize to matrices (wrap 1D reps / scalars)
        gens = []
        for g in irrep_generators:
            if isinstance(g, sp.MatrixBase):
                gens.append(g)
            else:
                gens.append(sp.ImmutableMatrix([[sp.sympify(g)]]))

        n = gens[0].shape[0]
        reps = []
        for word in self.words:
            if word == ():
                reps.append(sp.eye(n))
            else:
                M = sp.eye(n)
                # word stores LEFT multiplications; apply in reverse to build product
                for gen_idx in reversed(word):
                    M = gens[gen_idx] * M
                reps.append(M)
        return reps

    def word_str(self, idx):
        """
        Return the word for element at position idx in generator name format.
        Example: (0,1,0) -> "g0 g1 g0"
        """
        word = self.words[idx]
        if word == ():
            return "E"
        return " * ".join(self.names[i] for i in word)

    def generator_idx_in_elements(self, gen_idx: int | tuple | list | None = None):
        found = []
        if not gen_idx:
            gen_idx = [i for i in range(len(self.generators))]
        elif isinstance(gen_idx, int):
            gen_idx = [gen_idx]

        for gen_id in gen_idx:
            for idx, w in enumerate(self.words):
                if w == (gen_id,):
                    found.append(idx)
                    break

        return found


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
            phase = sp.cancel(jx * aa[0] + jy * aa[1] + jz * aa[2])
            elem = sp.exp(-sp.I * phase)
            reps.append(elem)
        return reps

    @staticmethod
    def axis_angle(direction: list, angle):
        axis = sp.Matrix(direction)
        unit_axis = axis / axis.norm()
        # return sp.simplify(angle * unit_axis)
        return list(sp.Matrix(direction) / sp.Matrix(direction).norm() * angle)


class GroupD6D3hC6v(BaseGroup):

    def __init__(self, name: str, generators: list[Generators], is_double_group: bool = False):
        """
        Matrices are all defined using D6. Supplying name ensures faithful and xyz rep are correct
        :param name:
        :param is_double_group:
        """

        super().__init__()
        self.name = name
        self.is_double_group = is_double_group
        self.generators = generators

        # self._e = Generators(axis=(0, 0, 1), angle=0, name='E')
        # self._r = Generators(axis=(0, 0, 1), angle=2 * sp.pi / 6, name='C6([001])')
        # self._s = Generators(axis=(0, 1, 0), angle=2 * sp.pi / 2, name='C2([010])')
        # if self.edge_x_orientation:
        #     self._s = Generators(axis=(1, 0, 0), angle=2 * sp.pi / 2, name='C2([100])')

        # self.generators: list[Generators] = [self._r, self._s]
        self.generators_faithful = [R.matrix for R in self.generators]

        self.group_generator = GroupGenerator(
            generators=self.generators_faithful,
            names=[R.name + str((list(R.axis))).replace(',', '').replace(' ', '') for R in self.generators]
        )

        self.elements = self.group_generator.element_names
        if self.is_double_group:
            self.elements += [f'E_bar * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()
        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):

        r1, s1 = 1, 1
        r2, s2 = 1, -1
        r3, s3 = -1, 1
        r4, s4 = -1, -1
        r5, s5 = [R.matrix[:2, :2] for R in self.generators]

        ir1 = self.group_generator.apply_irrep([r1, s1])
        ir2 = self.group_generator.apply_irrep([r2, s2])
        ir3 = self.group_generator.apply_irrep([r3, s3])
        ir4 = self.group_generator.apply_irrep([r4, s4])
        ir5 = self.group_generator.apply_irrep([r5, s5])
        ir6 = tensor(ir5, ir3)

        irrep_names = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6']
        irreps = [ir1, ir2, ir3, ir4, ir5, ir6]
        if self.is_double_group:
            irrep_names += ['g7', 'g8', 'g9']
            irreps = [ir * 2 for ir in irreps] + self.get_double_group_irreps()

        d6_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}
        return d6_irreps

    def get_su_n_rep(self, j):
        su_n_rep = SU_N()
        axis_angle_reps_generators = [su_n_rep.axis_angle(R.axis, R.angle) for R in self.generators]
        generators_su_n = su_n_rep.representation(j=j, axis_angle_representation=axis_angle_reps_generators)
        e_bar = su_n_rep.representation(j=j, axis_angle_representation=su_n_rep.axis_angle([0, 0, 1], 2 * sp.pi))[0]
        single_group_su_n_rep = self.group_generator.apply_irrep(generators_su_n)
        double_group_su_n_rep = list(single_group_su_n_rep) + [e_bar * R for R in single_group_su_n_rep]
        return double_group_su_n_rep

    def get_double_group_irreps(self):
        j = 5 / 2
        double_group_su_n_rep = self.get_su_n_rep(j)
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

        double_group_irreps = [g7, g8, g9]
        return double_group_irreps

    def get_xyz_representation(self):
        if self.name == 'D6':
            M = sp.eye(3)
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g2'])
        elif self.name == 'C6v':
            M = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g1'])
        elif self.name == 'D3h':
            M = sp.eye(3)
            rep = direct_sum(self.irreducible_representations['g6'], self.irreducible_representations['g4'])
        else:
            raise ValueError('name not provided')
        rep = [M.inv() * R * M for R in rep]
        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g7']
        return None


class GroupD6h(BaseGroup):

    def __init__(self, generators: list[Generators], is_double_group: bool = False):
        super().__init__()
        self.is_double_group = is_double_group
        self.generators = generators
        d6_generators = [gen for gen in self.generators if gen.name != 'I']
        self.group_d6 = GroupD6D3hC6v(name='D6', generators=d6_generators, is_double_group=is_double_group)

        # self._e = Generators(axis=(0, 0, 1), angle=0, name='E')
        # self._r = Generators(axis=(0, 0, 1), angle=2 * sp.pi / 6, name='C6([001])')
        # self._s = Generators(axis=(0, 1, 0), angle=2 * sp.pi / 2, name='C2([010])')
        # self._i = Generators(matrix=sp.Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), name='I')

        # self.generators: list[Generators] = self.group_d6_single.generators + [self._i]

        # self.group_generator = GroupGenerator(
        #     generators=[R.matrix for R in self.generators],
        #     identity=sp.Matrix.eye(3),
        #     names=[R.name for R in self.generators]
        # )

        self.elements = self.group_d6.elements
        if self.is_double_group:
            self.elements += [f'I * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()
        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):

        inversion = [1, -1]
        inv_sign = ['p', 'm']

        irrep_names = []
        irreps = []
        for inv, sign in zip(inversion, inv_sign):
            for ir_name, ir in self.group_d6.irreducible_representations.items():
                irrep = ir + [inv * R for R in ir]
                irrep_name = ir_name + sign
                irreps.append(irrep)
                irrep_names.append(irrep_name)

        d6h_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}

        return d6h_irreps

    def _faithful_representation(self):
        if self.is_double_group:
            self.faithful_irreps_double = ['g7p', 'g2m']
        else:
            self.faithful_irreps_single = ['g5m', 'g2m']
        rep = self.faithful_representation()
        return rep

    def get_xyz_representation(self):
        rep = direct_sum(self.irreducible_representations['g5m'], self.irreducible_representations['g2m'])
        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g7p']
        return None


class GroupD2C2v(BaseGroup):

    def __init__(self, name: str, generators: list[Generators], is_double_group: bool = False):
        super().__init__()
        self.name = name
        self.is_double_group = is_double_group
        self.generators = generators

        # self._e = Generators(axis=(0, 0, 1), angle=0, name='E')
        # self._r = Generators(axis=(0, 0, 1), angle=2 * sp.pi / 1, name='C2([001])')
        # self._s = Generators(axis=(0, 1, 0), angle=2 * sp.pi / 2, name='C2([010])')
        #
        # self.generators: list[Generators] = [self._r, self._s]
        self.generators_faithful = [R.matrix for R in self.generators]

        self.group_generator = GroupGenerator(
            generators=self.generators_faithful,
            names=[R.name for R in self.generators]
        )

        self.elements = self.group_generator.element_names
        if self.is_double_group:
            self.elements += [f'E_bar * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()
        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):
        r1, s1 = 1, 1
        r2, s2 = -1, 1
        r3, s3 = 1, -1
        r4, s4 = -1, -1

        g1 = self.group_generator.apply_irrep([r1, s1])
        g2 = self.group_generator.apply_irrep([r2, s2])
        g3 = self.group_generator.apply_irrep([r3, s3])
        g4 = self.group_generator.apply_irrep([r4, s4])

        irrep_names = ['g1', 'g2', 'g3', 'g4']
        irreps = [g1, g2, g3, g4]
        if self.is_double_group:
            irrep_names += ['g5']
            irreps = self.get_double_group_irreps(irreps)

        d2_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}

        return d2_irreps

    def get_SU_N_rep(self, j):
        su_n_rep = SU_N()

        # Precompute axis-angle reps only once
        r_aa, s_aa = [R.axis_angle for R in self.generators]

        # Get representation matrices
        r, s = su_n_rep.representation(j, axis_angle_representation=[r_aa, s_aa])
        r, s = (sp.cancel(gen.expand(complex=True)) for gen in (r, s))

        # e_bar representation
        e_bar_aa = su_n_rep.axis_angle([0, 0, 1], 2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_aa)[0]

        # Build subgroup
        single_group_su_n_rep = self.group_generator.apply_irrep([r, s])

        # Avoid copying then appending repeatedly (expensive)
        gd12 = single_group_su_n_rep + [e_bar * sg for sg in single_group_su_n_rep]

        # Post-process simplification only once
        gj = [sp.simplify(R) for R in gd12]

        return gj

    def get_double_group_irreps(self, single_group_irreps: list):

        j = sp.Rational(1, 2)
        g5 = self.get_SU_N_rep(j=j)

        single_group_doubled = [ir * 2 for ir in single_group_irreps]

        double_group_su_n_rep = [g5]

        double_group_irreps = single_group_doubled + double_group_su_n_rep

        return double_group_irreps

    def _faithful_representation(self):
        if self.is_double_group:
            self.faithful_irreps_double = ['g5', 'g3']
        else:
            self.faithful_irreps_single = ['g4', 'g2', 'g3']
        rep = self.faithful_representation()
        return rep

    def get_xyz_representation(self):
        if self.name == 'D2':
            rep = direct_sum(
                self.irreducible_representations['g4'],
                self.irreducible_representations['g2'],
                self.irreducible_representations['g3']
            )
        elif self.name == 'C2v':
            rep = direct_sum(
                self.irreducible_representations['g2'],
                self.irreducible_representations['g4'],
                self.irreducible_representations['g1']
            )
        else:
            raise ValueError('name not provided')

        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g5']
        return None


class GroupD2h(BaseGroup):

    def __init__(self, generators: list[Generators], is_double_group: bool = False):
        super().__init__()
        self.is_double_group = is_double_group
        self.generators = generators
        d2_generators = [gen for gen in self.generators if gen.name != 'I']
        self.group_d2 = GroupD2C2v(name='D2', generators=d2_generators, is_double_group=is_double_group)

        self.elements = self.group_d2.elements
        if self.is_double_group:
            self.elements += [f'I * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()
        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):

        inversion = [1, -1]
        inv_sign = ['p', 'm']

        irrep_names = []
        irreps = []
        for inv, sign in zip(inversion, inv_sign):
            for ir_name, ir in self.group_d2.irreducible_representations.items():
                irrep = ir + [inv * R for R in ir]
                irrep_name = ir_name + sign
                irreps.append(irrep)
                irrep_names.append(irrep_name)

        d2h_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}

        return d2h_irreps

    def get_xyz_representation(self):
        rep = direct_sum(
            self.irreducible_representations['g4m'],
            self.irreducible_representations['g2m'],
            self.irreducible_representations['g3m']
        )
        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g5p']
        return None


class GroupTdO(BaseGroup):

    def __init__(self, name: str, generators: list[Generators], is_double_group: bool):
        super().__init__()
        self.name = name
        self.is_double_group = is_double_group
        self.generators = generators

        # self._e = Generators(axis=(0, 0, 1), angle=0, name='E')
        # self._r = Generators(axis=(0, 0, 1), angle=2 * sp.pi / 4, name='C4([001])')
        # self._s = Generators(axis=(1, 1, 1), angle=2 * sp.pi / 3, name='C3([111])')
        #
        # self.generators: list[Generators] = [self._r, self._s]
        self.generators_faithful = [R.matrix for R in self.generators]

        self.group_generator = GroupGenerator(
            generators=self.generators_faithful,
            names=[R.name for R in self.generators]
        )

        self.elements = self.group_generator.element_names
        if self.is_double_group:
            self.elements += [f'E_bar * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()
        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):
        r1, s1 = 1, 1
        r2, s2 = -1, 1
        r3 = sp.Matrix([[1, 0], [0, -1]])
        s3 = sp.Matrix([
            [-sp.Rational(1, 2), -sp.sqrt(3) / 2],
            [sp.sqrt(3) / 2, -sp.Rational(1, 2)]
        ])

        r4, s4 = (R.matrix for R in self.generators)

        g1 = self.group_generator.apply_irrep([r1, s1])
        g2 = self.group_generator.apply_irrep([r2, s2])
        g3 = self.group_generator.apply_irrep([r3, s3])
        g4 = self.group_generator.apply_irrep([r4, s4])

        g5 = tensor(g4, g2)

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

        # Precompute axis-angle reps only once
        r_aa, s_aa = tuple(R.axis_angle for R in self.generators)

        # Get representation matrices
        r, s = su_n_rep.representation(j, axis_angle_representation=[r_aa, s_aa])
        r, s = (sp.cancel(gen.expand(complex=True)) for gen in (r, s))  # cheaper than expand+cancel

        # e_bar representation
        e_bar_aa = su_n_rep.axis_angle([0, 0, 1], 2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_aa)[0]

        # Build subgroup
        single_group_su_n_rep = self.group_generator.apply_irrep([r, s])

        # Avoid copying then appending repeatedly (expensive)
        g6 = single_group_su_n_rep + [e_bar * sg for sg in single_group_su_n_rep]

        # Post-process simplification only once
        g6 = [sp.cancel(R) for R in g6]

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

    def _faithful_representation(self):
        if self.is_double_group:
            if self.name == 'Td':
                self.faithful_irreps_double = ['g6', 'g2']
            elif self.name == 'O':
                self.faithful_irreps_double = ['g6', 'g2']
        else:
            if self.name == 'Td':
                self.faithful_irreps_single = ['g5', 'g1']
            elif self.name == 'O':
                self.faithful_irreps_single = ['g4', 'g2']

        rep = self.faithful_representation()
        return rep

    def get_xyz_representation(self):
        if self.name == 'Td':
            rep = self.irreducible_representations['g5']
        elif self.name == 'O':
            rep = self.irreducible_representations['g4']
        else:
            raise ValueError('name not provided')

        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g6']
        return None


class GroupD4C4vD2d(BaseGroup):

    def __init__(self, name: str, generators: list[Generators], is_double_group: bool):
        super().__init__()
        self.name = name
        self.is_double_group = is_double_group
        self.generators = generators

        # self._e = Generators(axis=(0, 0, 1), angle=0, name='E')
        # self._r = Generators(axis=(0, 0, 1), angle=2 * sp.pi / 4, name='C4([001])')
        # self._s = Generators(axis=(0, 1, 0), angle=2 * sp.pi / 2, name='C2([010])')
        # self.generators: list[Generators] = [self._r, self._s]

        self.generators_faithful = [R.matrix for R in self.generators]

        self.group_generator = GroupGenerator(
            generators=self.generators_faithful,
            names=[R.name for R in self.generators]
        )

        self.elements = self.group_generator.element_names
        if self.is_double_group:
            self.elements += [f'E_bar * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()
        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):
        r1, s1 = 1, 1
        r2, s2 = 1, -1
        r3, s3 = -1, 1
        r4, s4 = -1, -1

        r5 = self._r.matrix[0:2, 0:2]
        s5 = self._s.matrix[0:2, 0:2]

        g1 = self.group_generator.apply_irrep([r1, s1])
        g2 = self.group_generator.apply_irrep([r2, s2])
        g3 = self.group_generator.apply_irrep([r3, s3])
        g4 = self.group_generator.apply_irrep([r4, s4])

        g5 = self.group_generator.apply_irrep([r5, s5])

        irrep_names = ['g1', 'g2', 'g3', 'g4', 'g5']
        irreps = [g1, g2, g3, g4, g5]

        if self.is_double_group:
            irrep_names += ['g6', 'g7']
            irreps = self.get_double_group_irreps(irreps)

        irreps = [[R.expand(complex=True) if isinstance(R, sp.Matrix) else R for R in ir] for ir in irreps]
        full_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}
        return full_irreps

    def get_SU_N_rep(self, j):
        su_n_rep = SU_N()

        # Precompute axis-angle reps only once
        r_aa, s_aa = (R.axis_angle for R in self.generators)

        # Get representation matrices
        r, s = su_n_rep.representation(j, axis_angle_representation=[r_aa, s_aa])
        r, s = (sp.cancel(gen.expand(complex=True)) for gen in (r, s))

        # e_bar representation
        e_bar_aa = su_n_rep.axis_angle([0, 0, 1], 2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_aa)[0]

        # Build subgroup
        single_group_su_n_rep = self.group_generator.apply_irrep([r, s])

        # Avoid copying then appending repeatedly (expensive)
        gd12 = single_group_su_n_rep + [e_bar * sg for sg in single_group_su_n_rep]

        # Post-process simplification only once
        gj = [sp.simplify(R) for R in gd12]

        return gj

    def get_double_group_irreps(self, single_group_irreps: list):

        j = sp.Rational(1, 2)
        g6 = self.get_SU_N_rep(j=j)

        single_group_doubled = [ir * 2 for ir in single_group_irreps]
        g7 = tensor(g6, single_group_doubled[2])
        double_group_su_n_rep = [g6, g7]

        double_group_irreps = single_group_doubled + double_group_su_n_rep

        return double_group_irreps

    def _faithful_representation(self):
        if self.is_double_group:
            if self.name == 'D4':
                self.faithful_irreps_double = ['g6', 'g2']
            elif self.name == 'C4v':
                self.faithful_irreps_double = ['g6', 'g1']
            elif self.name == 'D2d':
                self.faithful_irreps_double = ['g6', 'g4']
        else:
            if self.name == 'D4':
                self.faithful_irreps_single = ['g5', 'g2']
            elif self.name == 'C4v':
                self.faithful_irreps_single = ['g5', 'g1']
            elif self.name == 'D2d':
                self.faithful_irreps_single = ['g5', 'g4']

        rep = self.faithful_representation()
        return rep

    def get_xyz_representation(self):
        if self.name == 'D4':
            M = sp.eye(3)
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g2'])
        elif self.name == 'C4v':
            M = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g1'])
        elif self.name == 'D2d':
            M = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
            rep = direct_sum(self.irreducible_representations['g5'], self.irreducible_representations['g4'])
        else:
            raise ValueError('name not provided')

        rep = [M.inv() * R * M for R in rep]
        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g6']
        return None


class GroupD3C3v(BaseGroup):

    def __init__(self, name: str, generators: list[Generators], is_double_group: bool):
        super().__init__()
        self.name = name
        self.generators = generators
        self.is_double_group = is_double_group

        # self._e = Generators(axis=(0, 0, 1), angle=0, name='E')
        # self._r = Generators(axis=(0, 0, 1), angle=2 * sp.pi / 3, name='C3([001])')
        # self._s = Generators(axis=(0, 1, 0), angle=2 * sp.pi / 2, name='C2([010])')
        # if self.edge_x_orientation:
        #     self._s = Generators(axis=(1, 0, 0), angle=2 * sp.pi / 2, name='C2([100])')
        # self.generators: list[Generators] = [self._r, self._s]

        self.generators_faithful = [R.matrix for R in self.generators]

        self.group_generator = GroupGenerator(
            generators=self.generators_faithful,
            names=[R.name for R in self.generators]
        )

        self.elements = self.group_generator.element_names
        if self.is_double_group:
            self.elements += [f'E_bar * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()
        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):
        r1, s1 = 1, 1
        r2, s2 = 1, -1

        r3, s3 = (R.matrix[0:2, 0:2] for R in self.generators)

        g1 = self.group_generator.apply_irrep([r1, s1])
        g2 = self.group_generator.apply_irrep([r2, s2])

        g3 = self.group_generator.apply_irrep([r3, s3])

        irrep_names = ['g1', 'g2', 'g3']
        irreps = [g1, g2, g3]

        if self.is_double_group:
            irrep_names += ['g4', 'g5', 'g6']
            irreps = self.get_double_group_irreps(irreps)

        irreps = [[R.expand(complex=True) if isinstance(R, sp.Matrix) else R for R in ir] for ir in irreps]
        full_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}
        return full_irreps

    def get_SU_N_rep(self, j):
        su_n_rep = SU_N()

        # Precompute axis-angle reps only once
        r_aa, s_aa = (R.axis_angle for R in self.generators)

        # Get representation matrices
        r, s = su_n_rep.representation(j, axis_angle_representation=[r_aa, s_aa])
        r, s = (sp.cancel(gen.expand(complex=True)) for gen in (r, s))

        # e_bar representation
        e_bar_aa = su_n_rep.axis_angle([0, 0, 1], 2 * sp.pi)
        e_bar = su_n_rep.representation(j, e_bar_aa)[0]

        # Build subgroup
        single_group_su_n_rep = self.group_generator.apply_irrep([r, s])

        # Avoid copying then appending repeatedly (expensive)
        gd12 = single_group_su_n_rep + [e_bar * sg for sg in single_group_su_n_rep]

        # Post-process simplification only once
        gj = [sp.simplify(R) for R in gd12]

        return gj

    def get_double_group_irreps(self, single_group_irreps: list):

        j = sp.Rational(1, 2)
        g4 = self.get_SU_N_rep(j=j)

        single_group_doubled = [ir * 2 for ir in single_group_irreps]

        su_j32 = self.get_SU_N_rep(j=3 / 2)
        Md32 = sp.Matrix([[1, sp.I, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [sp.I, 1, 0, 0]])
        Mp32 = sp.Matrix([[sp.I, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, sp.I, 0, 0]])

        g5 = [sp.cancel(Md32.inv() * R * Md32)[0, 0] for R in su_j32]
        g6 = [sp.cancel(Md32.inv() * R * Md32)[1, 1] for R in su_j32]

        double_group_su_n_rep = [g4, g5, g6]

        double_group_irreps = single_group_doubled + double_group_su_n_rep

        return double_group_irreps

    def _faithful_representation(self):
        if self.is_double_group:
            if self.name == 'D3':
                self.faithful_irreps_double = ['g4', 'g2']
            elif self.name == 'C3v':
                self.faithful_irreps_double = ['g4', 'g1']
        else:
            if self.name == 'D3':
                self.faithful_irreps_single = ['g3', 'g2']
            elif self.name == 'C3v':
                self.faithful_irreps_single = ['g3', 'g1']

        rep = self.faithful_representation()
        return rep

    def get_xyz_representation(self):
        if self.name == 'D3':
            M = sp.eye(3)
            rep = direct_sum(
                self.irreducible_representations['g3'],
                self.irreducible_representations['g2']
            )
        elif self.name == 'C3v':
            M = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            rep = direct_sum(
                self.irreducible_representations['g3'],
                self.irreducible_representations['g1']
            )
        else:
            raise ValueError('name not provided')
        rep = [M.inv() * R * M for R in rep]
        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g4']
        return None


class GroupOh(BaseGroup):

    def __init__(self, generators: list[Generators], is_double_group: bool = False):
        super().__init__()
        self.is_double_group = is_double_group
        self.generators = generators
        O_generators = [gen for gen in self.generators if gen.name != 'I']
        self.group_O = GroupTdO(name='O', generators=O_generators, is_double_group=is_double_group)

        self.elements = self.group_O.elements
        if self.is_double_group:
            self.elements += [f'I * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()

        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):

        inversion = [1, -1]
        inv_sign = ['p', 'm']

        irrep_names = []
        irreps = []
        for inv, sign in zip(inversion, inv_sign):
            for ir_name, ir in self.group_O.irreducible_representations.items():
                irrep = ir + [inv * R for R in ir]
                irrep_name = ir_name + sign
                irreps.append(irrep)
                irrep_names.append(irrep_name)

        Oh_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}
        return Oh_irreps

    def get_xyz_representation(self):
        rep = self.irreducible_representations['g5m']
        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g6p']
        return None


class GroupD4h(BaseGroup):

    def __init__(self, generators: list[Generators], is_double_group: bool = False):
        super().__init__()
        self.is_double_group = is_double_group
        self.generators = generators
        D4_generators = [gen for gen in self.generators if gen.name != 'I']

        self.group_d4 = GroupD4C4vD2d(name='D4', generators=D4_generators, is_double_group=is_double_group)

        self.elements = self.group_d4.elements
        if self.is_double_group:
            self.elements += [f'I * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()
        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):

        inversion = [1, -1]
        inv_sign = ['p', 'm']

        irrep_names = []
        irreps = []
        for inv, sign in zip(inversion, inv_sign):
            for ir_name, ir in self.group_d4.irreducible_representations.items():
                irrep = ir + [inv * R for R in ir]
                irrep_name = ir_name + sign
                irreps.append(irrep)
                irrep_names.append(irrep_name)

        d4h_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}
        return d4h_irreps

    def get_xyz_representation(self):
        rep = direct_sum(
            self.irreducible_representations['g5m'],
            self.irreducible_representations['g2m']
        )
        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g6p']
        return None


class GroupD3d(BaseGroup):

    def __init__(self, generators: list[Generators], is_double_group: bool = False):
        super().__init__()
        self.is_double_group = is_double_group
        self.generators = generators
        D3_generators = [gen for gen in self.generators if gen.name != 'I']
        self.group_d3 = GroupD3C3v(name='D3', generators=D3_generators, is_double_group=is_double_group)

        self.elements = self.group_d3.elements
        if self.is_double_group:
            self.elements += [f'I * {e}' for e in self.elements]
        self.elements = [R.replace(' * E', '').replace(' E * ', '') for R in self.elements]

        self.irreducible_representations = self._irreducible_representations()

        self.xyz_representation = self.get_xyz_representation()
        self.spinor_representation = self.get_spinor_representation()
        self.faithful_representation = self.xyz_representation.copy()
        if self.spinor_representation:
            self.faithful_representation = direct_sum(self.xyz_representation, self.spinor_representation)

        assert is_faithful(self.faithful_representation)

    def _irreducible_representations(self):

        inversion = [1, -1]
        inv_sign = ['p', 'm']

        irrep_names = []
        irreps = []
        for inv, sign in zip(inversion, inv_sign):
            for ir_name, ir in self.group_d3.irreducible_representations.items():
                irrep = ir + [inv * R for R in ir]
                irrep_name = ir_name + sign
                irreps.append(irrep)
                irrep_names.append(irrep_name)

        d3d_irreps = {name: ir for name, ir in zip(irrep_names, irreps)}
        return d3d_irreps

    def get_xyz_representation(self):
        rep = direct_sum(self.irreducible_representations['g3m'], self.irreducible_representations['g2m'])
        return rep

    def get_spinor_representation(self):
        if self.is_double_group:
            return self.irreducible_representations['g4p']
        return None


if __name__ == '__main__':
    Td = GroupD6h(is_double_group=True)
    Td.validate()
    g6g6 = tensor(Td.irreducible_representations['g6'], Td.irreducible_representations['g6'])
    irrep_decomposition(
        g6g6,
        Td.irreducible_representations
    )
    # print(d6h.irreducible_representations)
    # print(tensor(d6h.irreducible_representations, d6h.irreducible_representations))
