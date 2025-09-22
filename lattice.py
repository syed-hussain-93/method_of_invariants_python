import sympy as sp


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


class DiamondCubicLattice(Lattice):

    def _get_a1(self):
        a1 = (self.a / 2) * sp.Matrix([0, 1, 1])
        return a1

    def _get_a2(self):
        a2 = (self.a / 2) * sp.Matrix([1, 0, 1])
        return a2

    def _get_a3(self):
        return (self.a / 2) * sp.Matrix([1, 1, 0])
