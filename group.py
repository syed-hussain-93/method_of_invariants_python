import sympy as sp
import numpy as np
from itertools import product
from collections import defaultdict


class GroupRepresentation:
    def __init__(self, generators=None, dimension=2):
        """
        Initialize a group representation with given generators.

        Parameters:
        -----------
        generators : list of sympy matrices
            The generating elements of the group
        dimension : int
            The dimension of the representation matrices
        """
        self.dimension = dimension
        self.identity = sp.eye(dimension)
        self.generators = generators if generators else []
        self.elements = []
        self.multiplication_table = None
        self.irreps = []

    def add_generator(self, generator):
        """Add a generator to the group."""
        if generator.shape != (self.dimension, self.dimension):
            raise ValueError(f"Generator must be a {self.dimension}x{self.dimension} matrix")
        self.generators.append(generator)

    def create_symbolic_generator(self, name):
        """Create a symbolic generator with parameters."""
        symbols = sp.symbols(f'{name}_{1:{self.dimension ** 2 + 1}}')
        matrix = sp.Matrix(self.dimension, self.dimension, symbols)
        return matrix

    def group_operation(self, a, b):
        """Define the group operation (matrix multiplication)."""
        return a * b

    def generate_elements(self, max_depth=10):
        """
        Generate all group elements by applying generators up to a certain depth.
        Uses breadth-first generation to find all elements.

        Parameters:
        -----------
        max_depth : int
            Maximum number of generator applications
        """
        if not self.generators:
            raise ValueError("No generators defined")

        # Start with identity
        self.elements = [self.identity]
        new_elements = [self.identity]

        # Generate elements level by level to avoid excessive depth
        for _ in range(max_depth):
            level_elements = []

            for element in new_elements:
                for generator in self.generators:
                    # Right multiplication
                    new_right = self.group_operation(element, generator)
                    if not self._is_in_elements(new_right):
                        level_elements.append(new_right)
                        self.elements.append(new_right)

                    # Left multiplication
                    new_left = self.group_operation(generator, element)
                    if not self._is_in_elements(new_left):
                        level_elements.append(new_left)
                        self.elements.append(new_left)

            # If no new elements were added, we've found all group elements
            if not level_elements:
                break

            new_elements = level_elements

        print(f"Found {len(self.elements)} group elements")
        return self.elements

    def _is_in_elements(self, matrix, tolerance=1e-10):
        """Check if a matrix is already in the list of elements."""
        for element in self.elements:
            if self._matrices_equal(element, matrix, tolerance):
                return True
        return False

    def _matrices_equal(self, A, B, tolerance=1e-10):
        """Check if two matrices are equal within a tolerance."""
        diff = A - B
        return sp.simplify(diff.norm()) < tolerance

    def create_multiplication_table(self):
        """Create the multiplication table for the group elements."""
        n = len(self.elements)
        self.multiplication_table = [[None for _ in range(n)] for _ in range(n)]

        for i, a in enumerate(self.elements):
            for j, b in enumerate(self.elements):
                product = self.group_operation(a, b)
                for k, c in enumerate(self.elements):
                    if self._matrices_equal(product, c):
                        self.multiplication_table[i][j] = k
                        break

        return self.multiplication_table

    def verify_group_axioms(self):
        """Verify that the elements satisfy group axioms."""
        if not self.elements:
            self.generate_elements()

        if not self.multiplication_table:
            self.create_multiplication_table()

        n = len(self.elements)

        # Check closure (already ensured by construction)

        # Check associativity
        for i, j, k in product(range(n), repeat=3):
            a, b, c = self.elements[i], self.elements[j], self.elements[k]
            if not self._matrices_equal((a * b) * c, a * (b * c)):
                return False, "Associativity failed"

        # Check identity
        identity_index = None
        for i, e in enumerate(self.elements):
            if self._matrices_equal(e, self.identity):
                identity_index = i
                break

        if identity_index is None:
            return False, "Identity element not found"

        # Check inverses
        for i, a in enumerate(self.elements):
            has_inverse = False
            for j, b in enumerate(self.elements):
                if self._matrices_equal(a * b, self.identity) and self._matrices_equal(b * a, self.identity):
                    has_inverse = True
                    break
            if not has_inverse:
                return False, f"Inverse not found for element {i}"

        return True, "All group axioms verified"

    def get_character_table(self):
        """
        Compute the character table of the representation.
        Character of an element is the trace of its representation matrix.
        """
        if not self.elements:
            self.generate_elements()

        characters = [sp.trace(element) for element in self.elements]
        return characters

    def find_conjugacy_classes(self):
        """Find the conjugacy classes of the group."""
        if not self.elements:
            self.generate_elements()

        n = len(self.elements)
        # Initialize each element in its own class
        classes = [[i] for i in range(n)]

        # Merge classes of conjugate elements
        i = 0
        while i < len(classes):
            j = i + 1
            while j < len(classes):
                merged = False
                for a_idx in classes[i]:
                    for b_idx in classes[j]:
                        a, b = self.elements[a_idx], self.elements[b_idx]
                        # Check if b is conjugate to a
                        for g in self.elements:
                            if sp.det(g) != 0 and self._matrices_equal(g * a * g.inv(), b):
                                # Merge classes
                                classes[i].extend(classes[j])
                                classes.pop(j)
                                merged = True
                                break
                        if merged:
                            break
                    if merged:
                        break
                if not merged:
                    j += 1
            i += 1

        return classes

    def find_irreducible_representations(self):
        """
        Find the irreducible representations of the group.
        Uses character theory to decompose the current representation.
        """
        if not self.elements:
            self.generate_elements()

        # Get conjugacy classes
        conjugacy_classes = self.find_conjugacy_classes()
        num_conjugacy_classes = len(conjugacy_classes)

        # For a finite group, the number of irreducible representations
        # equals the number of conjugacy classes
        print(
            f"Group has {num_conjugacy_classes} conjugacy classes, so we expect {num_conjugacy_classes} irreducible representations")

        # Get character of current representation
        chars = self.get_character_table()

        # Use class equation and dimensions of irreps
        group_order = len(self.elements)

        # For simple groups, we can use specific techniques to find irreps
        irreps = self._find_irreps_by_construction()

        if not irreps:
            print("Cannot automatically construct all irreducible representations.")
            print("Consider using more specific methods for this group.")
            return []

        self.irreps = irreps
        return irreps

    def _find_irreps_by_construction(self):
        """
        Find irreducible representations using construction techniques.
        This implementation focuses on common groups like cyclic, dihedral, etc.
        """
        if not self.elements:
            self.generate_elements()

        group_order = len(self.elements)

        # Determine if it's a cyclic group
        is_cyclic = self._check_if_cyclic()

        if is_cyclic:
            print("Detected cyclic group")
            return self._construct_cyclic_group_irreps()

        # Determine if it's a dihedral group
        is_dihedral = self._check_if_dihedral()

        if is_dihedral:
            print("Detected dihedral group")
            return self._construct_dihedral_group_irreps()

        # Add more group types as needed

        # For other groups, we need more sophisticated methods
        # Try to use character projections when possible
        return self._construct_irreps_by_projection()

    def _check_if_cyclic(self):
        """Check if the group is cyclic."""
        if not self.elements:
            self.generate_elements()

        if not self.multiplication_table:
            self.create_multiplication_table()

        # A group is cyclic if it has an element that generates the entire group
        for i, element in enumerate(self.elements):
            # Check if element generates the group
            generated = {i}
            power = i

            for _ in range(len(self.elements) - 1):
                # Calculate next power
                power = self.multiplication_table[power][i]
                generated.add(power)

            if len(generated) == len(self.elements):
                return True

        return False

    def _check_if_dihedral(self):
        """
        Check if the group has dihedral structure.
        Dihedral group D_n has order 2n and is generated by two elements:
        a rotation r of order n and a reflection s of order 2, with srs=r^(-1).
        """
        if not self.elements:
            self.generate_elements()

        if not self.multiplication_table:
            self.create_multiplication_table()

        group_order = len(self.elements)

        # Check if order is even (dihedral groups have order 2n)
        if group_order % 2 != 0:
            return False

        # Try to find a rotation element of order n=group_order/2
        n = group_order // 2

        for i, element in enumerate(self.elements):
            # Check element order
            order = 1
            power = i

            for _ in range(group_order):
                power = self.multiplication_table[power][i]
                order += 1
                if self._matrices_equal(self.elements[power], self.identity):
                    break

            if order == n:
                # Found potential rotation element, now look for reflection
                for j, s_element in enumerate(self.elements):
                    if s_element != self.identity and self._matrices_equal(s_element * s_element, self.identity):
                        # Check relation srs=r^(-1)
                        rotation = self.elements[i]
                        reflection = s_element

                        if self._matrices_equal(reflection * rotation * reflection, rotation.inv()):
                            return True

        return False

    def _construct_cyclic_group_irreps(self):
        """Construct irreducible representations for a cyclic group Cn."""
        group_order = len(self.elements)

        # Find a generator
        generator_idx = None
        for i, element in enumerate(self.elements):
            # Check if element generates the group
            generated = {i}
            power = i

            for _ in range(group_order - 1):
                # Calculate next power
                if self.multiplication_table:
                    power = self.multiplication_table[power][i]
                else:
                    power = (power * i) % group_order  # Simplified for cyclic groups
                generated.add(power)

            if len(generated) == group_order:
                generator_idx = i
                break

        if generator_idx is None:
            return []

        # For cyclic group, all irreps are 1-dimensional
        irreps = []
        generator = self.elements[generator_idx]

        # Construct the irreducible representations
        for k in range(group_order):
            irr_rep = []

            # For each group element, calculate its character in this irrep
            for i, element in enumerate(self.elements):
                # Find the power of the generator that equals this element
                power = 0
                temp = self.identity

                for p in range(group_order):
                    if self._matrices_equal(temp, element):
                        power = p
                        break
                    temp = temp * generator

                # Character is exp(2πi·k·power/n)
                char = sp.exp(2 * sp.pi * sp.I * k * power / group_order)
                irr_rep.append(sp.Matrix([[char]]))

            irreps.append(irr_rep)

        return irreps

    def _construct_dihedral_group_irreps(self):
        """Construct irreducible representations for a dihedral group Dn."""
        group_order = len(self.elements)
        n = group_order // 2  # D_n has order 2n

        irreps = []

        # Find rotation and reflection generators
        rotation = None
        reflection = None

        for element in self.elements:
            # Check for rotation of order n
            if not rotation:
                temp = element
                order = 1
                while not self._matrices_equal(temp, self.identity) and order <= n:
                    temp = temp * element
                    order += 1
                if order == n:
                    rotation = element

            # Check for reflection of order 2
            if not reflection and not self._matrices_equal(element, self.identity):
                if self._matrices_equal(element * element, self.identity):
                    reflection = element

            if rotation and reflection:
                break

        if not rotation or not reflection:
            return []

        # Dihedral group D_n has:
        # - 2 one-dimensional irreps if n is odd
        # - 4 one-dimensional irreps if n is even
        # - (n-1)/2 two-dimensional irreps if n is odd
        # - (n-2)/2 two-dimensional irreps if n is even

        # One-dimensional representations
        # Trivial representation
        trivial_rep = [sp.Matrix([[1]]) for _ in self.elements]
        irreps.append(trivial_rep)

        # Sign representation (determinant)
        sign_rep = []
        for element in self.elements:
            # Check if it's a rotation or reflection
            is_reflection = False
            temp = element
            for i in range(n):
                if self._matrices_equal(temp, reflection):
                    is_reflection = True
                    break
                temp = temp * rotation

            sign_rep.append(sp.Matrix([[1 if not is_reflection else -1]]))
        irreps.append(sign_rep)

        # If n is even, add two more one-dimensional irreps
        if n % 2 == 0:
            rep3 = []
            rep4 = []
            for element in self.elements:
                # Determine power of rotation
                rotation_power = 0
                temp = self.identity
                while not self._matrices_equal(temp, element) and rotation_power < n:
                    temp = temp * rotation
                    rotation_power += 1

                # Check if it involves reflection
                is_reflection = False
                if rotation_power == n:  # Not a pure rotation
                    is_reflection = True
                    rotation_power = 0
                    temp = reflection
                    while not self._matrices_equal(temp, element) and rotation_power < n:
                        temp = temp * rotation
                        rotation_power += 1

                # Characters depend on parity of rotation power
                is_even_power = (rotation_power % 2 == 0)

                rep3.append(sp.Matrix([[1 if is_even_power else -1]]))
                rep4.append(sp.Matrix([[1 if is_even_power and not is_reflection else -1]]))

            irreps.append(rep3)
            irreps.append(rep4)

        # Two-dimensional irreps
        num_2d_irreps = (n - 1) // 2 if n % 2 == 1 else (n - 2) // 2

        for k in range(1, num_2d_irreps + 1):
            irrep_2d = []
            for element in self.elements:
                # Determine power of rotation
                rotation_power = 0
                temp = self.identity
                while not self._matrices_equal(temp, element) and rotation_power < n:
                    temp = temp * rotation
                    rotation_power += 1

                # Check if it involves reflection
                is_reflection = False
                if rotation_power == n:  # Not a pure rotation
                    is_reflection = True
                    rotation_power = 0
                    temp = reflection
                    while not self._matrices_equal(temp, element) and rotation_power < n:
                        temp = temp * rotation
                        rotation_power += 1

                # Matrix depends on whether it's a rotation or reflection
                theta = 2 * sp.pi * k * rotation_power / n
                if not is_reflection:
                    # Rotation matrix
                    matrix = sp.Matrix([
                        [sp.cos(theta), -sp.sin(theta)],
                        [sp.sin(theta), sp.cos(theta)]
                    ])
                else:
                    # Reflection matrix (depends on implementation)
                    matrix = sp.Matrix([
                        [sp.cos(theta), sp.sin(theta)],
                        [sp.sin(theta), -sp.cos(theta)]
                    ])

                irrep_2d.append(matrix)

            irreps.append(irrep_2d)

        return irreps

    def _construct_irreps_by_projection(self):
        """
        Attempt to construct irreducible representations using projection operators.
        This is a more general method but may not always succeed.
        """
        if not self.elements:
            self.generate_elements()

        # For simplicity, we'll just return an empty list here
        # In a full implementation, this would use character theory and projection operators
        print("Projection method for constructing irreps not fully implemented.")
        return []

    def decompose_representation(self):
        """
        Decompose the current representation into irreducible components.
        Uses character inner products.
        """
        if not self.elements:
            self.generate_elements()

        if not self.irreps:
            self.find_irreducible_representations()

        if not self.irreps:
            return None

        # Get character of current representation
        current_chars = self.get_character_table()

        # Calculate multiplicity of each irrep in the decomposition
        decomposition = {}
        group_order = len(self.elements)

        for i, irrep in enumerate(self.irreps):
            irrep_chars = [sp.trace(matrix) for matrix in irrep]

            # Calculate inner product of characters
            inner_product = 0
            for j, element in enumerate(self.elements):
                inner_product += current_chars[j] * sp.conjugate(irrep_chars[j])

            multiplicity = inner_product / group_order

            if abs(multiplicity) > 1e-10:
                decomposition[i] = multiplicity

        return decomposition

    def direct_sum_irreps(self, irreps_indices):
        """
        Create a representation that is the direct sum of specified irreps.

        Parameters:
        -----------
        irreps_indices : list of int
            Indices of irreducible representations to include
        """
        if not self.irreps:
            self.find_irreducible_representations()

        if not self.irreps:
            return None

        # Create direct sum representation
        direct_sum_rep = []

        for element_idx, element in enumerate(self.elements):
            blocks = []
            for irrep_idx in irreps_indices:
                if irrep_idx < len(self.irreps):
                    blocks.append(self.irreps[irrep_idx][element_idx])

            # Create block diagonal matrix
            direct_sum = sp.diag(*blocks)
            direct_sum_rep.append(direct_sum)

        return direct_sum_rep


# Example usage for the dihedral group D4
def create_dihedral_group(n=4):
    """Create the dihedral group D_n."""
    # D_n is generated by a rotation r and a reflection s
    # with relations r^n = s^2 = 1 and sr = r^(n-1)s

    # Create 2D representation
    theta = 2 * sp.pi / n

    # Rotation matrix
    r = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)],
        [sp.sin(theta), sp.cos(theta)]
    ])

    # Reflection matrix
    s = sp.Matrix([
        [1, 0],
        [0, -1]
    ])

    group = GroupRepresentation(generators=[r, s], dimension=2)
    return group


# Example usage for cyclic groups
def create_cyclic_group(n=4):
    """Create the cyclic group C_n."""
    # C_n is generated by a single rotation r with relation r^n = 1

    # Create 2D representation
    theta = 2 * sp.pi / n

    # Rotation matrix
    r = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)],
        [sp.sin(theta), sp.cos(theta)]
    ])

    group = GroupRepresentation(generators=[r], dimension=2)
    return group
