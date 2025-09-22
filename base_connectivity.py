from representation_matrices_explicit import BaseGroup, CosetReps
from utility_functions import ismember, match_subgroup_to_group

import numpy as np


class BaseConnectivity:

    def __init__(self):
        self.group = None
        pass

    def subgroup_indices_in_full_group(self, subgroup: BaseGroup):
        full_group = self.group
        full_group_faithful_rep = full_group.faithful_representation
        sub_group_faithful_rep = subgroup.faithful_representation

        sub_group_index_in_full_group = [full_group_faithful_rep.index(x) for x in sub_group_faithful_rep]

        return sub_group_index_in_full_group

    def subgroup_indices_in_full_group_2(self, subgroup: BaseGroup):
        sub_group_index_in_full_group = list(match_subgroup_to_group(self.group, subgroup).values())
        return sub_group_index_in_full_group

    def group_to_subgroup_mapping(self, subgroup: BaseGroup, coset_reps: CosetReps,
                                  from_subgroup: BaseGroup | None = None, extended_form: bool = False):
        """
        RRcol = RrowRh where R in group and Rh in subgroup Rrow, Rcol are coset reps
        by default R is in Full group if from_subgroup_indices is specified then R is restricted to be in subgroup
        defined by from_subgroup_indices of full group

        :param extended_form: show full mapping form. R_row | R_col | idx_from | idx_to
        :param from_subgroup:
        :param subgroup:
        :param coset_reps:
        :return:
        """

        full_group = self.group
        full_group_faithful_rep = full_group.faithful_representation

        sub_group_index_in_full_group = self.subgroup_indices_in_full_group(subgroup)
        sub_group_in_full_group = [full_group_faithful_rep[i] for i in sub_group_index_in_full_group]

        if from_subgroup:
            from_subgroup_indices = self.subgroup_indices_in_full_group(from_subgroup)
            from_group_faithful_rep = [full_group_faithful_rep[i] for i in from_subgroup_indices]
        else:
            from_group_faithful_rep = full_group_faithful_rep.copy()
            from_subgroup_indices = self.subgroup_indices_in_full_group(self.group)
        # sub_group_faithful_rep = subgroup.faithful_representation

        coset_reps_f = coset_reps.faithful_representation

        left_coset_sub_group = [[full_group_faithful_rep.index(R_row * R_sg) for R_sg in sub_group_in_full_group] for
                                R_row in coset_reps_f]

        right_coset_full_group = [[full_group_faithful_rep.index(R * R_col) for R in from_group_faithful_rep] for R_col
                                  in coset_reps_f]

        coset_pairs_mapping, _ = ismember(right_coset_full_group, left_coset_sub_group)
        element_mapping = []
        for cp_i, rc_i in zip(coset_pairs_mapping, right_coset_full_group):
            el_map = [
                full_group_faithful_rep.index(coset_reps_f[ci].inv() * full_group_faithful_rep[RRcol]) for ci, RRcol in
                zip(cp_i, rc_i)
            ]
            element_mapping.append(el_map)

        if extended_form:
            mappings = [[(c_row, c_col, e0, e) for c_col, e0, e in
                         zip(cmi, from_subgroup_indices, emi)] for
                        c_row, (cmi, emi) in enumerate(zip(coset_pairs_mapping, element_mapping))]
            return np.array(mappings).reshape(-1, np.array(mappings).shape[-1])

        return coset_pairs_mapping, np.array(element_mapping)

    def subgroup_to_elements_mapping_in_full_group(self, subgroup: BaseGroup):
        indices = self.subgroup_indices_in_full_group(subgroup)
        elements = np.array(self.group.elements)[indices]
        return list(elements)

    def subgroup_to_faithful_rep_mapping_in_full_group(self, subgroup: BaseGroup):
        indices = self.subgroup_indices_in_full_group(subgroup)
        ff = [self.group.faithful_representation[i] for i in indices]
        return ff

    # --- helpers to get/set blocks ---

    def get_block(M, i, j, block_size):
        a = block_size
        return M[i * a:(i + 1) * a, j * a:(j + 1) * a]

    def set_block(self, M, i, j, block, block_size):
        a = block_size
        if block.shape != (a, a):
            raise ValueError(f"block must be {a}x{a}, got {block.shape}")
        M[i * a:(i + 1) * a, j * a:(j + 1) * a] = block

    def group_to_subgroup_mapping_2(
            self,
            subgroup: BaseGroup,
            coset_reps: CosetReps,
            from_subgroup: BaseGroup | None = None
    ):
        """
        For each element R in 'from_group_faithful_rep', find all triples
        (row_idx, col_idx, sg_idx) such that:

            R * R_col = R_sg * R_row

        where R_col, R_row are coset representatives and R_sg is a subgroup element.
        returns: {R_idx: [(row_idx, col_idx, sg_idx) ]}
        """

        # faithful reps
        full_group_faithful_rep = self.group.faithful_representation
        sub_group_in_full_group = subgroup.faithful_representation
        coset_reps_f = coset_reps.faithful_representation

        # Decide which subset of group elements to map from
        if from_subgroup:
            from_subgroup_indices = self.subgroup_indices_in_full_group(from_subgroup)
            from_group_faithful_rep = [full_group_faithful_rep[i] for i in from_subgroup_indices]
        else:
            from_group_faithful_rep = full_group_faithful_rep
            from_subgroup_indices = list(range(len(full_group_faithful_rep)))

        mapping = {}

        # loop through all chosen group elements
        for idx, R in zip(from_subgroup_indices, from_group_faithful_rep):
            matches = []
            # try all pairs (R_row, R_col)
            for row_idx, R_row in enumerate(coset_reps_f):
                for col_idx, R_col in enumerate(coset_reps_f):
                    lhs = R * R_col
                    # check against each subgroup element
                    for sg_idx, R_sg in enumerate(sub_group_in_full_group):
                        rhs = R_sg * R_row
                        if lhs == rhs:
                            matches.append((row_idx, col_idx, sg_idx))
            if matches:
                mapping[idx] = matches
            else:
                raise ValueError(f"No mapping found for group element index {idx}")

        return mapping

    def group_to_subgroup_mapping_3(
            self,
            subgroup: BaseGroup,
            coset_reps_subgroup: CosetReps,
            coset_reps_ssg: CosetReps,
            from_subgroup: BaseGroup | None = None
    ):
        """
        For each element R in 'from_group_faithful_rep', find all triples
        (row_idx, col_idx, sg_idx) such that:

            R * R_col = R_sg * R_row

        where R_col, R_row are coset representatives and R_sg is a subgroup element.
        returns: {R_idx: [(row_idx, col_idx, sg_idx) ]}
        """

        # faithful reps
        full_group_faithful_rep = []
        for R_tau in coset_reps_subgroup.faithful_representation:
            for R in subgroup.faithful_representation:
                full_group_faithful_rep.append(R_tau * R)
        sub_group_in_full_group = subgroup.faithful_representation
        coset_reps_f = coset_reps_ssg.faithful_representation

        # Decide which subset of group elements to map from
        if from_subgroup:
            from_subgroup_indices = self.subgroup_indices_in_full_group(from_subgroup)
            from_group_faithful_rep = [full_group_faithful_rep[i] for i in from_subgroup_indices]
        else:
            from_group_faithful_rep = full_group_faithful_rep
            from_subgroup_indices = list(range(len(full_group_faithful_rep)))

        mapping = {}

        # loop through all chosen group elements
        for idx, R in zip(from_subgroup_indices, from_group_faithful_rep):
            matches = []
            # try all pairs (R_row, R_col)
            for row_idx, R_row in enumerate(coset_reps_f):
                for col_idx, R_col in enumerate(coset_reps_f):
                    lhs = R * R_col
                    # check against each subgroup element
                    for sg_idx, R_sg in enumerate(sub_group_in_full_group):
                        rhs = R_sg * R_row
                        if lhs == rhs:
                            matches.append((row_idx, col_idx, sg_idx))
            if matches:
                mapping[idx] = matches
            else:
                raise ValueError(f"No mapping found for group element index {idx}")

        return mapping
