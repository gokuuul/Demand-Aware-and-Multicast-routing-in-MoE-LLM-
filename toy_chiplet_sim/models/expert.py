"""
Expert and expert-slice data structures for MoE layers.
"""

from typing import List

from ..config import SystemConfig


class Expert:
    """One logical expert in a MoE layer; can be sliced into Expert_slice units."""

    def __init__(self, config: SystemConfig, layer_id: int, expert_id: int):
        self.config = config
        self.layer_id = layer_id
        self.id = expert_id
        self.tot_slice_num = config.expert_tot_slice_num
        self.size = config.expert_size
        self.slice_list: List["Expert_slice"] = []

    def display_info(self):
        print(f"Expert ID: {self.id}, Size: {self.size}")

    def slice_expert(self):
        if self.config.whole_expert_as_one_slice:
            self.slice_list.append(
                Expert_slice(self.id, self.layer_id, matrix_id=0, slice_id=0)
            )
        else:
            for matrix_id in range(self.config.expert_matrix_num):
                for slice_id in range(self.config.expert_matrix_slice_num):
                    self.slice_list.append(
                        Expert_slice(self.id, self.layer_id, matrix_id, slice_id)
                    )

    def __str__(self):
        return f"Layer {self.layer_id} Expert {self.id} with {self.tot_slice_num} slices"


class Expert_slice:
    """One slice of an expert (one matrix/slice index)."""

    def __init__(self, expert_id: int, layer_id: int, matrix_id: int, slice_id: int):
        self.expert_id = expert_id
        self.layer_id = layer_id
        self.matrix_id = matrix_id
        self.slice_id = slice_id

    def __str__(self):
        return f"{self.slice_id}/{getattr(self, 'tot_slice_num', '?')}th slice of expert {self.expert_id}"
