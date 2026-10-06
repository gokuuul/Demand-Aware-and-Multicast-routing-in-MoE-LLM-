"""
Address types for expert slices and chiplet-local addressing.
"""

from dataclasses import dataclass


@dataclass
class Addr_expert:
    """Logical address of an expert slice (expert, layer, matrix, slice)."""
    expert_id: int
    layer_id: int
    matrix_id: int  # 0 or 1 for two FFN matrices in MoE layer
    slice_id: int

    def __hash__(self):
        return hash((self.expert_id, self.layer_id, self.matrix_id, self.slice_id))


@dataclass
class Addr_chip:
    """Chiplet-local address."""
    chip_id: int
    addr_id: int
