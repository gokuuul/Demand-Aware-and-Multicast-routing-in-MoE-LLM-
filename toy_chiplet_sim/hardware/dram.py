"""
DRAM model per chiplet: stores expert slices and a bounded hardware-managed region.

Two regions share one address set:

* **Home region** -- slices this die owns under the static placement map, written
  once at init with ``is_hm_data=False``. Never evicted.
* **Hardware-managed region** -- replicas pulled in at runtime (prediction under
  ``PRED_NEXT_TOKEN``, or any other replication policy), written with
  ``is_hm_data=True``. Bounded by ``hardware_manage_size`` and evicted LRU.

The bound matters: without it replication is free and unbounded, which makes any
comparison against a replication baseline meaningless. Set ``ENFORCE_HM_CAPACITY``
False to restore the old unbounded behaviour.
"""

import math
from collections import OrderedDict
from typing import Set

from ..config import SystemConfig
from ..models import Addr_expert

# Bound the hardware-managed (replica) region and evict LRU when it is full.
ENFORCE_HM_CAPACITY = True


class DRAM:
    """Per-chiplet DRAM: set of expert-slice addresses plus a bounded LRU replica region."""

    def __init__(self, size: float, hardware_manage_size: float, config: SystemConfig):
        self.size = size
        self.hm_size = hardware_manage_size
        self.hm_free_size = hardware_manage_size
        self.data: Set[Addr_expert] = set()
        # LRU over the replica region only. OrderedDict, not deque: move_to_end is
        # O(1) whereas deque.remove is O(n), and this is touched on every replica
        # store and every DRAM access.
        self.hm_lru: "OrderedDict[Addr_expert, None]" = OrderedDict()
        self.hm_capacity_slices = max(1, math.floor(
            hardware_manage_size * (1024 ** 3) / config.expert_slice_size
        ))
        self.data_size = 0
        self._slice_size = config.expert_slice_size
        # Stats, for reporting how hard the bound bites.
        self.hm_evictions = 0
        self.hm_stores = 0

    @property
    def total_data_size_GB(self) -> float:
        return len(self.data) * self._slice_size / (1024 ** 3)

    @property
    def num_slices(self) -> int:
        return len(self.data)

    @property
    def hm_slices(self) -> int:
        return len(self.hm_lru)

    @property
    def hm_used_GB(self) -> float:
        return len(self.hm_lru) * self._slice_size / (1024 ** 3)

    def store(self, address: Addr_expert, data_size: float, is_hm_data: bool = False) -> None:
        """Add a slice. Replica stores (is_hm_data) are capacity-bounded and evict LRU."""
        self.data.add(address)
        if not (is_hm_data and ENFORCE_HM_CAPACITY):
            return

        self.hm_stores += 1
        if address in self.hm_lru:
            self.hm_lru.move_to_end(address)
            return
        # A slice this die already owns as home data is not a replica -- do not
        # charge it against the replica budget, and never make it evictable.
        self.hm_lru[address] = None
        self.hm_free_size -= data_size
        while len(self.hm_lru) > self.hm_capacity_slices:
            victim, _ = self.hm_lru.popitem(last=False)
            self.data.discard(victim)
            self.hm_free_size += self._slice_size
            self.hm_evictions += 1

    def access(self, address: Addr_expert) -> bool:
        """Record a read. Keeps the replica region's LRU order honest."""
        if ENFORCE_HM_CAPACITY and address in self.hm_lru:
            self.hm_lru.move_to_end(address)
        return address in self.data

    def check(self, address: Addr_expert) -> bool:
        return address in self.data

    def __str__(self) -> str:
        experts_dict = {}
        for address in self.data:
            experts_dict.setdefault(address.layer_id, set()).add(address.expert_id)
        return "".join(
            f"layer_id: {layer_id}, {experts}\n"
            for layer_id, experts in experts_dict.items()
        )

    def print_layer(self, prt_layer_id: int) -> None:
        experts_dict = {}
        for address in self.data:
            experts_dict.setdefault(address.layer_id, set()).add(address.expert_id)
        for layer_id, experts in experts_dict.items():
            if layer_id == prt_layer_id:
                print(f"layer_id: {layer_id}, {experts}")
