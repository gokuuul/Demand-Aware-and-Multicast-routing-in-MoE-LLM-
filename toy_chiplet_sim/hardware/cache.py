"""
Cache model: set-associative cache with LRU replacement per set.
"""

from collections import deque
from typing import Tuple

from ..models import Addr_expert


class LRU_Set:
    """One set of an set-associative cache with LRU replacement."""

    def __init__(self, assoc: int):
        self.ways = assoc
        self.lru_order = deque(maxlen=assoc)

    def access(self, tag: Addr_expert) -> bool:
        if tag in self.lru_order:
            self.lru_order.remove(tag)
            self.lru_order.append(tag)
            return True
        if len(self.lru_order) >= self.ways:
            self.lru_order.popleft()
        self.lru_order.append(tag)
        return False

    def store(self, tag: Addr_expert) -> None:
        if tag in self.lru_order:
            self.lru_order.remove(tag)
            self.lru_order.append(tag)
            return
        if len(self.lru_order) >= self.ways:
            self.lru_order.popleft()
        self.lru_order.append(tag)

    def check(self, tag: Addr_expert) -> bool:
        return tag in self.lru_order


class Cache:
    """Set-associative cache keyed by expert-slice address."""

    def __init__(self, size: int, assoc: int, line_size: float):
        self.size = size
        self.assoc = assoc
        self.line_size = line_size
        self.num_sets = max(1, int(size // (assoc * line_size)))
        self.sets = [LRU_Set(assoc) for _ in range(self.num_sets)]

    def parse_address(self, address: Addr_expert) -> Tuple[Addr_expert, int]:
        tag = address
        index = address.expert_id % self.num_sets
        return tag, index

    def access(self, address: Addr_expert) -> bool:
        tag, index = self.parse_address(address)
        return self.sets[index].access(tag)

    def store(self, address: Addr_expert) -> None:
        tag, index = self.parse_address(address)
        self.sets[index].store(tag)

    def check(self, address: Addr_expert) -> bool:
        tag, index = self.parse_address(address)
        return self.sets[index].check(tag)
