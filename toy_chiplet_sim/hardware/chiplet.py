"""
Chiplet node: one die with local cache and DRAM.
"""

from collections import defaultdict

from ..config import SystemConfig
from .cache import Cache
from .dram import DRAM


class Chiplet:
    """One chiplet (die): cache, DRAM, position, neighbors, and request count."""

    def __init__(self, cid: int, config: SystemConfig):
        self.id = cid
        self.yidx = 0
        self.xidx = 0
        self.name = None
        self.cache = Cache(config.cache_size, config.cache_assoc, config.line_size)
        self.dram: DRAM = DRAM(
            config.dram_size_per_die,
            config.hardware_manage_dram_size_per_die,
            config,
        )
        self.neighbors = []
        self.comm_stats = defaultdict(int)
        self.req_num = 0

    def set_pos(self, x: int, y: int) -> None:
        self.xidx = x
        self.yidx = y

    def add_neighbor(self, node: "Chiplet") -> None:
        self.neighbors.append(node)

    def add_name(self, name: str) -> None:
        self.name = name
