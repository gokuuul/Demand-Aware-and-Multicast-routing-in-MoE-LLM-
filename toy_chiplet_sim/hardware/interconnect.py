"""
2D mesh (and ring) interconnect: chiplets and XY / ring routing.
"""

from typing import List, Tuple

from ..config import SystemConfig
from .chiplet import Chiplet


class Interconnect:
    """Network of chiplets with mesh or ring topology and routing."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.chiplets: List[Chiplet] = []
        self.links: List[Tuple[int, int]] = []
        self._init_chiplets()
        self._setup_topology()

    def _init_chiplets(self) -> None:
        for i in range(self.config.x_chiplets):
            for j in range(self.config.y_chiplets):
                chiplet_id = j + i * self.config.y_chiplets
                new_chiplet = Chiplet(chiplet_id, self.config)
                new_chiplet.set_pos(x=i, y=j)
                self.chiplets.append(new_chiplet)
                assert chiplet_id + 1 == len(self.chiplets), "chiplet init error"

    def _chiplet_id_trans(self, xidx: int, yidx: int) -> int:
        return yidx + xidx * self.config.y_chiplets

    def _setup_topology(self) -> None:
        if self.config.interconnect == SystemConfig.InterconnectType.MESH:
            for i in range(self.config.x_chiplets):
                for j in range(self.config.y_chiplets):
                    chiplet_id = j + i * self.config.y_chiplets
                    if j < self.config.y_chiplets - 1:
                        self._connect(chiplet_id, chiplet_id + 1)
                        self.links.append(tuple(sorted([chiplet_id, chiplet_id + 1])))
                    if i < self.config.x_chiplets - 1:
                        self._connect(chiplet_id, chiplet_id + self.config.y_chiplets)
                        self.links.append(
                            tuple(sorted([chiplet_id, chiplet_id + self.config.y_chiplets]))
                        )
        # RING etc. can be added here

    def _connect(self, a: int, b: int) -> None:
        self.chiplets[a].add_neighbor(self.chiplets[b])
        self.chiplets[b].add_neighbor(self.chiplets[a])

    def route(self, src: Chiplet, dst: Chiplet) -> List[Chiplet]:
        if self.config.interconnect == SystemConfig.InterconnectType.MESH:
            return self._xy_routing(src, dst)
        elif self.config.interconnect == SystemConfig.InterconnectType.RING:
            return self._ring_routing(src, dst)
        return [src, dst]

    def multicast_tree(self, src: Chiplet, dsts) -> Tuple[set, int, set, set]:
        """Union of the routed paths from `src` to every destination.

        Deterministic routing from a fixed source gives each die exactly one
        path, so if two paths both touch a node they share the identical prefix
        src->node. The union therefore contains no cycle: it is a tree rooted at
        `src`, which is precisely the structure a destination-bitmap header
        produces when each router forwards one copy per distinct output port.

        Returns:
            links       undirected link ids, keyed as in ``self.links``
            depth       longest root-to-leaf hop count -- the latency term
            first_hops  distinct output ports at src -- supplier egress copies
            nodes       chiplet ids touched (|links| == |nodes| - 1 for a tree)
        """
        links, first_hops, nodes = set(), set(), {src.id}
        depth = 0
        for dst in dsts:
            path = self.route(src, dst)
            if len(path) < 2:
                continue
            if len(path) - 1 > depth:
                depth = len(path) - 1
            a, b = path[0].id, path[1].id
            first_hops.add((a, b) if a < b else (b, a))
            for i in range(len(path) - 1):
                a, b = path[i].id, path[i + 1].id
                links.add((a, b) if a < b else (b, a))
                nodes.add(b)
        return links, depth, first_hops, nodes

    def _xy_routing(self, src: Chiplet, dst: Chiplet) -> List[Chiplet]:
        path = []
        current_x, current_y = src.xidx, src.yidx
        path.append(src)
        while current_x < dst.xidx:
            current_x += 1
            path.append(self.chiplets[self._chiplet_id_trans(current_x, current_y)])
        while current_x > dst.xidx:
            current_x -= 1
            path.append(self.chiplets[self._chiplet_id_trans(current_x, current_y)])
        while current_y < dst.yidx:
            current_y += 1
            path.append(self.chiplets[self._chiplet_id_trans(current_x, current_y)])
        while current_y > dst.yidx:
            current_y -= 1
            path.append(self.chiplets[self._chiplet_id_trans(current_x, current_y)])
        return path

    def _ring_routing(self, src: Chiplet, dst: Chiplet) -> List[Chiplet]:
        path = []
        src_id, dst_id = src.id, dst.id
        if src_id <= dst_id:
            clockwise_dist = dst_id - src_id
            counter_dist = src_id + self.config.chiplet_num - dst_id
        else:
            clockwise_dist = self.config.chiplet_num - src_id + dst_id
            counter_dist = src_id - dst_id
        if clockwise_dist <= counter_dist:
            for i in range(clockwise_dist + 1):
                path.append(self.chiplets[(src_id + i) % self.config.chiplet_num])
        else:
            for i in range(counter_dist + 1):
                path.append(self.chiplets[(src_id - i) % self.config.chiplet_num])
        return path


