"""
Resource manager: allocates/releases DRAM, cache port, D2D link, compute unit; manages wait queues.
"""

import heapq
from typing import Dict, List, Tuple

from .events import Resource, ResourceType, SimulationEvent, _ResQueueItem


class ResourceManager:
    """Tracks resources and their availability; enqueues events when resources are busy."""

    def __init__(self):
        self.resources: Dict[Tuple[ResourceType, object], Resource] = {}

    def register_resource(self, res_type: ResourceType, res_id: object) -> None:
        key = (res_type, res_id)
        self.resources[key] = Resource(res_type, res_id)

    def check_resource(self, event: SimulationEvent) -> bool:
        return all(r.available for r in event.resources)

    def request_resource(self, event: SimulationEvent) -> bool:
        can_allocate = all(r.available for r in event.resources)
        if can_allocate:
            for res in event.resources:
                res.available = False
                if event.time < res.available_time:
                    event.time = res.available_time
            event.admitted = True
            return True
        for res in event.resources:
            heapq.heappush(res.queue, _ResQueueItem(event))
        return False

    def add_to_wait_queue(self, event: SimulationEvent) -> None:
        for res in event.resources:
            heapq.heappush(res.queue, _ResQueueItem(event))

    def release_resource(
        self,
        res: Resource,
        event_queue: List[SimulationEvent],
        current_time: float,
    ) -> None:
        res.available = True
        if res.available_time < current_time:
            res.available_time = current_time

    @staticmethod
    def peek_waiter(res: Resource):
        """Highest-priority event still waiting on `res`, or None.

        Admitted events are not eagerly removed from the queues they were
        pushed onto (an event waits on every link of its route, so eager removal
        meant an O(n) scan plus an O(n) heapify per resource per admission --
        quadratic once contention is heavy). Instead their entries are discarded
        here, the only place the queue head is read.
        """
        q = res.queue
        while q:
            event = q[0].event
            if event.admitted:
                heapq.heappop(q)
                continue
            return event
        return None

    def add_to_event_queue(
        self,
        pending_event: SimulationEvent,
        event_queue: List[SimulationEvent],
        current_time: float,
    ) -> None:
        if pending_event.admitted:
            return  # already claimed via another resource's queue this round
        if self.check_resource(pending_event):
            self.request_resource(pending_event)  # sets admitted; stale entries
                                                  # are skipped by peek_waiter
            if pending_event.time < current_time:
                pending_event.time = current_time
            heapq.heappush(event_queue, pending_event)
