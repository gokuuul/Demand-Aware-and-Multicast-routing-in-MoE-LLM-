"""
Discrete-event simulation: event types, resource types, and event/resource dataclasses.
"""

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union

from ..models import Addr_expert


class EventType(Enum):
    CACHE_ACCESS = 1
    NETWORK_TRANSFER = 2
    MEMORY_ACCESS = 3
    COMPLETION_SIGNAL = 4
    DRAM_STORE_COMPLETION_SIGNAL = 11
    CACHE_STORE = 5
    DRAM_STORE = 6
    COMPUTATION = 7
    NET_TRANS_REQ = 8
    NET_TRANS_DATA = 9
    ATTENTION = 10


class ResourceType(Enum):
    NOTHING = 0
    CACHE_PORT = 1
    D2D_LINK = 2
    DRAM_PACKAGE = 3
    COMPUTE_UNIT = 4


class _ResQueueItem:
    """Wrapper for res.queue: orders by higher priority, then earlier time, then
    arrival order.

    The arrival-order tiebreak makes the ordering *total*. Without it, events
    with equal priority and time compared equal and their relative order was
    whatever the heap happened to produce -- which meant the queue order was
    undefined and could differ between otherwise-equivalent implementations.
    FIFO among equals is both deterministic and the natural queueing policy.
    """

    __slots__ = ("event", "seq")
    _counter = itertools.count()

    def __init__(self, event: "SimulationEvent"):
        self.event = event
        self.seq = next(_ResQueueItem._counter)

    def __lt__(self, other: "_ResQueueItem") -> bool:
        return ((-self.event.priority, self.event.time, self.seq)
                < (-other.event.priority, other.event.time, other.seq))


@dataclass
class Resource:
    type: ResourceType
    id: Union[int, tuple]
    available: bool = True
    available_time: float = 0
    # queue stores _ResQueueItem(event) so pop order is: higher priority first, then earlier time
    queue: List["_ResQueueItem"] = field(default_factory=list)


@dataclass(order=True)
class SimulationEvent:
    time: float
    batch_size: int = field(compare=False)
    event_type: EventType = field(compare=False)
    chiplet_id: int = field(compare=False)
    address: Addr_expert = field(compare=False)
    resources: List[Resource] = field(compare=False)
    callback: callable = field(compare=False)
    # Used only for resource wait queues (res.queue): higher priority popped first, then earlier time
    priority: float = field(default=0.0, compare=False)
    # True once the event has claimed its resources. Wait-queue entries are
    # removed lazily, so a queue may still hold stale entries for an admitted
    # event; readers skip them by checking this flag.
    admitted: bool = field(default=False, compare=False)

    def __eq__(self, other):
        if not isinstance(other, SimulationEvent):
            return NotImplemented
        return (
            self.event_type == other.event_type
            and self.address == other.address
            and self.chiplet_id == other.chiplet_id
        )

    def __hash__(self):
        return hash((self.event_type, self.address))
