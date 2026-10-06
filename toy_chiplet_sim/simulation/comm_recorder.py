"""
D2D communication recorder.

The event loop already computes, for every remote expert-slice access, the route,
the message size and the link occupancy time -- then discards all of it except a
single scalar (``stats['hop']``). This keeps it.

Metrics are ordered for the multicast study, where the bottleneck is **supplier
egress**: a die holding a popular expert must push one copy per requester through
its 2-4 outgoing links, and no routing policy can relieve that. So per-die egress
is first-class here, alongside per-link occupancy and the pair matrix.

``record`` sits directly in the event loop, so accumulators are flat Python lists
(a list ``+=`` beats numpy scalar indexing by roughly an order of magnitude) and
every route is resolved once and cached. Numpy views are built on demand for the
exporters. Overhead measured at ~15% of event-loop time; set
``COMM_TRACE_ENABLED = False`` in simulator.py to disable entirely.
"""

import csv
from pathlib import Path

import numpy as np

KIND_REQ = 0       # 16-byte read request, requester -> supplier
KIND_DATA = 1      # expert slice payload from remote DRAM
KIND_DATA_LLC = 2  # expert slice payload served from the supplier's LLC

# A remote LLC hit still crosses the network, so it is link traffic -- but it is
# not a DRAM read. Tracking it separately keeps the recorder reconcilable against
# remote_dram_read_size, and measures the "accidental multicast" the LLC already
# provides when many dies request the same slice in one layer.


class CommRecorder:
    """Aggregates every D2D transfer the simulator issues."""

    def __init__(self, config, interconnect, enabled=True):
        self.enabled = enabled
        self.config = config
        self.interconnect = interconnect
        n = config.chiplet_num
        self.n_dies = n
        nn = n * n

        self.links = list(dict.fromkeys(interconnect.links))
        self.n_links = len(self.links)
        self._link_index = {l: i for i, l in enumerate(self.links)}
        chips = interconnect.chiplets
        self.link_axis = ["x" if chips[a].xidx != chips[b].xidx else "y"
                          for a, b in self.links]

        # pair accumulators, flat index requester * n + supplier
        self._req_count = [0] * nn
        self._data_count = [0] * nn
        self._data_bytes = [0.0] * nn
        self._pair_hops = [0] * nn
        self._pair_wait = [0.0] * nn

        # per-link
        self._link_bytes = [0.0] * self.n_links
        self._link_count = [0] * self.n_links
        self._link_busy = [0.0] * self.n_links
        self._link_wait = [0.0] * self.n_links

        # per-die egress / ingress -- the headline metric for multicast
        self._die_out_bytes = [0.0] * n
        self._die_in_bytes = [0.0] * n
        self._die_out_count = [0] * n

        # per (iteration, layer) epoch
        self.epochs = []       # (iter, layer, stage, t_begin)
        self._epoch_stat = []  # [transfers, bytes, hops, wait, t_last]
        self._epoch_die_out = []
        self._cur = None
        self._cur_out = None

        self._route_cache = {}
        self.total_bytes = 0.0
        self.total_transfers = 0
        self.llc_bytes = 0.0        # payload sourced from a remote LLC
        self.crossings = 0          # link traversals by payload (unicast hops
                                    # or tree links) -- the comparable metric
        self.mcast_deliveries = 0   # tree transfers issued
        self.mcast_destinations = 0 # requesters served by them
        self.dram_bytes = 0.0       # payload sourced from remote DRAM
        self.max_time_ns = 0.0

    # ---------------------------------------------------------- numpy views
    def _m(self, lst):
        return np.array(lst).reshape(self.n_dies, self.n_dies)

    @property
    def data_bytes(self): return self._m(self._data_bytes)
    @property
    def data_count(self): return self._m(self._data_count)
    @property
    def pair_hops(self): return self._m(self._pair_hops)
    @property
    def link_busy_ns(self): return np.array(self._link_busy)
    @property
    def link_bytes(self): return np.array(self._link_bytes)
    @property
    def payload_bytes(self):
        """Expert-slice bytes on the links, excluding 16-byte request messages."""
        return self.dram_bytes + self.llc_bytes

    @property
    def die_out_bytes(self): return np.array(self._die_out_bytes)
    @property
    def die_in_bytes(self): return np.array(self._die_in_bytes)

    # ---------------------------------------------------------------- epoch
    def begin_epoch(self, iter_id, layer_id, stage, t_begin):
        if not self.enabled:
            return
        self._flush()
        self.epochs.append((int(iter_id), int(layer_id), stage, float(t_begin)))
        self._cur = [0, 0.0, 0, 0.0, 0.0]
        self._cur_out = [0.0] * self.n_dies

    def _flush(self):
        if self._cur is None:
            return
        self._epoch_stat.append(self._cur)
        self._epoch_die_out.append(self._cur_out)
        self._cur = None

    # --------------------------------------------------------------- record
    def _build_route(self, key, path):
        ids = [c.id for c in path]
        idx = []
        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]
            idx.append(self._link_index[(a, b) if a < b else (b, a)])
        entry = (len(idx), idx)
        self._route_cache[key] = entry
        return entry

    def record(self, kind, path, data_size, service_ns, start_ns, wait_ns=0.0):
        """Log one D2D message.

        ``path`` is in data-flow order: requester -> supplier for a request leg,
        supplier -> requester for a payload leg. ``service_ns`` is the transfer's
        own link occupancy; ``wait_ns`` is how long it sat blocked on a busy link.
        """
        src, dst = path[0].id, path[-1].id
        key = (src, dst)
        entry = self._route_cache.get(key)
        if entry is None:
            entry = self._build_route(key, path)
        hops, link_idx = entry
        if hops == 0:
            return
        if wait_ns < 0.0:
            wait_ns = 0.0
        n = self.n_dies

        if kind != KIND_REQ:
            if kind == KIND_DATA_LLC:
                self.llc_bytes += data_size
            else:
                self.dram_bytes += data_size
            requester, supplier = dst, src
            flat = requester * n + supplier
            self._data_count[flat] += 1
            self._data_bytes[flat] += data_size
            # egress is charged to the die that sent the payload
            self._die_out_bytes[supplier] += data_size
            self._die_out_count[supplier] += 1
            self._die_in_bytes[requester] += data_size
            if self._cur_out is not None:
                self._cur_out[supplier] += data_size
        else:
            requester, supplier = src, dst
            flat = requester * n + supplier
            self._req_count[flat] += 1

        self._pair_hops[flat] = hops
        self._pair_wait[flat] += wait_ns
        if kind != KIND_REQ:
            self.crossings += hops

        lb, lc, lbusy, lwait = (self._link_bytes, self._link_count,
                                self._link_busy, self._link_wait)
        for li in link_idx:
            lb[li] += data_size
            lc[li] += 1
            lbusy[li] += service_ns
            lwait[li] += wait_ns

        cur = self._cur
        if cur is not None:
            cur[0] += 1
            cur[1] += data_size
            cur[2] += hops
            cur[3] += wait_ns
            end = start_ns + service_ns
            if end > cur[4]:
                cur[4] = end

        self.total_bytes += data_size
        self.total_transfers += 1
        end = start_ns + service_ns
        if end > self.max_time_ns:
            self.max_time_ns = end

    def record_tree(self, supplier, requesters, links, first_hops,
                    data_size, service_ns, start_ns, wait_ns=0.0):
        """Log one multicast delivery.

        Differs from `record` in two ways that are the whole point of multicast:
        each tree link carries the payload ONCE however many destinations sit
        behind it, and the supplier's egress is charged per distinct first hop
        rather than per destination.
        """
        if not self.enabled:
            return
        if wait_ns < 0.0:
            wait_ns = 0.0
        n = self.n_dies
        self.dram_bytes += data_size
        self.crossings += len(links)
        self.mcast_deliveries += 1
        self.mcast_destinations += len(requesters)

        for li_key in links:
            li = self._link_index[li_key]
            self._link_bytes[li] += data_size
            self._link_count[li] += 1
            self._link_busy[li] += service_ns
            self._link_wait[li] += wait_ns

        # egress: one copy per distinct output port at the supplier
        egress = len(first_hops) * data_size
        self._die_out_bytes[supplier] += egress
        self._die_out_count[supplier] += len(first_hops)
        if self._cur_out is not None:
            self._cur_out[supplier] += egress

        for r in requesters:
            flat = r * n + supplier
            self._data_count[flat] += 1
            self._data_bytes[flat] += data_size
            self._die_in_bytes[r] += data_size

        cur = self._cur
        if cur is not None:
            cur[0] += 1
            cur[1] += data_size
            cur[2] += len(links)
            cur[3] += wait_ns
            end = start_ns + service_ns
            if end > cur[4]:
                cur[4] = end
        self.total_bytes += data_size
        self.total_transfers += 1
        end = start_ns + service_ns
        if end > self.max_time_ns:
            self.max_time_ns = end

    # --------------------------------------------------------------- export
    def export(self, output_dir, stem, sim=None):
        if not self.enabled:
            return []
        self._flush()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return [self._w_dies(out / f"{stem}_comm_dies.csv", sim),
                self._w_links(out / f"{stem}_comm_links.csv"),
                self._w_pairs(out / f"{stem}_comm_pairs.csv"),
                self._w_timeline(out / f"{stem}_comm_timeline.csv"),
                self._w_summary(out / f"{stem}_comm_summary.csv", sim)]

    def _w_dies(self, path, sim):
        n = self.n_dies
        GB = 1024 ** 3
        peak_epoch_out = np.zeros(n)
        for row in self._epoch_die_out:
            peak_epoch_out = np.maximum(peak_epoch_out, row)
        incident = [0.0] * n
        for i, (a, b) in enumerate(self.links):
            incident[a] += self._link_busy[i]
            incident[b] += self._link_busy[i]
        dc = self.data_count
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["chiplet", "x", "y", "egress_GB", "ingress_GB",
                        "egress_transfers", "distinct_requesters_served",
                        "peak_epoch_egress_GB", "incident_link_busy_ns",
                        "hm_replica_GB", "hm_evictions"])
            for c in range(n):
                chip = self.interconnect.chiplets[c]
                dram = chip.dram
                w.writerow([c, chip.xidx, chip.yidx,
                            f"{self._die_out_bytes[c] / GB:.6f}",
                            f"{self._die_in_bytes[c] / GB:.6f}",
                            self._die_out_count[c],
                            int((dc[:, c] > 0).sum()),
                            f"{peak_epoch_out[c] / GB:.6f}",
                            f"{incident[c]:.0f}",
                            f"{getattr(dram, 'hm_used_GB', 0.0):.4f}",
                            getattr(dram, "hm_evictions", 0)])
        return path

    def _w_links(self, path):
        rt = max(self.max_time_ns, 1.0)
        GB = 1024 ** 3
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["link_a", "link_b", "axis", "transfers", "GB",
                        "busy_ns", "utilization", "wait_ns", "avg_wait_ns"])
            for i, (a, b) in enumerate(self.links):
                c = self._link_count[i]
                w.writerow([a, b, self.link_axis[i], c,
                            f"{self._link_bytes[i] / GB:.6f}",
                            f"{self._link_busy[i]:.0f}",
                            f"{self._link_busy[i] / rt:.6f}",
                            f"{self._link_wait[i]:.0f}",
                            f"{self._link_wait[i] / c:.1f}" if c else "0"])
        return path

    def _w_pairs(self, path):
        n = self.n_dies
        GB = 1024 ** 3
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["requester", "supplier", "hops", "req_msgs",
                        "data_transfers", "GB", "wait_ns"])
            for i in range(n):
                for j in range(n):
                    k = i * n + j
                    if self._data_count[k] or self._req_count[k]:
                        w.writerow([i, j, self._pair_hops[k], self._req_count[k],
                                    self._data_count[k],
                                    f"{self._data_bytes[k] / GB:.6f}",
                                    f"{self._pair_wait[k]:.0f}"])
        return path

    def _w_timeline(self, path):
        GB = 1024 ** 3
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iter", "layer", "stage", "begin_ns", "end_ns", "span_ns",
                        "transfers", "GB", "hops", "wait_ns", "peak_die_egress_GB"])
            m = len(self.epochs)
            for k in range(m):
                it, ly, st, t0 = self.epochs[k]
                t1 = self.epochs[k + 1][3] if k + 1 < m else self.max_time_ns
                s = self._epoch_stat[k]
                w.writerow([it, ly, st, f"{t0:.0f}", f"{t1:.0f}",
                            f"{max(t1 - t0, 0.0):.0f}", s[0], f"{s[1] / GB:.6f}",
                            s[2], f"{s[3]:.0f}",
                            f"{max(self._epoch_die_out[k]) / GB:.6f}"])
        return path

    def _w_summary(self, path, sim):
        rt = max(self.max_time_ns, 1.0)
        GB = 1024 ** 3
        util = self.link_busy_ns / rt
        out = self.die_out_bytes
        dc = self.data_count
        rows = {
            "chiplets": self.n_dies, "links": self.n_links,
            "runtime_ns": f"{rt:.0f}",
            "total_transfers": self.total_transfers,
            # Under multicast these diverge and the difference IS the result:
            # a slice is read once and crosses each tree link once, however many
            # destinations sit behind it.
            "source_read_GB": f"{(self.dram_bytes + self.llc_bytes) / GB:.4f}",
            "link_bytes_GB": f"{sum(self._link_bytes) / GB:.4f}",
            "payload_from_dram_GB": f"{self.dram_bytes / GB:.4f}",
            "payload_from_remote_llc_GB": f"{self.llc_bytes / GB:.4f}",
            "llc_share_of_payload": (f"{self.llc_bytes / self.total_bytes:.6f}"
                                     if self.total_bytes else "0"),
            # Counted at record time. The old form (data_count x pair_hops)
            # is wrong under multicast: a tree has no single per-pair hop count,
            # so it silently reported the unicast crossing count instead.
            "total_link_crossings": int(self.crossings),
            "die_egress_peak_GB": f"{out.max() / GB:.4f}",
            "die_egress_mean_GB": f"{out.mean() / GB:.4f}",
            "die_egress_max_over_mean": (f"{out.max() / out.mean():.4f}"
                                         if out.mean() else "0"),
            "busiest_supplier_die": int(np.argmax(out)),
            # distinct requester dies served by each supplier die. NOT expert
            # fanout (dies per expert) -- that is a per-expert quantity and is
            # measured by the allocation plan, not the pair matrix.
            "mean_requesters_per_supplier": f"{(dc > 0).sum(axis=0).mean():.3f}",
            "max_requesters_per_supplier": int((dc > 0).sum(axis=0).max()),
            "link_util_mean": f"{util.mean():.6f}",
            "link_util_max": f"{util.max():.6f}" if self.n_links else "0",
            "link_util_max_over_mean": (f"{util.max() / util.mean():.4f}"
                                        if self.n_links and util.mean() else "0"),
            "link_wait_total_ns": f"{sum(self._link_wait):.0f}",
            "mcast_deliveries": self.mcast_deliveries,
            "mcast_destinations": self.mcast_destinations,
            "mcast_mean_fanout": (f"{self.mcast_destinations / self.mcast_deliveries:.3f}"
                                  if self.mcast_deliveries else "0"),
        }
        if sim is not None:
            rows["sim_hop_counter"] = sim.stats.get("hop", 0)
            h = getattr(sim, "fanout_hist", None)
            if h is not None and h.sum():
                k = np.arange(h.size)
                tot = h.sum()
                rows["slice_broadcasts"] = int(tot)
                rows["fanout_mean"] = f"{(k * h).sum() / tot:.3f}"
                rows["fanout_max"] = int(np.max(k[h > 0]))
                csum = np.cumsum(h) / tot
                for q in (50, 90):
                    rows[f"fanout_p{q}"] = int(np.searchsorted(csum, q / 100.0))
                # unicast crossings this fanout implies, vs an n-1 spanning tree
                rows["fanout_hist"] = " ".join(
                    f"{i}:{int(v)}" for i, v in enumerate(h) if v)
            rows["hm_evictions_total"] = sum(
                getattr(c.dram, "hm_evictions", 0) for c in self.interconnect.chiplets)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in rows.items():
                w.writerow([k, v])
        return path
