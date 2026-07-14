"""Cross-frame object tracking with constant-velocity motion prediction (minimal SORT-style).

Gives objects a STABLE id across turns AND survives OCCLUSION: when an object can't be matched in the
current frame (e.g. it passed behind another object), instead of dropping it we keep its track alive for
up to `max_age` frames, extrapolating its position along its last velocity, and RE-IDENTIFY it when it
reappears. This preserves object permanence (a Chollet ARC core prior, arXiv:1911.01547) and fixes the
id-break seen when ar25's mirror objects cross the central bar. Motion is constant-velocity (ar25 objects
translate by a fixed step each turn), so no Kalman covariance is needed — just `pos += velocity`.
Self-contained (no scipy); reuses `grid_render._object_descriptors`.

Per-episode: build one ObjectTracker, call `update(grid)` each turn.
`update` returns `(tagged, bg, changes)`:
  - tagged  = [(stable_id, descriptor, state), ...]   state ∈ {"visible","occluded"}
              occluded entries carry the PREDICTED center (object permanence).
  - changes = list of ("MOVED",id,dx,dy) / ("RESHAPED",id,old,new,dx,dy) / ("OCCLUDED",id,(px,py))
              / ("REAPPEARED",id,dx,dy) / ("APPEARED",id,desc) / ("GONE",id,desc).
"""
from __future__ import annotations

from examples.arc_agi3.grid_render import _object_descriptors


def _shape_family(s: str) -> str:
    """'vline:64' -> 'vline', 'mask:...' -> 'mask', 'dot' -> 'dot'."""
    return s.split(":")[0]


def _hungarian(cost):
    """Min-cost assignment on an m×n cost matrix → list of (row, col). Self-contained Kuhn–Munkres."""
    m = len(cost)
    n = len(cost[0]) if m else 0
    if m == 0 or n == 0:
        return []
    transposed = m > n
    if transposed:
        cost = [[cost[i][j] for i in range(m)] for j in range(n)]
        m, n = n, m
    INF = float("inf")
    u = [0.0] * (m + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, m + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, n + 1):
        if p[j] > 0:
            pairs.append((j - 1, p[j] - 1) if transposed else (p[j] - 1, j - 1))
    return pairs


class _Track:
    __slots__ = ("id", "color", "desc", "cx", "cy", "vx", "vy", "missed", "state")

    def __init__(self, tid, desc):
        self.id = tid
        self.color = desc["color"]
        self.desc = desc          # last VISIBLE descriptor
        self.cx, self.cy = desc["cx"], desc["cy"]
        self.vx = self.vy = 0     # constant-velocity estimate (per-frame)
        self.missed = 0           # consecutive frames unmatched
        self.state = "visible"

    def predict(self):
        """Predicted current center = last seen center extrapolated along velocity for missed frames."""
        return (self.cx + self.vx * self.missed, self.cy + self.vy * self.missed)


class ObjectTracker:
    def __init__(self, dist_threshold: float = 40.0, max_age: int = 5):
        self.next_id = 0
        self.tracks = []          # list of _Track (alive: visible or occluded ≤ max_age)
        self.T = dist_threshold   # squared-distance gate for a match
        self.max_age = max_age    # frames an occluded track is kept alive before GONE

    def _new(self, desc) -> _Track:
        t = _Track(self.next_id, desc)
        self.next_id += 1
        return t

    def update(self, grid):
        cur, bg = _object_descriptors(grid)
        if not self.tracks:
            self.tracks = [self._new(d) for d in cur]
            return ([(t.id, t.desc, "visible") for t in self.tracks], bg,
                    [("APPEARED", t.id, t.desc) for t in self.tracks])

        tracks = self.tracks
        BIG = 1e9
        # cost[r][c] = current-object r matched to track c, scored against the track's PREDICTED position
        cost = []
        for d in cur:
            row = []
            for t in tracks:
                if d["color"] != t.color:
                    row.append(BIG)
                    continue
                px, py = t.predict()
                dist2 = (d["cx"] - px) ** 2 + (d["cy"] - py) ** 2
                if d["shape"] == t.desc["shape"]:
                    sp = 0.0
                elif _shape_family(d["shape"]) == _shape_family(t.desc["shape"]):
                    sp = 4.0
                else:
                    sp = 100.0    # different family forbidden (>T): don't let a partial/occlusion
                    #               fragment (e.g. a solid block) hijack a mask track and corrupt its
                    #               velocity — keep it OCCLUDED and re-id the real shape when it re-emerges
                row.append(dist2 + sp)
            cost.append(row)
        pairs = _hungarian(cost) if (cur and tracks) else []

        cur_to_track = [None] * len(cur)
        matched_tracks = set()
        for r, c in pairs:
            if cost[r][c] <= self.T:
                cur_to_track[r] = c
                matched_tracks.add(c)

        changes = []
        # 1) update matched tracks
        for r, c in enumerate(cur_to_track):
            if c is None:
                continue
            t = tracks[c]
            d = cur[r]
            frames = t.missed + 1
            dx, dy = d["cx"] - t.cx, d["cy"] - t.cy
            was_occ = t.state == "occluded"
            old_shape = t.desc["shape"]
            t.vx = round(dx / frames)
            t.vy = round(dy / frames)
            t.cx, t.cy = d["cx"], d["cy"]
            t.desc = d
            t.missed = 0
            t.state = "visible"
            if was_occ:
                changes.append(("REAPPEARED", t.id, dx, dy))
            elif old_shape != d["shape"]:
                changes.append(("RESHAPED", t.id, old_shape, d["shape"], dx, dy))
            elif dx or dy:
                changes.append(("MOVED", t.id, dx, dy))

        # 2) unmatched current objects → new tracks
        new_tracks = []
        for r, d in enumerate(cur):
            if cur_to_track[r] is None:
                t = self._new(d)
                new_tracks.append(t)
                changes.append(("APPEARED", t.id, d))

        # 3) unmatched tracks → occluded (keep alive, predict) or GONE (past max_age)
        survivors = []
        for c, t in enumerate(tracks):
            if c in matched_tracks:
                survivors.append(t)
                continue
            if t.state != "occluded":
                changes.append(("OCCLUDED", t.id, t.predict()))
            t.missed += 1
            t.state = "occluded"
            if t.missed > self.max_age:
                changes.append(("GONE", t.id, t.desc))
            else:
                survivors.append(t)
        self.tracks = survivors + new_tracks

        # 4) emit current view: visible = real descriptor; occluded = predicted center
        tagged = []
        for t in self.tracks:
            if t.state == "visible":
                tagged.append((t.id, t.desc, "visible"))
            else:
                px, py = t.predict()
                pd = dict(t.desc)
                pd["cx"], pd["cy"] = px, py
                tagged.append((t.id, pd, "occluded"))
        return tagged, bg, changes


def tracked_objects_text(tagged, bg) -> str:
    """Render tracker output (STABLE ids + occlusion) as prompt text. Occluded objects show their
    PREDICTED position (object permanence) so the model knows they're still there, just hidden."""
    lines = [f"# background={bg}; {len(tagged)} objects:"]
    for tid, d, state in sorted(tagged, key=lambda x: x[0]):
        col = "/".join(map(str, d["color"])) if isinstance(d["color"], list) else d["color"]
        if state == "occluded":
            lines.append(f"obj{tid}: color={col} size={d['size']} center=({d['cx']},{d['cy']}) "
                         f"shape={d['shape']} [OCCLUDED: hidden behind another object, position predicted]")
        else:
            lines.append(f"obj{tid}: color={col} size={d['size']} "
                         f"bbox=x[{d['x0']}-{d['x1']}]y[{d['y0']}-{d['y1']}] "
                         f"center=({d['cx']},{d['cy']}) shape={d['shape']}")
    return "\n".join(lines)


def changes_to_text(changes) -> str:
    """Render the tracker's per-frame changes (with stable ids) as the object-level diff d_t (recorded
    in the trajectory; not shown to the model — the model compares the BEFORE/AFTER object lists)."""
    if not changes:
        return "No object changed (no-op)."
    out = []
    for c in changes:
        k = c[0]
        if k == "MOVED":
            out.append(f"obj{c[1]} MOVED dx={c[2]:+d} dy={c[3]:+d}")
        elif k == "RESHAPED":
            out.append(f"obj{c[1]} RESHAPED {c[2]}->{c[3]} dx={c[4]:+d} dy={c[5]:+d}")
        elif k == "REAPPEARED":
            out.append(f"obj{c[1]} REAPPEARED (was hidden) dx={c[2]:+d} dy={c[3]:+d}")
        elif k == "OCCLUDED":
            out.append(f"obj{c[1]} now HIDDEN (occluded) at predicted ({c[2][0]},{c[2][1]})")
        elif k == "APPEARED":
            out.append(f"obj{c[1]} APPEARED")
        elif k == "GONE":
            out.append(f"obj{c[1]} GONE")
    return "\n".join(out)
