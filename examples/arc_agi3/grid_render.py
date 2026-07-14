"""Render an ARC-AGI-3 64x64 integer grid to a PIL image and compute the exact
per-object diff text.

Ported verbatim (palette + geometry) from
``ARC-AGI-3-Agents/agents/templates/memory_agent.py`` (``PALETTE``, ``_grid_image``,
``_grid_diff_text``) so the miles rollout does not depend on the ARC agent classes.

- ``render_grid(grid2d) -> PIL.Image`` : the image observation block (VLM input).
- ``grid_diff_text(before, after) -> str`` : the doc's perception input ``d_t`` (what changed).
"""
from __future__ import annotations

from collections import Counter, defaultdict

# Standard ARC-AGI 16-colour palette (value -> RGB).
PALETTE = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}


def render_grid(grid2d, cell: int = 14, margin: int = 20, grid_step: int = 4, fixed_size: int = 64):
    """Render a 2D integer grid to an RGB ``PIL.Image`` with coordinate gridlines/labels.

    The grid is padded (with 0) / cropped to a fixed ``fixed_size`` x ``fixed_size`` so EVERY rendered
    image has identical pixel dimensions. This is required: Qwen-VL ``pixel_values`` from different
    turns are concatenated into one training micro-batch and must share their non-batch dimensions
    (ARC frames are not always 64x64, and terminal/error frames may be empty).

    ``cell`` = pixels per grid cell, chosen for LEGIBILITY (the model must read a 64x64/16-colour grid),
    not to save memory: at cell=14 the image resizes to ~58x58 vision patches (~1 patch/cell), much
    clearer than cell=10's 42x42. (Image size only affects forward activation, not the optimizer-step
    memory that bounds this setup, so there is no reason to shrink it.)
    """
    from PIL import Image, ImageDraw

    H = W = fixed_size
    norm = [[0] * W for _ in range(H)]
    for r in range(min(len(grid2d), H)):
        row = grid2d[r]
        for c in range(min(len(row), W)):
            try:
                norm[r][c] = int(row[c])
            except (TypeError, ValueError):
                norm[r][c] = 0
    img = Image.new("RGB", (margin + W * cell, margin + H * cell), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for r in range(H):
        for c in range(W):
            col = PALETTE.get(norm[r][c], (180, 180, 180))
            x0, y0 = margin + c * cell, margin + r * cell
            d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=col)
    # coordinate gridlines + labels every grid_step cells (x = col, y = row)
    step = max(1, grid_step)
    for k in range(0, W + 1, step):
        x = margin + k * cell
        d.line([(x, margin), (x, margin + H * cell)], fill=(120, 120, 120))
        if k < W:
            d.text((x + 1, 5), str(k), fill=(0, 0, 0))
    for k in range(0, H + 1, step):
        y = margin + k * cell
        d.line([(margin, y), (margin + W * cell, y)], fill=(120, 120, 120))
        if k < H:
            d.text((2, y + 1), str(k), fill=(0, 0, 0))
    return img


def grid_diff_text(before2d, after2d) -> str:
    """Exact per-object diff of two grids (what changed + rigid translations).

    This is the doc's ``d_t`` perception input. Computed straight from the integer
    grids so it is reliable even when the rendered image hides a rigid slide.
    """
    if not before2d or not after2d or len(before2d) != len(after2d):
        return "(diff unavailable)"
    H = len(before2d)
    W = len(before2d[0]) if H else 0
    cb: dict[int, set] = defaultdict(set)
    ca: dict[int, set] = defaultdict(set)
    changed = 0
    for r in range(H):
        for c in range(W):
            b, a = before2d[r][c], after2d[r][c]
            cb[b].add((r, c))
            ca[a].add((r, c))
            if b != a:
                changed += 1
    if changed == 0:
        return (
            f"COMPUTED DIFF: NOTHING changed — every one of the {H * W} cells is "
            "identical. The action was a true NO-OP."
        )

    def box(s: set) -> tuple:
        rs = [p[0] for p in s]
        cs = [p[1] for p in s]
        return (min(cs), max(cs), min(rs), max(rs))

    lines = []
    for v in sorted(set(cb) | set(ca)):
        B, A = cb.get(v, set()), ca.get(v, set())
        if B == A:
            continue  # this value's cells did not change
        tag = f"value {v}"
        if not B:
            x0, x1, y0, y1 = box(A)
            lines.append(f"  {tag}: APPEARED, n={len(A)} at x[{x0}-{x1}] y[{y0}-{y1}]")
            continue
        if not A:
            lines.append(f"  {tag}: DISAPPEARED (was n={len(B)})")
            continue
        x0b, x1b, y0b, y1b = box(B)
        x0a, x1a, y0a, y1a = box(A)
        note = ""
        if len(B) == len(A):
            dx, dy = x0a - x0b, y0a - y0b
            if {(r + dy, c + dx) for (r, c) in B} == A:
                note = f"  => MOVED rigidly dx={dx:+d} dy={dy:+d} (n={len(B)} unchanged)"
        if not note:
            note = f"  (n {len(B)}->{len(A)}, shape changed)"
        lines.append(
            f"  {tag}: x[{x0b}-{x1b}] y[{y0b}-{y1b}] -> x[{x0a}-{x1a}] y[{y0a}-{y1a}]{note}"
        )
    return (
        "COMPUTED DIFF (exact, from the raw grid): "
        f"{changed} of {H * W} cells changed. Per-value change "
        "(x=col 0-63, y=row 0-63; values with NO change are omitted; dx>0=right, dy>0=down):\n"
        + "\n".join(lines)
    )


def grid_matrix_text(grid2d, fixed_size: int = 64) -> str:
    """Raw integer grid as text WITH x (column) header + y (row) labels — for precise perception.

    Pairs with ``render_grid`` (the image): the image conveys shapes, this gives the model the EXACT
    cell values and coordinates. VLM perception of a 64x64 image misreads x/y and misses small or
    multi-region changes (e.g. a 1-cell slide of two blocks), so the matrix is fed alongside the image.
    Padded/cropped to ``fixed_size`` (same as ``render_grid``) so before/after matrices stay aligned.
    Each cell is right-padded to width 3 so columns line up under the header.
    """
    H = W = fixed_size
    norm = [[0] * W for _ in range(H)]
    for r in range(min(len(grid2d or []), H)):
        row = grid2d[r]
        for c in range(min(len(row), W)):
            try:
                norm[r][c] = int(row[c])
            except (TypeError, ValueError):
                norm[r][c] = 0
    header = "y\\x" + "".join(f"{c:3d}" for c in range(W))
    lines = [header]
    for r in range(H):
        lines.append(f"{r:3d}" + "".join(f"{norm[r][c]:3d}" for c in range(W)))
    return "\n".join(lines)


# ======================================================================================
# Object-centric representation (TEXT-ONLY, no image): segment the grid into connected
# objects and serialize compact descriptors (color/size/bbox/center/shape). Replaces BOTH
# the image and the raw matrix as the state representation — grounded in ARC's "objectness"
# priors (color-continuity vs spatial-contiguity) and arc-dsl's as_objects; self-contained.
# ======================================================================================

def _segment(grid, single_color=True, diagonal=True):
    """Flood-fill connected components -> (list_of_objects, bg_color). Each object = list of
    (value,(row,col)). Background = most-common value, excluded. single_color=True: components
    are single-color (color continuity); False: any adjacent non-bg cells merge (spatial
    contiguity). diagonal=True: 8-connectivity, else 4-connectivity."""
    H, W = len(grid), (len(grid[0]) if grid else 0)
    if H == 0 or W == 0:
        return [], 0
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    seen = [[False] * W for _ in range(H)]
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    objs = []
    for i in range(H):
        for j in range(W):
            if seen[i][j] or grid[i][j] == bg:
                continue
            c0 = grid[i][j]
            stack = [(i, j)]
            seen[i][j] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((grid[r][c], (r, c)))
                for dr, dc in steps:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and grid[nr][nc] != bg:
                        if (grid[nr][nc] == c0) if single_color else True:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            objs.append(cells)
    return objs, bg


def _shape_desc(cells, x0, y0, x1, y1):
    """Compact shape signature: dot / hline:N / vline:N / solid:WxH / mask:rows(0/1)."""
    h, w = y1 - y0 + 1, x1 - x0 + 1
    occ = {(r - y0, c - x0) for _, (r, c) in cells}
    n = len(occ)
    if n == 1:
        return "dot"
    if n == h * w:  # filled rectangle
        if h == 1:
            return f"hline:{w}"
        if w == 1:
            return f"vline:{h}"
        return f"solid:{w}x{h}"
    rows = ["".join("1" if (r, c) in occ else "0" for c in range(w)) for r in range(h)]
    return "mask:" + "/".join(rows)


def _object_descriptors(grid, single_color=True, diagonal=True):
    """(sorted list of object dicts, bg). Sorted by size desc then position for stable ordering."""
    objs, bg = _segment(grid, single_color, diagonal)
    out = []
    for cells in objs:
        rows = [r for _, (r, c) in cells]
        cols = [c for _, (r, c) in cells]
        colors = sorted(set(v for v, _ in cells))
        x0, x1, y0, y1 = min(cols), max(cols), min(rows), max(rows)
        out.append({
            "color": colors[0] if len(colors) == 1 else colors,
            "size": len(cells), "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "cx": round(sum(cols) / len(cols)), "cy": round(sum(rows) / len(rows)),
            "shape": _shape_desc(cells, x0, y0, x1, y1),
        })
    out.sort(key=lambda o: (-o["size"], o["y0"], o["x0"]))
    return out, bg


def _color_str(c):
    return "/".join(map(str, c)) if isinstance(c, list) else str(c)


def grid_to_objects_text(grid, single_color=True, diagonal=True) -> str:
    """Object-centric TEXT representation of a grid (replaces image + raw matrix as s_t)."""
    descs, bg = _object_descriptors(grid, single_color, diagonal)
    lines = [f"# background={bg}; {len(descs)} objects (x=col, y=row, 0-63):"]
    for i, o in enumerate(descs):
        lines.append(
            f"obj{i}: color={_color_str(o['color'])} size={o['size']} "
            f"bbox=x[{o['x0']}-{o['x1']}]y[{o['y0']}-{o['y1']}] "
            f"center=({o['cx']},{o['cy']}) shape={o['shape']}")
    return "\n".join(lines)


def objects_diff_text(before_grid, after_grid, single_color=True, diagonal=True) -> str:
    """Object-level diff d_t: match objects before->after (same color+shape, nearest center) and
    report MOVED dx/dy, APPEARED, GONE. The clean per-step prediction target for L_WM."""
    b, _ = _object_descriptors(before_grid, single_color, diagonal)
    a, _ = _object_descriptors(after_grid, single_color, diagonal)
    used, lines = set(), []
    for ob in b:
        best, bestd = None, None
        for j, oa in enumerate(a):
            if j in used:
                continue
            if oa["color"] == ob["color"] and oa["shape"] == ob["shape"]:
                d = abs(oa["cx"] - ob["cx"]) + abs(oa["cy"] - ob["cy"])
                if bestd is None or d < bestd:
                    best, bestd = j, d
        if best is None:
            lines.append(f"GONE: color={_color_str(ob['color'])} shape={ob['shape']} @({ob['cx']},{ob['cy']})")
        else:
            used.add(best)
            oa = a[best]
            dx, dy = oa["cx"] - ob["cx"], oa["cy"] - ob["cy"]
            if dx or dy:
                lines.append(f"MOVED color={_color_str(ob['color'])} shape={ob['shape']}: dx={dx:+d} dy={dy:+d}")
    for j, oa in enumerate(a):
        if j not in used:
            lines.append(f"APPEARED: color={_color_str(oa['color'])} shape={oa['shape']} @({oa['cx']},{oa['cy']})")
    return "Object changes:\n" + "\n".join(lines) if lines else "No object changed (NO-OP)."
