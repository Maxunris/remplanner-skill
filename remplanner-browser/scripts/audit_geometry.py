"""Read-only checks of supplied schema-v1 centimetre geometry, not compliance.

No native .plan parsing, application access, routing inference, or electrical/site
certification. Matrix weights declare edges; they are not certified route lengths.
Geometry within EPS of a boundary is deliberately UNKNOWN. Polygon intersections
partition each whole segment into intervals; this is not fixed-point sampling.
"""

import json
import math
import sys


SCOPE = "geometry_only_not_compliance"
EPS = 1e-7  # cm; contact/roundoff tolerance, not a construction tolerance
MAX_BYTES = 10 * 1024 * 1024
MAX_INTEGER_DIGITS = 128  # JSON integer digits, excluding an optional minus sign
MAX_ITEMS = 128
MAX_POLYGON = 256
MAX_GRAPH_VERTICES = 128
MAX_EDGES = 512
MAX_WORK = 2_000_000  # conservative bound including interval classification


def finding(code, status, subject, detail):
    return {"code": code, "status": status, "subject": subject, "detail": detail}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def point(value):
    return (isinstance(value, list) and len(value) == 2
            and all(number(x) and abs(x) <= 1e9 for x in value))


def identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return distance(p, a)
    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return distance(p, (a[0] + t * dx, a[1] + t * dy))


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def cuts(a, b, c, d):
    """Parameters on ab where cd intersects/touches it, including overlap ends."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    q = (c[0] - a[0], c[1] - a[1])
    lr, ls = math.hypot(*r), math.hypot(*s)
    if lr <= EPS:
        return [0.0] if segment_distance(a, c, d) <= EPS else []
    denominator = cross(r, s)
    tolerance = EPS / lr
    # Cross products have cm² units: scale the cm tolerance by actual
    # segment length, without imposing a 1 cm floor on short segments.
    if abs(denominator) <= EPS * max(lr, ls):
        if abs(cross(q, r)) > EPS * lr:
            return []
        rr = lr * lr
        t0 = (q[0] * r[0] + q[1] * r[1]) / rr
        t1 = t0 + (s[0] * r[0] + s[1] * r[1]) / rr
        lo, hi = max(0, min(t0, t1)), min(1, max(t0, t1))
        return [max(0, min(1, lo)), max(0, min(1, hi))] if lo <= hi + tolerance else []
    t, u = cross(q, s) / denominator, cross(q, r) / denominator
    if -tolerance <= t <= 1 + tolerance and -EPS / max(ls, EPS) <= u <= 1 + EPS / max(ls, EPS):
        return [max(0, min(1, t))]
    return []


def polygon(value):
    if not isinstance(value, list) or not 3 <= len(value) <= MAX_POLYGON or not all(map(point, value)):
        return False
    # Closure is implicit: a repeated first/last vertex is invalid normalized data.
    edges = list(zip(value, value[1:] + value[:1]))
    if any(distance(a, b) <= EPS for a, b in edges):
        return False
    area2 = sum(cross((a[0] - value[0][0], a[1] - value[0][1]),
                      (b[0] - value[0][0], b[1] - value[0][1])) for a, b in edges)
    if abs(area2) <= EPS * sum(distance(a, b) for a, b in edges):
        return False
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if cuts(a, b, *edges[j]):
                return False
    return True


def location(p, poly):
    """-1 outside, 0 boundary/uncertain contact, +1 strictly inside."""
    inside = False
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if segment_distance(p, a, b) <= EPS:
            return 0
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if p[0] < x:
                inside = not inside
    return 1 if inside else -1


def segment_locations(a, b, poly):
    """Classify every interval between consecutive intersections with polygon."""
    parameters = [0.0, 1.0]
    contact = False
    for c, d in zip(poly, poly[1:] + poly[:1]):
        intersections = cuts(a, b, c, d)
        contact = contact or bool(intersections)
        parameters.extend(intersections)
    parameters = sorted(set(parameters))
    states = {location(a, poly), location(b, poly)}
    for lo, hi in zip(parameters, parameters[1:]):
        t = (lo + hi) / 2
        states.add(location((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), poly))
    if contact:
        states.add(0)
    return states


def input_within_limits(data):
    pending, count = [(data, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 100_000 or depth > 32:
            return False
        if isinstance(item, dict):
            if len(item) > 1024:
                return False
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            if len(item) > 4096:
                return False
            pending.extend((v, depth + 1) for v in item)
    return True


def audit(data):
    """Return findings without mutating data. PASS applies only to supplied scope."""
    if not isinstance(data, dict) or not input_within_limits(data):
        return [finding("INPUT_ERROR", "UNKNOWN", "input", "Expected a bounded JSON object (depth <=32, nodes <=100000).")]
    result = []

    def add(code, status, subject, detail):
        result.append(finding(code, status, subject, detail))

    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        return [finding("SCHEMA", "UNKNOWN", "input", "Only integer schema_version 1 is supported.")]
    add("SCHEMA", "PASS", "input", "Normalized schema version 1; geometry-only checks, not compliance certification.")
    if data.get("units") != "cm":
        add("UNITS", "UNKNOWN", "input", "Only explicit cm units are supported; no conversion or cm checks performed.")
        return result
    add("UNITS", "PASS", "input", "Coordinates and supplied preferences use cm.")

    def collection(key, expected):
        value = data.get(key)
        if not isinstance(value, expected) or len(value) > MAX_ITEMS:
            add(key.upper(), "UNKNOWN", key, "Section missing, malformed, or above 128 entries.")
            return None
        add(key.upper(), "PASS", key, "Explicitly supplied collection; emptiness is limited to this audit scope.")
        return value

    boundary = data.get("boundary")
    if not polygon(boundary):
        add("BOUNDARY", "UNKNOWN", "boundary", "Missing/invalid simple polygon: finite cm pairs, 3..256 vertices, implicit closure required.")
        boundary = None
    else:
        add("BOUNDARY", "PASS", "boundary", "Supplied polygon is simple and nondegenerate.")
    obstacles = collection("obstacles", list)
    points = collection("points", list)
    doors = collection("doors", list)
    graphs = collection("graphs", list)
    preferences = collection("preferences", dict)
    facts = collection("required_facts", dict)

    def records(values, geometry_valid, code):
        if values is None:
            return {}, False
        valid, seen, complete = {}, set(), True
        for index, item in enumerate(values):
            subject = str(index)
            if not isinstance(item, dict) or not identifier(item.get("id")):
                add(code, "UNKNOWN", subject, "Object with a nonempty string id is required.")
                complete = False
                continue
            subject = item["id"]
            if subject in seen:
                valid.pop(subject, None)
                add(code, "UNKNOWN", subject, "Duplicate id is ambiguous.")
                complete = False
                continue
            seen.add(subject)
            if not geometry_valid(item):
                add(code, "UNKNOWN", subject, "Geometry missing, invalid, degenerate, or beyond supported limits.")
                complete = False
                continue
            valid[subject] = item
        return valid, complete

    obstacle_map, obstacles_complete = records(obstacles, lambda x: polygon(x.get("polygon")), "OBSTACLE")
    point_map, points_complete = records(points, lambda x: point(x.get("point")), "POINT")
    door_map, doors_complete = records(doors, lambda x: point(x.get("a")) and point(x.get("b"))
                                     and distance(x["a"], x["b"]) > EPS, "DOOR")
    work_left = MAX_WORK

    def spatial(a, b, subject, prefix, allow_exterior=False):
        nonlocal work_left
        sizes = ([len(boundary)] if boundary and not allow_exterior else [])
        sizes.extend(len(x["polygon"]) for x in obstacle_map.values())
        needed = sum(size * (size + 4) for size in sizes)
        work_left -= needed
        if work_left < 0:
            add("INPUT_ERROR", "UNKNOWN", subject, "Geometry work limit exceeded; remaining spatial checks unavailable.")
            return
        if allow_exterior:
            add(prefix + "_BOUNDARY", "PASS", subject, "Boundary containment explicitly excluded by allow_exterior=true.")
        elif boundary is None:
            add(prefix + "_BOUNDARY", "UNKNOWN", subject, "Valid boundary unavailable.")
        else:
            states = segment_locations(a, b, boundary)
            status = "FAIL" if -1 in states else "UNKNOWN" if 0 in states else "PASS"
            add(prefix + "_BOUNDARY", status, subject, "Whole supplied segment: outside boundary." if status == "FAIL" else
                "Boundary contact or numerical proximity; verify placement." if status == "UNKNOWN" else
                "Whole supplied segment lies strictly inside boundary.")
        if not obstacles_complete:
            add(prefix + "_OBSTACLE", "UNKNOWN", subject, "Obstacle collection incomplete or invalid.")
        for obstacle_id, obstacle in obstacle_map.items():
            states = segment_locations(a, b, obstacle["polygon"])
            status = "FAIL" if 1 in states else "UNKNOWN" if 0 in states else "PASS"
            add(prefix + "_OBSTACLE", status, subject + "/" + obstacle_id,
                "Supplied segment enters obstacle interior." if status == "FAIL" else
                "Obstacle contact or numerical proximity; verify placement." if status == "UNKNOWN" else
                "Supplied segment is clear of this obstacle.")
        if obstacles_complete and not obstacle_map:
            add(prefix + "_OBSTACLE", "PASS", subject, "Explicit obstacle collection is empty; no obstacle intersections to check.")

    for point_id, item in point_map.items():
        spatial(item["point"], item["point"], point_id, "POINT")

    graph_ids = set()
    for index, graph in enumerate(graphs or []):
        if not isinstance(graph, dict) or not identifier(graph.get("id")):
            add("GRAPH", "UNKNOWN", str(index), "Graph requires a nonempty string id.")
            continue
        graph_id = graph["id"]
        if graph_id in graph_ids:
            add("GRAPH", "UNKNOWN", graph_id, "Duplicate graph id is ambiguous.")
        graph_ids.add(graph_id)
        vertices = graph.get("vertices")
        if not isinstance(vertices, list) or not 1 <= len(vertices) <= MAX_GRAPH_VERTICES:
            add("VERTICES", "UNKNOWN", graph_id, "Expected 1..128 vertices.")
            continue
        vertex_map, vertices_complete = records(vertices, lambda x: point(x.get("point")), "VERTEX")
        if "allow_exterior" in graph and type(graph["allow_exterior"]) is not bool:
            add("GRAPH", "UNKNOWN", graph_id, "allow_exterior must be boolean; boundary exclusion not applied.")
        for vertex_id, vertex in vertex_map.items():
            spatial(vertex["point"], vertex["point"], graph_id + "/" + vertex_id,
                    "VERTEX", graph.get("allow_exterior") is True)
            if "anchor_id" not in vertex:
                continue
            anchor_id = vertex["anchor_id"]
            if not identifier(anchor_id) or anchor_id not in point_map:
                add("ANCHOR", "UNKNOWN", graph_id + "/" + vertex_id, "Referenced anchor point is missing or invalid.")
            else:
                match = distance(vertex["point"], point_map[anchor_id]["point"]) <= EPS
                add("ANCHOR", "PASS" if match else "FAIL", graph_id + "/" + vertex_id,
                    "Vertex matches referenced point centre." if match else "Vertex does not match referenced point centre.")
        known_geometry = graph.get("geometry") == "polyline"
        add("GRAPH_GEOMETRY", "PASS" if known_geometry and vertices_complete else "UNKNOWN", graph_id,
            "Edges are explicit straight segments between supplied vertices." if known_geometry and vertices_complete else
            "Complete explicit polyline geometry unavailable; chords cannot establish route geometry.")
        matrix = graph.get("matrix")
        n = len(vertices)
        if matrix is None:
            add("MATRIX", "UNKNOWN", graph_id, "Matrix missing.")
            continue
        valid_matrix = isinstance(matrix, list) and len(matrix) == n and all(isinstance(row, list) and len(row) == n for row in matrix)
        if valid_matrix:
            for i in range(n):
                for j in range(n):
                    weight = matrix[i][j]
                    if ((i == j and weight is not None) or
                            (weight is not None and (not number(weight) or weight <= 0)) or
                            weight != matrix[j][i]):
                        valid_matrix = False
        if not valid_matrix:
            add("MATRIX", "FAIL", graph_id, "Matrix must match vertex count, be square/symmetric, have null diagonal and null or finite positive weights.")
            continue
        add("MATRIX", "PASS", graph_id, "Square symmetric adjacency matrix; no self edges; finite positive declared weights.")
        visited, todo = set(), [0]
        while todo:
            current = todo.pop()
            if current in visited:
                continue
            visited.add(current)
            todo.extend(j for j, weight in enumerate(matrix[current]) if weight is not None and j not in visited)
        add("CONNECTIVITY", "PASS" if len(visited) == n else "FAIL", graph_id,
            "All supplied vertices are connected." if len(visited) == n else "Graph contains disconnected vertices/components.")
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if matrix[i][j] is not None]
        if len(edges) > MAX_EDGES:
            add("INPUT_ERROR", "UNKNOWN", graph_id, "More than 512 edges; spatial checks unavailable.")
            continue
        if known_geometry and vertices_complete:
            for i, j in edges:
                a, b = vertices[i], vertices[j]
                subject = graph_id + "/" + a["id"] + "-" + b["id"]
                if distance(a["point"], b["point"]) <= EPS:
                    add("EDGE_GEOMETRY", "UNKNOWN", subject, "Coincident endpoints do not establish nondegenerate route geometry.")
                    continue
                spatial(a["point"], b["point"], subject, "EDGE", graph.get("allow_exterior") is True)

    if preferences is not None:
        heights = preferences.get("heights_cm")
        valid_heights = isinstance(heights, dict) and all(identifier(role) and number(height) and height >= 0 for role, height in heights.items())
        if "heights_cm" in preferences and not valid_heights:
            add("PREFERENCES", "UNKNOWN", "heights_cm", "Expected role-to-finite-nonnegative-height mapping.")
        if valid_heights:
            if not points_complete:
                add("HEIGHT", "UNKNOWN", "points", "Incomplete points prevent checking all supplied height preferences.")
            for point_id, item in point_map.items():
                role = item.get("role")
                if not isinstance(role, str) or role not in heights:
                    continue
                actual = item.get("height_cm")
                status = "UNKNOWN" if not number(actual) or actual < 0 else "PASS" if abs(actual - heights[role]) <= EPS else "FAIL"
                add("HEIGHT", status, point_id, "Point centre height compared only with explicitly supplied role preference; not an electrical rule.")
        if "door_clearance_cm" in preferences:
            clearance = preferences["door_clearance_cm"]
            if not number(clearance) or clearance < 0:
                add("PREFERENCES", "UNKNOWN", "door_clearance_cm", "Expected a finite nonnegative minimum centre-to-door-segment distance.")
            else:
                if not points_complete:
                    add("DOOR_CLEARANCE", "UNKNOWN", "points", "Incomplete points prevent checking all switch centres.")
                for point_id, item in point_map.items():
                    if item.get("role") != "switch":
                        continue
                    if not door_map:
                        add("DOOR_CLEARANCE", "UNKNOWN", point_id, "No valid door segment supplied for switch centre clearance.")
                        continue
                    actual = min(segment_distance(item["point"], door["a"], door["b"]) for door in door_map.values())
                    status = "FAIL" if actual + EPS < clearance else "PASS" if doors_complete else "UNKNOWN"
                    add("DOOR_CLEARANCE", status, point_id,
                        "Switch centre-to-nearest-door-segment distance %.6g cm; supplied minimum %.6g cm. This does not measure frame or trim edges." % (actual, clearance))
    for key, value in (facts or {}).items():
        add("REQUIRED_FACT", "UNKNOWN" if value is None else "PASS", str(key),
            "Required fact not supplied." if value is None else "Required fact supplied; its truth is not independently verified.")
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    error = None
    if len(args) != 1:
        error = "Usage: python3 audit_geometry.py input.json"
    else:
        try:
            with open(args[0], "rb") as handle:
                raw = handle.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("oversize")

            def reject_constant(_value):
                raise ValueError("nonfinite JSON number")

            def finite_float(value):
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError("nonfinite JSON number")
                return parsed

            def bounded_int(value):
                # Enforce before int(), independently of interpreter digit guards.
                digits = len(value) - (1 if value.startswith("-") else 0)
                if digits > MAX_INTEGER_DIGITS:
                    raise ValueError("JSON integer exceeds 128 digits")
                return int(value)

            def unique_object(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        raise ValueError("duplicate JSON key")
                    obj[key] = value
                return obj

            data = json.loads(raw, parse_constant=reject_constant,
                              parse_float=finite_float, parse_int=bounded_int,
                              object_pairs_hook=unique_object)
            findings = audit(data)
        except (OSError, ValueError, UnicodeError, RecursionError, OverflowError):
            error = "Input unreadable, invalid JSON, nonfinite JSON literal, or above 10 MiB/depth/128-integer-digit limits."
    if error:
        findings = [finding("INPUT_ERROR", "UNKNOWN", "input", error)]
    print(json.dumps({"scope": SCOPE, "findings": findings}, ensure_ascii=True, allow_nan=False))
    statuses = {item["status"] for item in findings}
    return 1 if "FAIL" in statuses else 2 if "UNKNOWN" in statuses else 0


if __name__ == "__main__":
    sys.exit(main())
