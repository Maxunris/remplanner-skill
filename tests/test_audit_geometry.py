"""Behavioral contracts for the public, geometry-only RemPlanner audit."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "remplanner-browser" / "scripts" / "audit_geometry.py"
FIXTURE = Path(__file__).parent / "fixtures" / "geometry.json"
_UNSET = object()


class GeometryAuditTests(unittest.TestCase):
    def setUp(self):
        # An absent implementation is an intentional feature failure, not an
        # import error that could conceal an unrelated broken test harness.
        self.assertTrue(SCRIPT.is_file(), "geometry audit CLI has not been implemented")
        spec = importlib.util.spec_from_file_location("audit_geometry", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.data = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def audit(self, data=_UNSET):
        data = self.data if data is _UNSET else data
        original = copy.deepcopy(data)
        findings = self.module.audit(data)
        self.assertEqual(data, original, "audit must not mutate caller data")
        self.assertTrue(findings)
        for finding in findings:
            self.assertEqual(set(finding), {"code", "status", "subject", "detail"})
            self.assertIn(finding["status"], {"PASS", "FAIL", "UNKNOWN"})
            self.assertTrue(all(isinstance(v, str) for v in finding.values()))
        return findings

    def require(self, code, status, data=_UNSET):
        findings = self.audit(data)
        self.assertTrue(any(f["code"] == code and f["status"] == status for f in findings), findings)
        return [f for f in findings if f["code"] == code and f["status"] == status]

    def edge(self, a, b):
        self.data["graphs"][0]["vertices"] = [
            {"id": "a", "point": a}, {"id": "b", "point": b}
        ]

    def test_complete_supplied_geometry_passes_without_certification(self):
        findings = self.audit()
        self.assertTrue(all(f["status"] == "PASS" for f in findings), findings)

    def test_concave_boundary_checks_whole_edge(self):
        self.data["boundary"] = [[0, 0], [100, 0], [100, 100], [70, 100],
                                 [70, 30], [30, 30], [30, 100], [0, 100]]
        self.edge([10, 80], [90, 80])
        self.require("EDGE_BOUNDARY", "FAIL")

    def test_narrow_obstacle_between_any_fixed_samples_is_detected(self):
        self.edge([10, 50], [90, 50])
        self.data["obstacles"] = [{"id": "thin", "polygon": [
            [41.123, 40], [41.124, 40], [41.124, 60], [41.123, 60]
        ]}]
        self.require("EDGE_OBSTACLE", "FAIL")

    def test_short_perpendicular_segments_detect_thin_obstacle(self):
        self.edge([10, 10], [10.001, 10])
        self.data["obstacles"] = [{"id": "tiny", "polygon": [
            [10.00021, 9.99995], [10.00022, 9.99995],
            [10.00022, 10.00005], [10.00021, 10.00005]
        ]}]
        self.require("EDGE_OBSTACLE", "FAIL")

    def test_edge_inside_obstacle_fails_without_crossing(self):
        self.edge([40, 50], [60, 50])
        self.data["obstacles"] = [{"id": "block", "polygon": [
            [30, 30], [70, 30], [70, 70], [30, 70]
        ]}]
        self.require("EDGE_OBSTACLE", "FAIL")

    def test_boundary_contact_is_unknown(self):
        self.edge([0, 10], [90, 10])
        self.require("EDGE_BOUNDARY", "UNKNOWN")

    def test_boundary_collinear_edge_is_unknown(self):
        self.edge([10, 0], [90, 0])
        self.require("EDGE_BOUNDARY", "UNKNOWN")

    def test_obstacle_tangent_is_unknown(self):
        self.edge([10, 50], [90, 50])
        self.data["obstacles"] = [{"id": "touch", "polygon": [
            [50, 50], [60, 60], [40, 60]
        ]}]
        self.require("EDGE_OBSTACLE", "UNKNOWN")

    def test_allow_exterior_skips_only_boundary(self):
        self.edge([-10, 50], [90, 50])
        self.data["graphs"][0]["allow_exterior"] = True
        self.data["obstacles"] = [{"id": "block", "polygon": [
            [20, 40], [30, 40], [30, 60], [20, 60]
        ]}]
        self.require("EDGE_BOUNDARY", "PASS")
        self.require("EDGE_OBSTACLE", "FAIL")

    def test_unknown_geometry_never_certifies_chord(self):
        self.data["graphs"][0]["geometry"] = "unknown"
        self.require("GRAPH_GEOMETRY", "UNKNOWN")
        self.assertFalse(any(f["code"].startswith("EDGE_") and f["status"] == "PASS"
                             for f in self.audit()))

    def test_missing_top_level_sections_cannot_pass(self):
        for section in ("boundary", "obstacles", "graphs", "points", "doors",
                        "preferences", "required_facts"):
            with self.subTest(section=section):
                data = copy.deepcopy(self.data)
                del data[section]
                self.assertIn("UNKNOWN", [f["status"] for f in self.audit(data)])

    def test_unsupported_units_do_not_run_cm_checks(self):
        self.data["units"] = "m"
        self.require("UNITS", "UNKNOWN")
        self.assertFalse(any(f["code"] in {"HEIGHT", "DOOR_CLEARANCE"} for f in self.audit()))

    def test_invalid_schema_version_is_unknown(self):
        for version in (None, True, 2, "1"):
            with self.subTest(version=version):
                self.data["schema_version"] = version
                self.require("SCHEMA", "UNKNOWN")

    def test_invalid_geometry_is_unknown(self):
        for point in ([True, 10], [float("nan"), 10], [float("inf"), 10],
                      [10], "10,10", None):
            with self.subTest(point=point):
                self.data["graphs"][0]["vertices"][0]["point"] = point
                findings = self.module.audit(self.data)
                self.assertIn("UNKNOWN", [f["status"] for f in findings])
                self.assertFalse(any(f["code"].startswith("EDGE_") and f["status"] == "PASS"
                                     for f in findings))

    def test_non_simple_or_degenerate_boundary_is_unknown(self):
        for polygon in ([[0, 0], [100, 100], [0, 100], [100, 0]],
                        [[0, 0], [10, 0], [20, 0]], [[0, 0], [0, 0], [10, 0]]):
            with self.subTest(polygon=polygon):
                self.data["boundary"] = polygon
                self.require("BOUNDARY", "UNKNOWN")

    def test_matrix_requires_square_symmetric_positive_weights_without_self_edges(self):
        for matrix in ([[None, 1]], [[None, 1], [2, None]], [[1, 1], [1, None]],
                       [[None, 0], [0, None]], [[None, -1], [-1, None]],
                       [[None, True], [True, None]], [[None, float("inf")], [float("inf"), None]]):
            with self.subTest(matrix=matrix):
                self.data["graphs"][0]["matrix"] = matrix
                self.require("MATRIX", "FAIL")

    def test_missing_matrix_is_unknown(self):
        del self.data["graphs"][0]["matrix"]
        self.require("MATRIX", "UNKNOWN")

    def test_disconnected_graph_fails(self):
        self.data["graphs"][0]["matrix"] = [[None, None], [None, None]]
        self.require("CONNECTIVITY", "FAIL")

    def test_single_graph_vertex_outside_boundary_fails(self):
        self.data["graphs"][0]["vertices"] = [{"id": "outside", "point": [-10, 10]}]
        self.data["graphs"][0]["matrix"] = [[None]]
        self.require("VERTEX_BOUNDARY", "FAIL")

    def test_single_graph_vertex_inside_obstacle_fails(self):
        self.data["graphs"][0]["vertices"] = [{"id": "blocked", "point": [50, 50]}]
        self.data["graphs"][0]["matrix"] = [[None]]
        self.data["obstacles"] = [{"id": "block", "polygon": [
            [30, 30], [70, 30], [70, 70], [30, 70]
        ]}]
        self.require("VERTEX_OBSTACLE", "FAIL")

    def test_geometry_work_budget_preserves_unknown(self):
        self.module.MAX_WORK = 1
        findings = self.module.audit(self.data)
        self.assertTrue(any(f["code"] == "INPUT_ERROR" and f["status"] == "UNKNOWN" for f in findings))

    def test_anchor_reference_missing_is_unknown(self):
        self.data["graphs"][0]["vertices"][0]["anchor_id"] = "absent"
        self.require("ANCHOR", "UNKNOWN")

    def test_anchor_coordinate_mismatch_fails(self):
        self.data["graphs"][0]["vertices"][0]["anchor_id"] = "switch"
        self.require("ANCHOR", "FAIL")

    def test_anchor_coordinate_match_passes(self):
        self.data["graphs"][0]["vertices"][0]["anchor_id"] = "switch"
        self.data["points"][0]["point"] = [10, 10]
        self.require("ANCHOR", "PASS")

    def test_height_is_checked_only_for_supplied_role_preferences(self):
        self.data["preferences"] = {}
        self.data["points"][0]["height_cm"] = 1
        self.assertFalse(any(f["code"] == "HEIGHT" for f in self.audit()))
        self.data["preferences"]["heights_cm"] = {"switch": 90}
        self.require("HEIGHT", "FAIL")
        del self.data["points"][0]["height_cm"]
        self.require("HEIGHT", "UNKNOWN")

    def test_clearance_uses_clamped_door_segment_not_infinite_line(self):
        self.data["points"][0]["point"] = [20, 60]
        self.data["doors"] = [{"id": "door", "a": [30, 20], "b": [30, 40]}]
        self.data["preferences"]["door_clearance_cm"] = 20
        findings = self.require("DOOR_CLEARANCE", "PASS")
        self.assertIn("centre", findings[0]["detail"])
        self.assertIn("segment", findings[0]["detail"])
        self.data["points"][0]["point"] = [25, 30]
        self.require("DOOR_CLEARANCE", "FAIL")

    def test_clearance_only_applies_to_switch_role(self):
        self.data["points"][0]["role"] = "socket"
        self.assertFalse(any(f["code"] == "DOOR_CLEARANCE" for f in self.audit()))

    def test_missing_doors_with_clearance_is_unknown(self):
        self.data["doors"] = []
        self.require("DOOR_CLEARANCE", "UNKNOWN")

    def test_null_required_fact_is_unknown(self):
        self.data["required_facts"]["wall_material"] = None
        self.require("REQUIRED_FACT", "UNKNOWN")

    def test_false_required_fact_is_known_without_truth_claim(self):
        self.data["required_facts"] = {"hidden_route": False}
        self.require("REQUIRED_FACT", "PASS")

    def test_malformed_sections_return_findings_without_crashing(self):
        for section in ("boundary", "obstacles", "graphs", "points", "doors", "preferences", "required_facts"):
            for value in (None, 123, "invalid", [None]):
                with self.subTest(section=section, value=value):
                    data = copy.deepcopy(self.data)
                    data[section] = value
                    self.assertIn("UNKNOWN", [f["status"] for f in self.audit(data)])

    def test_root_must_be_an_object(self):
        for data in (None, [], 1, "invalid"):
            with self.subTest(data=data):
                self.require("INPUT_ERROR", "UNKNOWN", data=data)

    def cli(self, path):
        return subprocess.run([sys.executable, str(SCRIPT), str(path)],
                              capture_output=True, text=True, check=False)

    def test_cli_reads_input_immutably_and_outputs_geometry_scope(self):
        before = FIXTURE.read_bytes()
        result = self.cli(FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(FIXTURE.read_bytes(), before)
        payload = json.loads(result.stdout)
        self.assertEqual(set(payload), {"scope", "findings"})
        self.assertEqual(payload["scope"], "geometry_only_not_compliance")
        self.assertTrue(all(f["status"] == "PASS" for f in payload["findings"]))

    def test_cli_exit_status_fail_takes_precedence_over_unknown(self):
        self.data["graphs"][0]["matrix"] = [[None, None], [None, None]]
        self.data["required_facts"]["unknown"] = None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text(json.dumps(self.data), encoding="utf-8")
            self.assertEqual(self.cli(path).returncode, 1)

    def test_cli_unknown_exits_two(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text('{"schema_version":1,"units":"cm"}', encoding="utf-8")
            result = self.cli(path)
            self.assertEqual(result.returncode, 2)
            self.assertIn("UNKNOWN", [f["status"] for f in json.loads(result.stdout)["findings"]])

    def test_cli_invalid_json_is_structured_without_traceback(self):
        for contents in ("{", "{\"x\": NaN}", "{\"x\": Infinity}", "[]", "{\"x\": 1e999}",
                         "{\"schema_version\": 1, \"schema_version\": 2}"):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "input.json"
                path.write_text(contents, encoding="utf-8")
                result = self.cli(path)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["findings"][0]["code"], "INPUT_ERROR")
                self.assertNotIn("Traceback", result.stderr + result.stdout)

    def test_cli_rejects_large_integer_with_interpreter_digit_guard_disabled(self):
        contents = json.dumps(self.data)[:-1] + ', "extra_integer": ' + "9" * 100_000 + "}"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text(contents, encoding="utf-8")
            self.assertLess(path.stat().st_size, 10 * 1024 * 1024)
            result = subprocess.run(
                [sys.executable, "-X", "int_max_str_digits=0", str(SCRIPT), str(path)],
                capture_output=True, text=True, check=False, timeout=5,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["findings"][0]["code"], "INPUT_ERROR")
            self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_cli_integer_digit_limit_excludes_optional_minus_sign(self):
        for digits, expected_exit in ((128, 0), (129, 2)):
            for sign in ("", "-"):
                with self.subTest(digits=digits, sign=sign), tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "input.json"
                    contents = json.dumps(self.data)[:-1] + ', "extra_integer": ' + sign + "9" * digits + "}"
                    path.write_text(contents, encoding="utf-8")
                    result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                                            capture_output=True, text=True, check=False, timeout=5)
                    self.assertEqual(result.returncode, expected_exit, result.stdout + result.stderr)
                    if expected_exit:
                        self.assertEqual(json.loads(result.stdout)["findings"][0]["code"], "INPUT_ERROR")

    def test_cli_missing_or_oversize_file_is_structured(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            for exists in (False, True):
                if exists:
                    with path.open("wb") as handle:
                        handle.truncate(10 * 1024 * 1024 + 1)
                result = self.cli(path)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["findings"][0]["code"], "INPUT_ERROR")


if __name__ == "__main__":
    unittest.main()
