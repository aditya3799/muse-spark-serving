"""Tests for route conservation, falsifiable scenarios, and isolated source edits."""

import ast
import copy
import os
import unittest
from pathlib import Path

from marlin_bench import SOURCE, load_method
from tile_study import (
    BLOCKS,
    PATTERNS,
    choose_block,
    make_routes,
    map_routes,
    pack_reference,
    verify_pack,
)


class GeometryTests(unittest.TestCase):
    def test_expected_ep8_examples(self):
        self.assertEqual(
            [choose_block(m, 8, 32, 256, "baseline") for m in (8, 32, 128, 256)],
            [8, 16, 48, 64],
        )
        self.assertEqual(
            [choose_block(m, 8, 32, 256, "ep_density") for m in (8, 32, 128, 256)],
            [8, 8, 8, 16],
        )

    def test_no_ep_means_no_policy_difference(self):
        for tokens in range(2049):
            self.assertEqual(
                choose_block(tokens, 8, 32, 32, "baseline"),
                choose_block(tokens, 8, 32, 32, "ep_density"),
            )

    def test_boundary_uses_strict_less_than(self):
        self.assertEqual(choose_block(71, 8, 10, 10, "baseline"), 64)
        self.assertEqual(choose_block(9, 8, 10, 10, "baseline"), 16)
        self.assertEqual(choose_block(8, 8, 10, 10, "baseline"), 8)

    def test_reject_invalid_geometry(self):
        for args in (
            (-1, 8, 32, 256, "baseline"),
            (1, 8, 0, 256, "baseline"),
            (1, 8, 257, 256, "baseline"),
            (1, 8, 32, 256, "fixed_7"),
        ):
            with self.assertRaises(ValueError):
                choose_block(*args)

    def test_remote_is_not_globally_empty(self):
        rows = make_routes(33, 8, 256, 8, "all_remote", 0)
        self.assertTrue(all(x < 0 for x in map_routes(rows, 32, 0)))
        self.assertEqual(
            sum(sum(x >= 0 for x in map_routes(rows, 32, r)) for r in range(8)), 33 * 8
        )

    def test_unique_topk_and_partition_conservation(self):
        for pattern in PATTERNS:
            for seed in range(3):
                rows = make_routes(129, 8, 256, 8, pattern, seed)
                for row in rows:
                    live = [x for x in row if x >= 0]
                    self.assertEqual(len(live), len(set(live)))
                self.assertEqual(
                    sum(sum(x >= 0 for x in map_routes(rows, 32, r)) for r in range(8)),
                    sum(x >= 0 for row in rows for x in row),
                )

    def test_hot_rank_exposes_balanced_assumption(self):
        rows = make_routes(128, 8, 256, 8, "hot_rank", 0)
        counts = [sum(x >= 0 for x in map_routes(rows, 32, r)) for r in range(8)]
        self.assertEqual(counts, [1024, 0, 0, 0, 0, 0, 0, 0])

    def test_route_identity_survives_each_tile(self):
        for pattern in PATTERNS:
            rows = make_routes(9, 8, 256, 8, pattern, 2)
            for rank in range(8):
                mapped = map_routes(rows, 32, rank)
                for block in BLOCKS:
                    token_ids, experts = pack_reference(mapped, 32, block)
                    verify_pack(mapped, token_ids, experts, block)

    def test_detects_lost_duplicate_and_misassigned_routes(self):
        mapped = [-1, 0, 1, 0]
        token_ids, experts = pack_reference(mapped, 2, 8)
        for changed in (token_ids[:-1], [4] + token_ids[1:], [1] + token_ids[1:]):
            with self.assertRaises(AssertionError):
                verify_pack(mapped, changed, experts, 8)

    def test_smaller_tiles_can_increase_blocks(self):
        mapped = [0] * 17
        small, small_experts = pack_reference(mapped, 1, 8)
        large, large_experts = pack_reference(mapped, 1, 32)
        self.assertLess(len(small), len(large))
        self.assertGreater(len(small_experts), len(large_experts))


class SourceIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(
            os.environ.get("TILE_SGLANG_ROOT", r"D:\OpenKernelChallenge\serving\sglang")
        )
        if not (cls.root / SOURCE).exists():
            raise unittest.SkipTest("Set TILE_SGLANG_ROOT to the frozen checkout")

    def test_baseline_matches_exact_source_ast(self):
        tree = ast.parse((self.root / SOURCE).read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MuseSpark1xFp4Experts"
        )
        original = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_marlin_expert_outputs"
        )
        _, text = load_method(self.root, "baseline")
        self.assertEqual(ast.dump(original), ast.dump(ast.parse(text).body[0]))

    def test_candidate_changes_only_selector(self):
        _, text = load_method(self.root, "baseline")
        original = ast.parse(text).body[0]
        selector_index = next(
            i
            for i, n in enumerate(original.body)
            if isinstance(n, ast.For)
            and getattr(n.target, "id", None) == "block_size_m"
        )
        for policy in (
            "ep_density",
            "fixed_8",
            "fixed_16",
            "fixed_32",
            "fixed_48",
            "fixed_64",
        ):
            _, text = load_method(self.root, policy)
            candidate = ast.parse(text).body[0]
            self.assertIsInstance(candidate.body[selector_index], ast.Assign)
            restored = copy.deepcopy(candidate)
            restored.body[selector_index] = original.body[selector_index]
            self.assertEqual(ast.dump(restored), ast.dump(original))

    def test_sorted_ep_only_adds_explicit_gemm_contract(self):
        _, original_text = load_method(self.root, "baseline")
        _, candidate_text = load_method(self.root, "sorted_baseline")
        original = ast.parse(original_text).body[0]
        candidate = ast.parse(candidate_text).body[0]
        changed = 0
        for node in ast.walk(candidate):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "moe_wna16_marlin_gemm"
            ):
                explicit = [k for k in node.keywords if k.arg == "allow_sorted_ep"]
                self.assertEqual(len(explicit), 1)
                self.assertEqual(
                    ast.dump(explicit[0].value),
                    ast.dump(ast.parse("expert_map is not None", mode="eval").body),
                )
                node.keywords.remove(explicit[0])
                changed += 1
        self.assertEqual(changed, 2)
        self.assertEqual(ast.dump(candidate), ast.dump(original))


if __name__ == "__main__":
    unittest.main(verbosity=2)
