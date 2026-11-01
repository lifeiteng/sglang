import unittest
from unittest.mock import patch

import torch

class TestCandidateIndexerGating(unittest.TestCase):
    def test_fp4_indexer_device_capability(self):
        from sglang.srt.layers.attention.dsv4 import v41_indexer

        check = v41_indexer.supports_deep_gemm_fp4_indexer
        self.addCleanup(check.cache_clear)
        for capability, expected in (
            ((9, 0), False),
            ((10, 0), True),
            ((10, 3), True),
            ((12, 0), False),
            ((12, 1), False),
        ):
            with (
                self.subTest(capability=capability),
                patch.object(v41_indexer, "is_cuda", return_value=True),
                patch("torch.cuda.get_device_capability", return_value=capability),
            ):
                check.cache_clear()
                self.assertEqual(check(), expected)
                if not expected:
                    v41_indexer._use_deep_gemm_prefill.cache_clear()
                    self.addCleanup(v41_indexer._use_deep_gemm_prefill.cache_clear)
                    full = v41_indexer.make_full_topk_indexer(
                        token_to_kv_pool=None, req_to_token=None
                    )
                    self.assertFalse(full.use_deep_gemm_prefill)
                    self.assertFalse(full.use_deep_gemm_decode)
                    prefill, decode = v41_indexer.make_candidate_indexer(
                        token_to_kv_pool=None,
                        req_to_token=None,
                        page_size=256,
                        candidate_topk_blocks=128,
                        candidate_block_size=128,
                    )
                    self.assertIs(prefill, decode)
                    self.assertFalse(prefill.use_deep_gemm_prefill)
        with (
            patch.object(v41_indexer, "is_cuda", return_value=False),
            patch(
                "torch.cuda.get_device_capability",
                side_effect=AssertionError("non-CUDA"),
            ),
        ):
            check.cache_clear()
            self.assertFalse(check())

    def test_candidate_indexer_gating(self):
        from sglang.srt.layers.attention.dsv4 import v41_indexer
        from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
            DenseBlocksBackend,
        )

        def sm100(value):
            return patch.object(
                v41_indexer, "supports_deep_gemm_fp4_indexer", lambda: value
            )

        def make(topk_blocks, block_size):
            # The prefill choice reads the real DeepGEMM build and is cached.
            with patch.object(v41_indexer, "_use_deep_gemm_prefill", lambda: False):
                return v41_indexer.make_candidate_indexer(
                    token_to_kv_pool=None,
                    req_to_token=torch.zeros((1, 1), dtype=torch.int32),
                    page_size=256,
                    candidate_topk_blocks=topk_blocks,
                    candidate_block_size=block_size,
                )

        flag = "sglang.srt.layers.deep_gemm_wrapper.configurer.DEEPGEMM_PAGED_SPARSE_MQA_LOGITS"
        # V4 configs carry no candidate fields (0 / 0): no layer is a candidate
        # source, so Blackwell must not build the DeepGEMM backend for them.
        # Hopper serves candidates on dense block ids.
        for is_sm100, topk_blocks, block_size in ((True, 0, 0), (False, 2048, 8)):
            with self.subTest(sm100=is_sm100, topk_blocks=topk_blocks):
                with sm100(is_sm100), patch(flag, False):
                    prefill, decode = make(topk_blocks, block_size)
                self.assertIsInstance(prefill, DenseBlocksBackend)
                self.assertIsInstance(decode, DenseBlocksBackend)
        # Blackwell without DeepGEMM's sparse logits fails instead of falling back.
        with sm100(True), patch(flag, False):
            with self.assertRaises(RuntimeError):
                make(2048, 8)


if __name__ == "__main__":
    unittest.main()
