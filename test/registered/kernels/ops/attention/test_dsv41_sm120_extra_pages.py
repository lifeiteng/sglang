"""V4.1 ratio-2 footer pages must be repacked for SM120 prefill."""

import unittest

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _flash_mla_sm120_prefill,
    _gather_and_dequant,
    _sm120_sparse_decode_fwd,
    _split_kv_pages_to_64,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def make_cache(page_size, pages=4):
    stride = ((page_size * 584 + 575) // 576) * 576
    raw = torch.zeros(pages, stride, device="cuda", dtype=torch.uint8)
    data = raw[:, : page_size * 576].view(pages, page_size, 576)
    data[..., :448] = (
        torch.randn(pages, page_size, 448, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    data[..., 448:] = (
        torch.randn(pages, page_size, 64, device="cuda")
        .to(torch.bfloat16)
        .view(torch.uint8)
    )
    raw[:, page_size * 576 : page_size * 584] = 127
    return raw.as_strided((pages, page_size, 1, 584), (stride, 584, 584, 1))


def reference_split(cache):
    # Independent byte copies, without the production Triton page splitter.
    pages, page_size = cache.shape[:2]
    stride = cache.stride(0)
    source = cache.as_strided((pages, stride), (stride, 1))
    ratio = page_size // 64
    dest_stride = ((64 * 584 + 575) // 576) * 576
    dest = torch.zeros(
        pages * ratio, dest_stride, dtype=torch.uint8, device=cache.device
    )
    for page in range(pages):
        for sub in range(ratio):
            dest[page * ratio + sub, : 64 * 576] = source[
                page, sub * 64 * 576 : (sub + 1) * 64 * 576
            ]
            dest[page * ratio + sub, 64 * 576 : 64 * 584] = source[
                page,
                page_size * 576 + sub * 64 * 8 : page_size * 576 + (sub + 1) * 64 * 8,
            ]
    return dest.as_strided((pages * ratio, 64, 1, 584), (dest_stride, 584, 584, 1))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestExtraPages(unittest.TestCase):
    def test_repacking_preserves_values_and_independent_swa_buffer(self):
        torch.manual_seed(42)
        swa, extra = make_cache(256), make_cache(128)
        idx = torch.tensor([0, 63, 64, 127, 128, 511], device="cuda")
        split_swa = _split_kv_pages_to_64(swa, 256, touched_indices=idx)
        before = split_swa.clone()
        split_extra = _split_kv_pages_to_64(
            extra, 128, touched_indices=idx, buffer_name="extra"
        )
        self.assertNotEqual(split_swa.data_ptr(), split_extra.data_ptr())
        self.assertTrue(torch.equal(before, split_swa))
        for original, split, page_size in (
            (swa, split_swa, 256),
            (extra, split_extra, 128),
        ):
            torch.testing.assert_close(
                _gather_and_dequant(original, idx, page_size),
                _gather_and_dequant(split, idx, 64),
                rtol=0,
                atol=0,
            )

    def test_repacking_cuda_graph_replays_current_pages(self):
        extra = make_cache(128)
        idx = torch.tensor([0, 63, 64, 127, 128, 511], device="cuda")
        for _ in range(3):
            _split_kv_pages_to_64(extra, 128, touched_indices=idx, buffer_name="extra")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            split = _split_kv_pages_to_64(
                extra, 128, touched_indices=idx, buffer_name="extra"
            )
        extra.as_strided((4, extra.stride(0)), (extra.stride(0), 1))[
            :, : 128 * 576
        ].zero_()
        graph.replay()
        torch.testing.assert_close(
            _gather_and_dequant(extra, idx, 128),
            _gather_and_dequant(split, idx, 64),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
        "requires SM120 FlashInfer",
    )
    def test_prefill_matches_reference(self):
        torch.manual_seed(43)
        swa, extra = make_cache(256), make_cache(128)
        q = torch.randn(65, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, 1024, (65, 1, 128), device="cuda", dtype=torch.int32)
        extra_idx = torch.randint(
            0, 512, (65, 1, 512), device="cuda", dtype=torch.int32
        )
        idx[..., -3:] = -1
        extra_idx[..., -7:] = -1
        sink = torch.randn(8, device="cuda")
        for lengths in (False, True):
            with self.subTest(lengths=lengths):
                args = dict(
                    q=q,
                    k_cache=swa,
                    indices=idx,
                    topk_length=(
                        torch.full((65,), 100, device="cuda", dtype=torch.int32)
                        if lengths
                        else None
                    ),
                    attn_sink=sink,
                    head_dim_v=512,
                    softmax_scale=512**-0.5,
                    extra_k_cache=extra,
                    extra_indices=extra_idx,
                    extra_topk_length=(
                        torch.full((65,), 400, device="cuda", dtype=torch.int32)
                        if lengths
                        else None
                    ),
                )
                expected = _sm120_sparse_decode_fwd(**args)[0]
                actual = _flash_mla_sm120_prefill(**args)[0]
                reference_args = dict(
                    args,
                    k_cache=reference_split(swa),
                    extra_k_cache=reference_split(extra),
                )
                reference_output = _flash_mla_sm120_prefill(**reference_args)[0]
                torch.testing.assert_close(actual, reference_output, rtol=0, atol=0)
                # FlashInfer quantizes internal Q/probabilities to FP8; this is
                # an approximate BF16 reference, separate from exact repacking.
                relative_rmse = (
                    actual.float() - expected.float()
                ).square().mean().sqrt() / expected.float().square().mean().sqrt()
                self.assertLess(relative_rmse.item(), 0.04)
                self.assertLess(
                    (actual.float() - expected.float()).abs().max().item(), 0.02
                )


if __name__ == "__main__":
    unittest.main()
