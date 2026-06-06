import unittest

import torch
import torch.nn as nn

from sglang.srt.model_loader import remote_sync_state as rss
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _Attn(rss.RemoteSyncStateMixin, nn.Module):
    _REMOTE_SYNC_ATTRS = (
        "w_kc",
        "w_vc",
        "w_scale",
        "w_scale_v",
        "use_deep_gemm_bmm",
    )

    def __init__(self):
        super().__init__()
        self.weight_scale_inv = nn.Parameter(torch.randn(4, 4))
        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Attn(), _Attn()])


def _simulate_transport(extra, manifest):
    """Mimic NCCL/RDMA: transfer only the raw storage bytes."""
    recv = {}
    for e in manifest:
        src = rss.storage_byte_view(extra[e["name"]])
        buf = torch.empty(e["storage_nbytes"], dtype=torch.uint8)
        buf.copy_(src)
        recv[e["name"]] = buf
    return recv


class TestRemoteSyncState(CustomTestCase):
    def _build_seed(self):
        seed = _Model()
        for i, layer in enumerate(seed.layers):
            base = torch.randn(3, 5, 7)
            # non-contiguous, mimics `.transpose(1,2).contiguous().transpose(1,2)`
            layer.w_kc = base.transpose(1, 2).contiguous().transpose(1, 2)
            layer.w_vc = torch.randn(3, 7, 5).contiguous().transpose(1, 2)
            layer.w_scale = torch.tensor(float(i) + 0.5)
            layer.w_scale_v = torch.randn(2, 2)
            layer.use_deep_gemm_bmm = True
            layer.weight_scale_inv.format_ue8m0 = True
        return seed

    def test_noncontiguous_roundtrip_and_apply(self):
        seed = self._build_seed()
        self.assertFalse(seed.layers[0].w_kc.is_contiguous())

        extra, payload = rss.collect_remote_sync_payload(seed)
        self.assertTrue(rss.payload_uses_sync_providers(payload))
        # 4 tensor attrs * 2 layers = 8
        self.assertEqual(len(extra), 8)

        recv = _simulate_transport(extra, payload["manifest"])

        client = _Model()
        rss.apply_remote_sync_state(
            client, payload["manifest"], recv, payload["metadata"]
        )

        for i in range(2):
            s, c = seed.layers[i], client.layers[i]
            self.assertTrue(torch.equal(s.w_kc, c.w_kc))
            self.assertEqual(s.w_kc.stride(), c.w_kc.stride())
            self.assertFalse(c.w_kc.is_contiguous())
            self.assertTrue(torch.equal(s.w_vc, c.w_vc))
            self.assertEqual(s.w_vc.stride(), c.w_vc.stride())
            self.assertTrue(torch.equal(s.w_scale, c.w_scale))
            self.assertTrue(torch.equal(s.w_scale_v, c.w_scale_v))
            self.assertIs(c.use_deep_gemm_bmm, True)

    def test_tensor_object_flag_synced(self):
        seed = self._build_seed()
        _, payload = rss.collect_remote_sync_payload(seed)
        flags = payload["metadata"]["tensor_flags"]
        self.assertIn("layers.0.weight_scale_inv", flags)
        self.assertEqual(flags["layers.0.weight_scale_inv"], {"format_ue8m0": True})

        extra, payload = rss.collect_remote_sync_payload(seed)
        recv = _simulate_transport(extra, payload["manifest"])
        client = _Model()
        rss.apply_remote_sync_state(
            client, payload["manifest"], recv, payload["metadata"]
        )
        sd = client.state_dict(keep_vars=True)
        self.assertTrue(getattr(sd["layers.0.weight_scale_inv"], "format_ue8m0"))

    def test_scalar_w_scale_routed_to_metadata(self):
        seed = _Model()
        seed.layers[0].w_kc = torch.randn(2, 3)
        seed.layers[0].w_scale = 2.5  # float -> metadata channel

        _, payload = rss.collect_remote_sync_payload(seed)
        self.assertEqual(
            payload["metadata"]["module_attrs"]["layers.0"]["w_scale"], 2.5
        )
        extra_local_names = {m["name"].rsplit(".", 1)[1] for m in payload["manifest"]}
        self.assertNotIn("w_scale", extra_local_names)

    def test_no_providers_payload(self):
        plain = nn.Linear(3, 3)
        _, payload = rss.collect_remote_sync_payload(plain)
        self.assertFalse(rss.payload_uses_sync_providers(payload))
        self.assertEqual(payload["manifest"], [])


if __name__ == "__main__":
    unittest.main()
