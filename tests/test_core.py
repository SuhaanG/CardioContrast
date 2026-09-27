"""
tests/test_core.py — CPU-only checks for metrics, EF, loss and model wiring.
Run:  python tests/test_core.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.contrastive import ContrastiveAnatomicalLoss, anatomical_repulsion_loss  # noqa: E402
from lib.mask_predictor import DecoderCrossAttention, SimpleDecoding  # noqa: E402
from lib.metrics import binary_metrics, ejection_fraction, leakage, simpson_biplane_volume  # noqa


def ellipse(shape, cy, cx, ry, rx):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1


def test_metrics():
    a = np.zeros((100, 100), bool)
    a[20:60, 20:60] = True
    m = binary_metrics(a, a, (0.5, 0.5))
    assert abs(m["dice"] - 1) < 1e-9 and m["hd95"] == 0 and m["mad"] == 0
    b = np.zeros_like(a)
    b[20:60, 30:70] = True                       # shifted 10 px = 5 mm
    m = binary_metrics(b, a, (0.5, 0.5))
    assert abs(m["dice"] - 0.75) < 1e-9, m
    assert abs(m["hd"] - 5.0) < 1e-6, m
    assert binary_metrics(np.zeros_like(a), a, (1, 1))["dice"] == 0
    assert binary_metrics(np.zeros_like(a), np.zeros_like(a), (1, 1))["dice"] == 1
    gt = np.zeros((100, 100), np.uint8)
    gt[20:60, 20:60] = 1
    gt[20:60, 60:80] = 2
    pred = np.zeros((100, 100), bool)
    pred[20:60, 40:80] = True                    # half in LV, half in myo
    assert abs(leakage(pred, gt, 1) - 0.5) < 1e-9
    print("metrics ok")


def test_simpson():
    sp = (0.3, 0.3)
    shape = (300, 200)
    a_mm, b_mm = 30.0, 12.0                      # semi-axes (long, short)
    ed = ellipse(shape, 150, 100, a_mm / sp[0], b_mm / sp[1])
    v = simpson_biplane_volume(ed, sp, ed, sp)
    true = 4 / 3 * np.pi * a_mm * b_mm * b_mm / 1000
    assert abs(v - true) / true < 0.05, (v, true)
    es = ellipse(shape, 150, 100, 0.85 * a_mm / sp[0], 0.8 * b_mm / sp[1])
    ef = ejection_fraction(v, simpson_biplane_volume(es, sp, es, sp))
    true_ef = 100 * (1 - 0.85 * 0.8 * 0.8)
    assert abs(ef - true_ef) < 3, (ef, true_ef)
    # with an LA touching the base
    la = ellipse(shape, 150 + a_mm / sp[0] + 25, 100, 25, 30) & ~ed
    v_la = simpson_biplane_volume(ed, sp, ed, sp, la, la)
    assert abs(v_la - true) / true < 0.10, (v_la, true)
    print("simpson ok  (V={:.1f} mL vs {:.1f}; EF={:.1f} vs {:.1f})".format(v, true, ef, true_ef))


def test_contrastive():
    z = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    img = torch.tensor([0, 0, 0, 1, 1, 1])
    st = torch.tensor([1, 2, 3, 1, 2, 3])
    loss, cos, n = anatomical_repulsion_loss(z, img, st)
    assert n == 6 and loss.item() > 0
    # no negatives -> zero loss that still has a graph
    loss0, _, n0 = anatomical_repulsion_loss(z, torch.arange(6), st)
    assert n0 == 0 and loss0.item() == 0
    # v2 default: no projection head -> no parameters; gradient reaches features
    m2 = ContrastiveAnatomicalLoss(32)
    assert m2.proj_head == "none" and m2.pool_region == "union"
    assert sum(p.numel() for p in m2.parameters()) == 0
    # identical non-negative features for all prompts -> loss stays well above 0
    f_same = torch.rand(1, 32, 11, 11).repeat(3, 1, 1, 1).requires_grad_(True)
    u = torch.ones(3, 44, 44)
    l2, c2, _ = m2(f_same, torch.randn(3, 2, 44, 44), torch.zeros(3, dtype=torch.long),
                   torch.tensor([1, 2, 3]), union=u)
    assert c2 > 0.99 and l2.item() > 0.5, (c2, l2.item())
    l2.backward()
    assert f_same.grad is not None and f_same.grad.abs().sum() > 0
    for pool in ("pred", "gt", "union"):
        m = ContrastiveAnatomicalLoss(32, 32, 16, pool_region=pool, proj_head="mlp")
        feats = torch.randn(6, 32, 11, 11, requires_grad=True)
        logits = torch.randn(6, 2, 44, 44, requires_grad=True)
        tgt = (torch.rand(6, 44, 44) > 0.5).long()
        l, _, _ = m(feats, logits, img, st, target=tgt, union=tgt.float())
        l.backward()
        assert feats.grad is not None
        assert logits.grad is None, "pool weights must be detached by default"
    print("contrastive ok")


def test_decoder_gate():
    torch.manual_seed(0)
    dec = SimpleDecoding(8 * 32)                       # tiny dims
    x4, x3 = torch.randn(2, 256, 4, 4), torch.randn(2, 128, 8, 8)
    x2, x1 = torch.randn(2, 64, 16, 16), torch.randn(2, 32, 32, 32)
    lang = torch.randn(2, 768, 5)
    mask = torch.ones(2, 5, 1)
    dec.eval()
    a = dec(x4, x3, x2, x1)
    b = dec(x4, x3, x2, x1, lang_feat=lang, lang_mask=mask)
    assert torch.allclose(a, b, atol=1e-5), "gate=0 at init must equal the baseline decoder"
    # no batch coupling: a sample's output must not depend on the rest of the batch
    dec.train()
    full = dec(x4, x3, x2, x1, lang_feat=lang, lang_mask=mask)
    single = dec(x4[:1], x3[:1], x2[:1], x1[:1], lang_feat=lang[:1], lang_mask=mask[:1])
    assert torch.allclose(full[:1], single, atol=1e-5), "decoder output depends on batch"
    dec.eval()
    ca = DecoderCrossAttention(16, 768, 4)
    with torch.no_grad():
        ca.gate.fill_(1.0)
    m = torch.tensor([[1, 1, 0, 0, 0]]).unsqueeze(-1)
    l1 = torch.randn(1, 768, 5)
    l2 = l1.clone()
    l2[:, :, 2:] = torch.randn(1, 768, 3)             # change only padded tokens
    v = torch.randn(1, 16, 4, 4)
    assert torch.allclose(ca(v, l1, m), ca(v, l2, m), atol=1e-5), "padding must be ignored"
    print("decoder ok")


if __name__ == "__main__":
    test_metrics()
    test_simpson()
    test_contrastive()
    test_decoder_gate()
    print("ALL TESTS PASSED")
