import torch

from tools.diagnose_ruler_direct import handoff_streams


def test_fixed_interpolation_uses_integer_stride4_centers():
    y, x = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    depth = (2 + 4 * x + 8 * y).float()[None, None]
    full = torch.ones(1, 1, 12, 16)
    result = {"depth": full, "fine": {"depth": depth, "ref_valid": torch.ones_like(depth, dtype=torch.bool)}}
    output = handoff_streams(result, full, 10.)
    fy, fx = torch.meshgrid(torch.arange(9), torch.arange(13), indexing="ij")
    expected = (2 + fx + 2 * fy).float()[None, None]
    torch.testing.assert_close(output["fine_bilinear_aligned"][..., :9, :13], expected)
    # x=3距离x=4格点更近；不能误用重复4次得到x=0的值。
    assert output["fine_nearest_aligned"][0, 0, 0, 3] == 6
    torch.testing.assert_close(output["fine_bilinear_aligned"][..., ::4, ::4], depth)


def test_neighbor_oracle_excludes_padding_and_preserves_unobserved_fallback():
    depth = torch.full((1, 1, 2, 2), 20.)
    truth = torch.full((1, 1, 8, 8), 2.)
    result = {"depth": truth, "fine": {"depth": depth, "ref_valid": torch.ones_like(depth, dtype=torch.bool)}}
    output = handoff_streams(result, truth, 10.)
    assert (output["fine_neighbor3x3_oracle"] == 20).all()
    result["fine"]["ref_valid"][:] = False
    output = handoff_streams(result, truth, 10.)
    for name in ("fine_neighbor3x3_oracle", "fine_nearest_aligned", "fine_bilinear_aligned"):
        assert (output[name] == 10).all()


def test_fixed_interpolation_does_not_depend_on_gt():
    depth = torch.tensor([[[[3., 7.], [12., 20.]]]])
    truth = torch.full((1, 1, 8, 8), 3.)
    result = {"depth": truth, "fine": {"depth": depth, "ref_valid": torch.ones_like(depth, dtype=torch.bool)}}
    first = handoff_streams(result, truth, 10.)
    second = handoff_streams(result, truth + 10, 10.)
    for name in ("fine_nearest_aligned", "fine_bilinear_aligned"):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    assert not torch.equal(first["fine_neighbor3x3_oracle"], second["fine_neighbor3x3_oracle"])
