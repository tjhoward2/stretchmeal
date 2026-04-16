"""Tests for layered PSD scaling engine."""

from pathlib import Path

from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer
import pytest

from scaling.layered_psd import (
    LayerZone,
    _classify_layer,
    _ScaleContext,
    scale_layered_psd,
)


def _make_ctx(src_w=1000, src_h=1400, target_w=800, target_h=1120,
              bt=100, bb=100, bl=100, br=100):
    """Create a scale context for testing."""
    src_iw = src_w - bl - br
    src_ih = src_h - bt - bb
    tgt_iw = target_w - bl - br
    tgt_ih = target_h - bt - bb
    return _ScaleContext(
        src_w=src_w, src_h=src_h,
        target_w=target_w, target_h=target_h,
        bt=bt, bb=bb, bl=bl, br=br,
        interior_scale_x=tgt_iw / src_iw,
        interior_scale_y=tgt_ih / src_ih,
    )


def _make_test_psd(path: Path, width=1000, height=1400, border=100):
    """Create a multi-layer test PSD with border and interior layers."""
    psd = PSDImage.new("RGB", (width, height))

    # Background layer (full canvas)
    bg = Image.new("RGB", (width, height), (50, 50, 150))
    bg_layer = PixelLayer.frompil(bg, psd, "Background", compression=0)
    psd.append(bg_layer)

    # Border layer (top border strip)
    border_img = Image.new("RGB", (width - 2 * border, border), (200, 50, 50))
    border_layer = PixelLayer.frompil(border_img, psd, "Top Border", compression=0)
    border_layer.left = border
    border_layer.top = 0
    psd.append(border_layer)

    # Interior layer (centered in interior)
    interior_size = 200
    interior_img = Image.new("RGB", (interior_size, interior_size), (50, 200, 50))
    interior_layer = PixelLayer.frompil(interior_img, psd, "Center Motif", compression=0)
    interior_layer.left = width // 2 - interior_size // 2
    interior_layer.top = height // 2 - interior_size // 2
    psd.append(interior_layer)

    # Corner layer (top-left corner)
    corner_img = Image.new("RGB", (border, border), (200, 200, 50))
    corner_layer = PixelLayer.frompil(corner_img, psd, "Corner TL", compression=0)
    corner_layer.left = 0
    corner_layer.top = 0
    psd.append(corner_layer)

    # Group with a child layer
    group = Group.new(parent=psd, name="Details")
    detail_img = Image.new("RGB", (80, 80), (200, 100, 200))
    detail_layer = PixelLayer.frompil(detail_img, group, "Detail 1", compression=0)
    detail_layer.left = 300
    detail_layer.top = 400
    group.append(detail_layer)
    psd.append(group)

    psd.save(str(path))
    return psd


class TestClassifyLayer:
    """Test layer zone classification."""

    def test_full_span(self):
        ctx = _make_ctx()
        zone = _classify_layer(0, 0, 1000, 1400, ctx)
        assert zone == LayerZone.FULL_SPAN

    def test_interior(self):
        ctx = _make_ctx()
        zone = _classify_layer(300, 400, 700, 800, ctx)
        assert zone == LayerZone.INTERIOR

    def test_top_border(self):
        ctx = _make_ctx()
        zone = _classify_layer(200, 10, 800, 90, ctx)
        assert zone == LayerZone.BORDER

    def test_left_border(self):
        ctx = _make_ctx()
        zone = _classify_layer(10, 300, 90, 700, ctx)
        assert zone == LayerZone.BORDER

    def test_corner(self):
        ctx = _make_ctx()
        zone = _classify_layer(0, 0, 80, 80, ctx)
        assert zone == LayerZone.CORNER

    def test_bottom_right_corner(self):
        ctx = _make_ctx()
        zone = _classify_layer(920, 1320, 1000, 1400, ctx)
        assert zone == LayerZone.CORNER


class TestScaleLayeredPsd:
    """Test the full layered PSD scaling pipeline."""

    def test_output_is_created(self, tmp_path):
        """Should create an output PSD file."""
        input_psd = tmp_path / "input.psd"
        output_psd = tmp_path / "output.psd"
        _make_test_psd(input_psd)

        zone_map = {
            "border_top_px": 100, "border_bottom_px": 100,
            "border_left_px": 100, "border_right_px": 100,
            "master_width_px": 1000, "master_height_px": 1400,
        }

        scale_layered_psd(input_psd, output_psd, zone_map, 800, 1120)
        assert output_psd.exists()

    def test_output_dimensions(self, tmp_path):
        """Output PSD should have correct target dimensions."""
        input_psd = tmp_path / "input.psd"
        output_psd = tmp_path / "output.psd"
        _make_test_psd(input_psd)

        zone_map = {
            "border_top_px": 100, "border_bottom_px": 100,
            "border_left_px": 100, "border_right_px": 100,
            "master_width_px": 1000, "master_height_px": 1400,
        }

        scale_layered_psd(input_psd, output_psd, zone_map, 800, 1120)

        result = PSDImage.open(str(output_psd))
        assert result.width == 800
        assert result.height == 1120

    def test_layers_preserved(self, tmp_path):
        """All layers should be present in the output."""
        input_psd = tmp_path / "input.psd"
        output_psd = tmp_path / "output.psd"
        _make_test_psd(input_psd)

        zone_map = {
            "border_top_px": 100, "border_bottom_px": 100,
            "border_left_px": 100, "border_right_px": 100,
            "master_width_px": 1000, "master_height_px": 1400,
        }

        scale_layered_psd(input_psd, output_psd, zone_map, 800, 1120)

        result = PSDImage.open(str(output_psd))
        names = [l.name for l in result]
        assert "Background" in names
        assert "Top Border" in names
        assert "Center Motif" in names
        assert "Corner TL" in names
        assert "Details" in names

    def test_group_children_preserved(self, tmp_path):
        """Group children should be present in the output."""
        input_psd = tmp_path / "input.psd"
        output_psd = tmp_path / "output.psd"
        _make_test_psd(input_psd)

        zone_map = {
            "border_top_px": 100, "border_bottom_px": 100,
            "border_left_px": 100, "border_right_px": 100,
            "master_width_px": 1000, "master_height_px": 1400,
        }

        scale_layered_psd(input_psd, output_psd, zone_map, 800, 1120)

        result = PSDImage.open(str(output_psd))
        group = [l for l in result if l.name == "Details"][0]
        children = list(group)
        assert len(children) == 1
        assert children[0].name == "Detail 1"

    def test_corner_layer_not_resized(self, tmp_path):
        """Corner layers should keep original pixel dimensions."""
        input_psd = tmp_path / "input.psd"
        output_psd = tmp_path / "output.psd"
        _make_test_psd(input_psd)

        zone_map = {
            "border_top_px": 100, "border_bottom_px": 100,
            "border_left_px": 100, "border_right_px": 100,
            "master_width_px": 1000, "master_height_px": 1400,
        }

        scale_layered_psd(input_psd, output_psd, zone_map, 800, 1120)

        result = PSDImage.open(str(output_psd))
        corner = [l for l in result if l.name == "Corner TL"][0]
        assert corner.right - corner.left == 100
        assert corner.bottom - corner.top == 100

    def test_upscale(self, tmp_path):
        """Should work for upscaling too."""
        input_psd = tmp_path / "input.psd"
        output_psd = tmp_path / "output.psd"
        _make_test_psd(input_psd)

        zone_map = {
            "border_top_px": 100, "border_bottom_px": 100,
            "border_left_px": 100, "border_right_px": 100,
            "master_width_px": 1000, "master_height_px": 1400,
        }

        scale_layered_psd(input_psd, output_psd, zone_map, 1500, 2100)

        result = PSDImage.open(str(output_psd))
        assert result.width == 1500
        assert result.height == 2100

    def test_zero_borders(self, tmp_path):
        """Should handle borderless designs (all interior)."""
        input_psd = tmp_path / "input.psd"
        output_psd = tmp_path / "output.psd"

        psd = PSDImage.new("RGB", (400, 600))
        bg = Image.new("RGB", (400, 600), (100, 100, 100))
        psd.append(PixelLayer.frompil(bg, psd, "BG", compression=0))
        psd.save(str(input_psd))

        zone_map = {
            "border_top_px": 0, "border_bottom_px": 0,
            "border_left_px": 0, "border_right_px": 0,
            "master_width_px": 400, "master_height_px": 600,
        }

        scale_layered_psd(input_psd, output_psd, zone_map, 200, 300)

        result = PSDImage.open(str(output_psd))
        assert result.width == 200
        assert result.height == 300
