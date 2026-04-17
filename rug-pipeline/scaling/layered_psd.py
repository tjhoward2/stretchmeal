"""Layered PSD scaling engine.

Scales a PSD file to a new rug size while preserving layer structure.
Each layer is classified as border, interior, full-span, or corner,
and scaled accordingly using 9-slice logic.

Pixel layers and groups are fully preserved. Unsupported layer types
(text, smart objects, adjustment layers) are rasterized with a warning.
"""

import logging
from enum import Enum
from pathlib import Path

from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer
from psd_tools.constants import BlendMode

logger = logging.getLogger(__name__)


class LayerZone(Enum):
    """Classification of a layer's position relative to border zones."""
    FULL_SPAN = "full_span"    # covers most of the canvas → 9-slice
    BORDER = "border"          # centered in border zone → reposition only
    INTERIOR = "interior"      # centered in interior → uniform scale + reposition
    CORNER = "corner"          # fits within a corner → reposition only


def scale_layered_psd(
    input_path: Path,
    output_path: Path,
    zone_map: dict,
    target_w: int,
    target_h: int,
) -> Path:
    """Scale a layered PSD to new dimensions using border-aware logic.

    Args:
        input_path: Path to the source PSD file.
        output_path: Path for the scaled output PSD.
        zone_map: Border zone dict with border_*_px and master_*_px fields.
        target_w: Target width in pixels.
        target_h: Target height in pixels.

    Returns:
        Path to the output PSD file.
    """
    psd = PSDImage.open(str(input_path))
    src_w, src_h = psd.width, psd.height

    logger.info(
        "Scaling layered PSD %dx%d → %dx%d (%d layers)",
        src_w, src_h, target_w, target_h, len(list(_iter_all_layers(psd))),
    )

    bt = zone_map.get("border_top_px", 0)
    bb = zone_map.get("border_bottom_px", 0)
    bl = zone_map.get("border_left_px", 0)
    br = zone_map.get("border_right_px", 0)

    # Scale factors for interior region
    src_interior_w = src_w - bl - br
    src_interior_h = src_h - bt - bb
    tgt_interior_w = target_w - bl - br
    tgt_interior_h = target_h - bt - bb

    if tgt_interior_w <= 0 or tgt_interior_h <= 0:
        logger.warning("Target too small for borders, falling back to uniform scale")
        scale_x = target_w / src_w
        scale_y = target_h / src_h
        tgt_interior_w = int(src_interior_w * scale_x)
        tgt_interior_h = int(src_interior_h * scale_y)

    interior_scale_x = tgt_interior_w / src_interior_w if src_interior_w > 0 else 1.0
    interior_scale_y = tgt_interior_h / src_interior_h if src_interior_h > 0 else 1.0

    ctx = _ScaleContext(
        src_w=src_w, src_h=src_h,
        target_w=target_w, target_h=target_h,
        bt=bt, bb=bb, bl=bl, br=br,
        interior_scale_x=interior_scale_x,
        interior_scale_y=interior_scale_y,
    )

    # Create new PSD at target size
    # PSDImage.new() expects a string mode like "RGB", not the ColorMode enum
    mode_str = {1: "L", 3: "RGB", 4: "CMYK"}.get(psd.channels, "RGB")
    new_psd = PSDImage.new(
        mode=mode_str,
        size=(target_w, target_h),
        depth=psd.depth,
    )

    # Check PSD format size limit (~2 GB for layer data, ~30000px max dimension)
    # PSD format uses 32-bit offsets, so total layer data must be < 4 GB
    estimated_bytes = target_w * target_h * psd.channels * len(list(psd)) * 2
    if estimated_bytes > 3_500_000_000:
        logger.error(
            "Target size %dx%d with %d layers would exceed PSD format limits (~4 GB). "
            "Try a lower DPI or fewer layers. PSB (Large Document Format) is not yet supported.",
            target_w, target_h, len(list(psd)),
        )
        raise ValueError(
            f"Output too large for PSD format ({estimated_bytes / 1e9:.1f} GB estimated). "
            f"Reduce DPI or target dimensions."
        )

    # Process layers
    rasterized_count = 0
    for layer in psd:
        result = _process_layer(layer, new_psd, ctx)
        if result == "rasterized":
            rasterized_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_psd.save(str(output_path))

    layer_count = len(list(_iter_all_layers(new_psd)))
    logger.info(
        "Saved scaled PSD: %s (%dx%d, %d layers, %d rasterized)",
        output_path, target_w, target_h, layer_count, rasterized_count,
    )

    return output_path


class _ScaleContext:
    """Holds scaling parameters passed through recursive layer processing."""

    def __init__(self, src_w, src_h, target_w, target_h,
                 bt, bb, bl, br, interior_scale_x, interior_scale_y):
        self.src_w = src_w
        self.src_h = src_h
        self.target_w = target_w
        self.target_h = target_h
        self.bt = bt
        self.bb = bb
        self.bl = bl
        self.br = br
        self.interior_scale_x = interior_scale_x
        self.interior_scale_y = interior_scale_y


def _process_layer(layer, parent, ctx: _ScaleContext) -> str | None:
    """Process a single layer or group, adding to parent.

    Returns "rasterized" if the layer was rasterized from a non-pixel type.
    """
    if isinstance(layer, Group):
        new_group = Group.new(parent=parent, name=layer.name)
        new_group.opacity = layer.opacity
        new_group.blend_mode = layer.blend_mode

        for child in layer:
            _process_layer(child, new_group, ctx)

        parent.append(new_group)
        logger.debug("Group: %s (%d children)", layer.name, len(list(layer)))
        return None

    # Get pixel data — rasterize if needed
    pil_image, was_rasterized = _extract_pixels(layer)
    if pil_image is None:
        logger.warning("Skipping empty layer: %s", layer.name)
        return None

    # Classify layer zone
    zone = _classify_layer(
        layer.left, layer.top, layer.right, layer.bottom,
        ctx,
    )

    # Scale and reposition
    scaled_image, new_left, new_top = _scale_layer_pixels(
        pil_image, layer.left, layer.top, zone, ctx,
    )

    # Create new pixel layer
    new_layer = PixelLayer.frompil(
        scaled_image, parent, layer.name, compression=0,
    )
    new_layer.top = new_top
    new_layer.left = new_left
    new_layer.opacity = layer.opacity
    new_layer.blend_mode = layer.blend_mode

    parent.append(new_layer)

    logger.debug(
        "Layer: %s [%s] (%d,%d)-(%d,%d) → (%d,%d) size %dx%d%s",
        layer.name, zone.value,
        layer.left, layer.top, layer.right, layer.bottom,
        new_left, new_top, scaled_image.width, scaled_image.height,
        " (rasterized)" if was_rasterized else "",
    )

    return "rasterized" if was_rasterized else None


def _extract_pixels(layer) -> tuple[Image.Image | None, bool]:
    """Extract pixel data from a layer, rasterizing if needed.

    RGBA layers are flattened to RGB by compositing alpha onto white.
    This prevents transparency artifacts in viewers that don't composite
    PSD layers properly (e.g., macOS Preview).

    Returns (PIL Image, was_rasterized).
    """
    if isinstance(layer, PixelLayer):
        try:
            img = layer.topil()
            if img is not None:
                return _flatten_alpha(img), False
        except Exception as e:
            logger.warning("Failed to read pixels from %s: %s", layer.name, e)

    # Try to composite/rasterize the layer
    try:
        img = layer.composite()
        if img is not None:
            logger.warning(
                "Rasterized layer '%s' (type: %s) — editability lost for this layer",
                layer.name, type(layer).__name__,
            )
            return _flatten_alpha(img), True
    except Exception as e:
        logger.warning("Failed to rasterize %s: %s", layer.name, e)

    return None, False


def _classify_layer(
    left: int, top: int, right: int, bottom: int,
    ctx: _ScaleContext,
) -> LayerZone:
    """Classify a layer as full-span, border, interior, or corner."""
    layer_w = right - left
    layer_h = bottom - top

    # Full-span: covers more than 80% of the canvas in both dimensions
    if layer_w > ctx.src_w * 0.8 and layer_h > ctx.src_h * 0.8:
        return LayerZone.FULL_SPAN

    # Center point of the layer
    cx = (left + right) / 2
    cy = (top + bottom) / 2

    in_left_border = cx < ctx.bl
    in_right_border = cx > ctx.src_w - ctx.br
    in_top_border = cy < ctx.bt
    in_bottom_border = cy > ctx.src_h - ctx.bb

    in_h_border = in_left_border or in_right_border
    in_v_border = in_top_border or in_bottom_border

    # Corner: center is in both a horizontal and vertical border zone
    if in_h_border and in_v_border:
        return LayerZone.CORNER

    # Border: center is in one border zone
    if in_h_border or in_v_border:
        return LayerZone.BORDER

    # Interior
    return LayerZone.INTERIOR


def _scale_layer_pixels(
    image: Image.Image,
    src_left: int,
    src_top: int,
    zone: LayerZone,
    ctx: _ScaleContext,
) -> tuple[Image.Image, int, int]:
    """Scale a layer's pixels and compute new position.

    Returns (scaled_image, new_left, new_top).
    """
    if zone == LayerZone.FULL_SPAN:
        # Apply 9-slice scaling: borders stay fixed, interior stretches
        scaled = _pil_nine_slice(image, ctx)
        return scaled, 0, 0

    if zone == LayerZone.CORNER:
        # No resize, just reposition to correct corner
        new_left, new_top = _remap_position(src_left, src_top, image.width, image.height, ctx)
        return image, new_left, new_top

    if zone == LayerZone.BORDER:
        # Scale in the axis parallel to the border, keep perpendicular axis fixed
        cx = src_left + image.width / 2
        cy = src_top + image.height / 2

        if cx < ctx.bl or cx > ctx.src_w - ctx.br:
            # Left or right border: scale height, keep width
            new_h = max(1, int(image.height * ctx.interior_scale_y))
            scaled = _pil_resize(image, image.width, new_h)
        else:
            # Top or bottom border: scale width, keep height
            new_w = max(1, int(image.width * ctx.interior_scale_x))
            scaled = _pil_resize(image, new_w, image.height)

        # Use SOURCE dims for position remapping
        new_left, new_top = _remap_position(src_left, src_top, image.width, image.height, ctx)
        return scaled, new_left, new_top

    # INTERIOR: uniform scale + reposition
    new_w = max(1, int(image.width * ctx.interior_scale_x))
    new_h = max(1, int(image.height * ctx.interior_scale_y))
    scaled = _pil_resize(image, new_w, new_h)
    # Use SOURCE dimensions for position remapping (not scaled dims)
    new_left, new_top = _remap_position(src_left, src_top, image.width, image.height, ctx)
    return scaled, new_left, new_top


def _remap_position(
    src_left: int, src_top: int,
    src_layer_w: int, src_layer_h: int,
    ctx: _ScaleContext,
) -> tuple[int, int]:
    """Map a layer's position from source coordinates to target coordinates.

    Uses the SOURCE layer dimensions to determine which zone the layer
    center falls in. This prevents large scaled layers from having their
    center pushed into the wrong zone.

    The mapping preserves:
    - Absolute position within border zones (borders don't move)
    - Relative position within interior (scales with interior)
    - Right/bottom border positions anchor to the new right/bottom edges
    """
    # Horizontal position — use SOURCE layer width for center calc
    cx = src_left + src_layer_w / 2

    if cx < ctx.bl:
        new_left = src_left
    elif cx > ctx.src_w - ctx.br:
        dist_from_right = ctx.src_w - src_left
        new_left = ctx.target_w - dist_from_right
    else:
        rel_x = src_left - ctx.bl
        new_left = ctx.bl + int(rel_x * ctx.interior_scale_x)

    # Vertical position — use SOURCE layer height for center calc
    cy = src_top + src_layer_h / 2

    if cy < ctx.bt:
        new_top = src_top
    elif cy > ctx.src_h - ctx.bb:
        dist_from_bottom = ctx.src_h - src_top
        new_top = ctx.target_h - dist_from_bottom
    else:
        rel_y = src_top - ctx.bt
        new_top = ctx.bt + int(rel_y * ctx.interior_scale_y)

    return new_left, new_top


def _pil_nine_slice(image: Image.Image, ctx: _ScaleContext) -> Image.Image:
    """Apply 9-slice scaling to a PIL image using the scale context borders.

    Corners stay fixed, border strips scale in one axis, interior scales freely.
    """
    bt, bb, bl, br = ctx.bt, ctx.bb, ctx.bl, ctx.br
    src_w, src_h = image.width, image.height
    tgt_w, tgt_h = ctx.target_w, ctx.target_h

    # Interior dimensions
    src_iw = src_w - bl - br
    src_ih = src_h - bt - bb
    tgt_iw = tgt_w - bl - br
    tgt_ih = tgt_h - bt - bb

    if src_iw <= 0 or src_ih <= 0 or tgt_iw <= 0 or tgt_ih <= 0:
        return _pil_resize(image, tgt_w, tgt_h)

    # If any border is zero, skip cropping that edge
    if bl == 0 and br == 0 and bt == 0 and bb == 0:
        return _pil_resize(image, tgt_w, tgt_h)

    # Crop 9 regions
    corner_tl = image.crop((0, 0, bl, bt))
    corner_tr = image.crop((src_w - br, 0, src_w, bt))
    corner_bl = image.crop((0, src_h - bb, bl, src_h))
    corner_br = image.crop((src_w - br, src_h - bb, src_w, src_h))

    strip_top = image.crop((bl, 0, src_w - br, bt))
    strip_bottom = image.crop((bl, src_h - bb, src_w - br, src_h))
    strip_left = image.crop((0, bt, bl, src_h - bb))
    strip_right = image.crop((src_w - br, bt, src_w, src_h - bb))

    interior = image.crop((bl, bt, src_w - br, src_h - bb))

    # Resize pieces
    strip_top = _pil_resize(strip_top, tgt_iw, bt)
    strip_bottom = _pil_resize(strip_bottom, tgt_iw, bb)
    strip_left = _pil_resize(strip_left, bl, tgt_ih)
    strip_right = _pil_resize(strip_right, br, tgt_ih)
    interior = _pil_resize(interior, tgt_iw, tgt_ih)

    # Assemble
    canvas = Image.new(image.mode, (tgt_w, tgt_h))
    canvas.paste(corner_tl, (0, 0))
    canvas.paste(corner_tr, (tgt_w - br, 0))
    canvas.paste(corner_bl, (0, tgt_h - bb))
    canvas.paste(corner_br, (tgt_w - br, tgt_h - bb))
    canvas.paste(strip_top, (bl, 0))
    canvas.paste(strip_bottom, (bl, tgt_h - bb))
    canvas.paste(strip_left, (0, bt))
    canvas.paste(strip_right, (tgt_w - br, bt))
    canvas.paste(interior, (bl, bt))

    return canvas


def _flatten_alpha(image: Image.Image) -> Image.Image:
    """Keep RGBA as-is — PSD format supports per-layer alpha for transparency."""
    return image


def _pil_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize a PIL image using Lanczos resampling."""
    if image.width == width and image.height == height:
        return image
    return image.resize((width, height), Image.LANCZOS)


def _iter_all_layers(psd_or_group):
    """Recursively iterate all layers in a PSD or group."""
    for layer in psd_or_group:
        yield layer
        if isinstance(layer, Group):
            yield from _iter_all_layers(layer)
