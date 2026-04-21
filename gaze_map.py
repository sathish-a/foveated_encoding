#!/usr/bin/env python3
"""
Standalone Gaussian QP map generator for foveated video encoding.

Computes a 2D array of per-block QP offsets based on a Gaussian falloff
from a gaze fixation point. High quality (offset=0) at the foveal center,
progressively lower quality (offset=delta) toward the periphery.

Formula: q(x,y) = delta * (1 - exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2)))

The output array has one value per 16x16 quantization group (QG), matching
the x265 quantOffsets array layout expected by x265_picture.quantOffsets.
"""

import math
import struct
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Physical setup for sigma calculation (2.5 degrees visual angle)
# Viewing distance: 60 cm, Monitor: 24-inch 16:9 (53.1 x 29.9 cm)
VIEWING_DISTANCE_CM = 60.0
MONITOR_WIDTH_CM    = 53.1
MONITOR_HEIGHT_CM   = 29.9
MONITOR_WIDTH_PX    = 1920
MONITOR_HEIGHT_PX   = 1080

def pixels_per_degree(width_px=MONITOR_WIDTH_PX, monitor_width_cm=MONITOR_WIDTH_CM,
                       viewing_distance_cm=VIEWING_DISTANCE_CM):
    """Compute pixels-per-degree of visual angle."""
    px_per_cm = width_px / monitor_width_cm
    cm_per_degree = 2.0 * viewing_distance_cm * math.tan(math.radians(0.5))
    return px_per_cm * cm_per_degree

def sigma_for_degrees(degrees, width_px=MONITOR_WIDTH_PX,
                       monitor_width_cm=MONITOR_WIDTH_CM,
                       viewing_distance_cm=VIEWING_DISTANCE_CM):
    """Convert visual angle in degrees to pixel sigma."""
    ppd = pixels_per_degree(width_px, monitor_width_cm, viewing_distance_cm)
    return degrees * ppd

def compute_qp_map(width, height, qg_size=16,
                   gaze_x=None, gaze_y=None,
                   sigma=None, delta=20.0,
                   fovea_degrees=2.5):
    """
    Compute per-QG QP offset map.

    Args:
        width, height: frame dimensions in pixels
        qg_size: quantization group size in pixels (16 for default x265)
        gaze_x, gaze_y: gaze fixation in pixels (None = frame center)
        sigma: Gaussian sigma in pixels (None = compute from fovea_degrees)
        delta: max QP offset at periphery (positive = lower quality)
        fovea_degrees: foveal radius in visual angle degrees (used if sigma is None)

    Returns:
        np.ndarray of shape (blocks_y, blocks_x), dtype float32
        Flat C-contiguous array suitable for x265_picture.quantOffsets
    """
    if gaze_x is None:
        gaze_x = width / 2.0
    if gaze_y is None:
        gaze_y = height / 2.0
    if sigma is None:
        sigma = sigma_for_degrees(fovea_degrees, width)

    blocks_x = math.ceil(width / qg_size)
    blocks_y = math.ceil(height / qg_size)

    # Block center coordinates in pixel space
    cx = (np.arange(blocks_x) + 0.5) * qg_size
    cy = (np.arange(blocks_y) + 0.5) * qg_size
    bx, by = np.meshgrid(cx, cy)  # (blocks_y, blocks_x)

    dist_sq = (bx - gaze_x) ** 2 + (by - gaze_y) ** 2
    qp_map = delta * (1.0 - np.exp(-dist_sq / (2.0 * sigma ** 2)))

    return qp_map.astype(np.float32)

def visualize_qp_map(qp_map, gaze_x, gaze_y, width, height, qg_size,
                     sigma, delta, output_path, title=None):
    """Render QP map as heatmap PNG."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: standalone QP map heatmap
    ax = axes[0]
    im = ax.imshow(qp_map, cmap='RdBu_r', vmin=0, vmax=delta, origin='upper',
                   extent=[0, width, height, 0])
    plt.colorbar(im, ax=ax, label='QP Offset')
    # Mark gaze point
    ax.plot(gaze_x, gaze_y, 'g+', markersize=15, markeredgewidth=2, label='Gaze')
    # Draw sigma circle
    circle = plt.Circle((gaze_x, gaze_y), sigma, color='lime', fill=False,
                         linewidth=1.5, linestyle='--', label=f'σ={sigma:.0f}px')
    ax.add_patch(circle)
    circle2 = plt.Circle((gaze_x, gaze_y), 2 * sigma, color='yellow', fill=False,
                          linewidth=1.0, linestyle=':', label=f'2σ={2*sigma:.0f}px')
    ax.add_patch(circle2)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(title or f'QP Offset Map (delta={delta}, σ={sigma:.1f}px)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend(loc='upper right', fontsize=8)

    # Right: 1D cross-section through gaze point
    ax2 = axes[1]
    blocks_x = qp_map.shape[1]
    gaze_block_y = int(gaze_y / qg_size)
    gaze_block_y = min(gaze_block_y, qp_map.shape[0] - 1)
    row_x = np.arange(blocks_x) * qg_size + qg_size / 2

    ax2.plot(row_x, qp_map[gaze_block_y, :], 'b-', linewidth=2, label='Horizontal slice')

    blocks_y = qp_map.shape[0]
    gaze_block_x = int(gaze_x / qg_size)
    gaze_block_x = min(gaze_block_x, blocks_x - 1)
    col_y = np.arange(blocks_y) * qg_size + qg_size / 2
    ax2.plot(col_y, qp_map[:, gaze_block_x], 'r--', linewidth=2, label='Vertical slice')

    ax2.axvline(x=gaze_x, color='green', linestyle=':', alpha=0.7, label='Gaze position')
    ax2.axhline(y=delta, color='gray', linestyle=':', alpha=0.5, label=f'Max delta={delta}')
    ax2.set_xlabel('Pixel position')
    ax2.set_ylabel('QP Offset')
    ax2.set_title('Cross-sections through gaze point')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, delta + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {output_path}")

def write_flat_binary(qp_map, path):
    """Write flat row-major float32 binary matching x265 quantOffsets layout."""
    flat = qp_map.flatten().astype(np.float32)
    flat.tofile(path)
    print(f"Binary QP map written: {path} ({len(flat)} floats)")

def run_unit_tests(width=1920, height=1080, delta=20.0):
    """Verify correctness of the Gaussian map."""
    sigma = sigma_for_degrees(2.5, width)
    qp_map = compute_qp_map(width, height, gaze_x=width/2, gaze_y=height/2,
                             sigma=sigma, delta=delta)

    blocks_x = math.ceil(width / 16)
    blocks_y = math.ceil(height / 16)
    corner_val = qp_map[0, 0]

    # Minimum value in map should be near the gaze block (< 5% of delta)
    min_val = qp_map.min()
    # Corner blocks should be near max delta (within 2% tolerance at large sigma)
    # For sigma=95px and corner at ~1000px away: delta*(1-exp(-1000^2/(2*95^2))) ≈ delta*1.0
    print(f"Unit tests for {width}x{height}, delta={delta}, sigma={sigma:.1f}px:")
    print(f"  Min QG (closest to gaze): {min_val:.4f} (expect < 5% of delta = {0.05*delta:.3f})")
    print(f"  Corner QG (0,0): {corner_val:.4f} (expect ~{delta:.1f})")

    assert min_val < 0.05 * delta, \
        f"Min map value {min_val:.4f} should be < 5% of delta {0.05*delta:.4f}"
    assert abs(corner_val - delta) < delta * 0.05, \
        f"Corner value {corner_val} not near delta {delta}"

    # All values in [0, delta]
    assert qp_map.min() >= 0.0, f"Min value {qp_map.min()} below 0"
    assert qp_map.max() <= delta + 1e-4, f"Max value {qp_map.max()} above delta"

    # Shape check
    assert qp_map.shape == (blocks_y, blocks_x), \
        f"Shape mismatch: {qp_map.shape} vs ({blocks_y},{blocks_x})"

    # Test that gaze at a block-aligned position gives 0 at that block
    aligned_gx = 8.0  # center of block (0,0)
    aligned_gy = 8.0
    qp_aligned = compute_qp_map(width, height, gaze_x=aligned_gx, gaze_y=aligned_gy,
                                 sigma=sigma, delta=delta)
    assert qp_aligned[0, 0] < 1e-4, \
        f"Block-aligned center should give 0 at that block, got {qp_aligned[0,0]}"

    print("  All unit tests PASSED")
    return True

def main():
    parser = argparse.ArgumentParser(description='Gaussian foveal QP map generator')
    parser.add_argument('--width',    type=int, default=1920)
    parser.add_argument('--height',   type=int, default=1080)
    parser.add_argument('--qg-size',  type=int, default=16,
                        help='Quantization group size in pixels (match x265 --qg-size)')
    parser.add_argument('--gaze-x',   type=float, default=None,
                        help='Gaze X in pixels (default: frame center)')
    parser.add_argument('--gaze-y',   type=float, default=None,
                        help='Gaze Y in pixels (default: frame center)')
    parser.add_argument('--sigma',    type=float, default=None,
                        help='Gaussian sigma in pixels (default: computed from --fovea-degrees)')
    parser.add_argument('--fovea-degrees', type=float, default=2.5,
                        help='Foveal radius in visual angle degrees (default: 2.5)')
    parser.add_argument('--delta',    type=float, default=20.0,
                        help='Max QP offset at periphery (default: 20)')
    parser.add_argument('--output',   type=str, default=None,
                        help='Output binary file for quantOffsets (float32 raw)')
    parser.add_argument('--heatmap',  type=str, default=None,
                        help='Output PNG heatmap path')
    parser.add_argument('--test',     action='store_true',
                        help='Run unit tests and exit')
    parser.add_argument('--info',     action='store_true',
                        help='Print sigma/ppd info and exit')
    args = parser.parse_args()

    if args.test:
        run_unit_tests()
        return

    ppd = pixels_per_degree(args.width)
    sigma = args.sigma if args.sigma else sigma_for_degrees(args.fovea_degrees, args.width)

    if args.info:
        blocks_x = math.ceil(args.width / args.qg_size)
        blocks_y = math.ceil(args.height / args.qg_size)
        print(f"Frame:          {args.width}x{args.height}")
        print(f"QG size:        {args.qg_size}x{args.qg_size}")
        print(f"QG grid:        {blocks_x}x{blocks_y} = {blocks_x*blocks_y} blocks")
        print(f"Pixels/degree:  {ppd:.2f}")
        print(f"Fovea degrees:  {args.fovea_degrees}°")
        print(f"Sigma (pixels): {sigma:.2f}")
        print(f"Sigma (blocks): {sigma/args.qg_size:.2f}")
        print(f"Delta:          {args.delta}")
        return

    gaze_x = args.gaze_x if args.gaze_x is not None else args.width / 2.0
    gaze_y = args.gaze_y if args.gaze_y is not None else args.height / 2.0

    qp_map = compute_qp_map(args.width, args.height, args.qg_size,
                             gaze_x, gaze_y, sigma, args.delta)

    print(f"QP map: shape={qp_map.shape}, min={qp_map.min():.3f}, "
          f"max={qp_map.max():.3f}, mean={qp_map.mean():.3f}")

    if args.output:
        write_flat_binary(qp_map, args.output)

    if args.heatmap:
        visualize_qp_map(qp_map, gaze_x, gaze_y, args.width, args.height,
                         args.qg_size, sigma, args.delta, args.heatmap)

if __name__ == '__main__':
    main()
