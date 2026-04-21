#!/usr/bin/env python3
"""
Saliency-based gaze path extractor for foveated encoding.

Reads a raw YUV420p file, computes per-frame saliency using:
  1. Gaussian center-bias (eye fixation tends toward frame center)
  2. Motion energy (frame difference from previous frame)
  3. Saliency = center_prior * (1 + motion_weight * motion_energy)

Outputs a gaze path file: one line per frame, "frame_num x y"
where (x,y) is the estimated gaze fixation point in pixels.

Usage:
    python3 saliency_gaze.py --input nature.yuv --width 1920 --height 1080 \
                              --fps 25 --output gaze.txt
"""

import math
import struct
import argparse
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def read_yuv_frame(f, width, height):
    """Read one YUV420p frame as numpy luma (Y) plane."""
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    raw = f.read(y_size + 2 * uv_size)
    if len(raw) < y_size:
        return None
    y = np.frombuffer(raw[:y_size], dtype=np.uint8).reshape(height, width).astype(np.float32)
    return y


def make_center_prior(height, width, sigma_frac=0.35):
    """
    Gaussian center-prior map. Peak at frame center with sigma = sigma_frac * frame_size.
    Returns float32 array in [0, 1].
    """
    cy, cx = height / 2.0, width / 2.0
    sigma_y = height * sigma_frac
    sigma_x = width * sigma_frac

    yv = np.arange(height, dtype=np.float32)
    xv = np.arange(width,  dtype=np.float32)
    yy, xx = np.meshgrid(yv, xv, indexing='ij')

    prior = np.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))
    return prior.astype(np.float32)


def compute_motion_energy(frame, prev_frame, blur_sigma=15):
    """
    Compute motion energy as absolute difference, spatially smoothed.
    Returns float32 array in [0, 1].
    """
    if prev_frame is None:
        h, w = frame.shape
        return np.zeros((h, w), dtype=np.float32)

    diff = np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))

    # Gaussian blur to get spatial extent of motion regions
    if HAS_CV2:
        ksize = int(blur_sigma * 6) | 1  # odd kernel size
        blurred = cv2.GaussianBlur(diff, (ksize, ksize), blur_sigma)
    else:
        # Simple box blur fallback
        k = max(3, int(blur_sigma * 2) | 1)
        kernel = np.ones((k, k), dtype=np.float32) / (k * k)
        from scipy.ndimage import convolve
        try:
            from scipy.ndimage import convolve as ndconvolve
            blurred = ndconvolve(diff, kernel, mode='reflect')
        except ImportError:
            blurred = diff  # no blur if scipy not available

    max_val = blurred.max()
    if max_val > 0:
        blurred = blurred / max_val
    return blurred.astype(np.float32)


def compute_saliency(frame, prev_frame, center_prior, motion_weight=2.0):
    """
    Combine center-prior with motion energy to produce per-pixel saliency.
    Returns normalized float32 map in [0, 1].
    """
    motion = compute_motion_energy(frame, prev_frame)
    saliency = center_prior * (1.0 + motion_weight * motion)

    # Normalize
    s_max = saliency.max()
    if s_max > 0:
        saliency = saliency / s_max
    return saliency


def smooth_gaze_path(gaze_points, window=5):
    """Temporal smoothing of gaze trajectory to reduce jitter."""
    if len(gaze_points) <= 1:
        return gaze_points
    xs = np.array([g[0] for g in gaze_points], dtype=np.float32)
    ys = np.array([g[1] for g in gaze_points], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    xs_smooth = np.convolve(xs, kernel, mode='same')
    ys_smooth = np.convolve(ys, kernel, mode='same')
    return [(float(xs_smooth[i]), float(ys_smooth[i])) for i in range(len(gaze_points))]


def extract_gaze_path(yuv_path, width, height, output_path,
                       motion_weight=2.0, smooth_window=5,
                       saliency_map_output=None, verbose=False):
    """
    Main function: read YUV, compute saliency, extract gaze path.

    Args:
        yuv_path: path to raw YUV420p file
        width, height: frame dimensions
        output_path: output gaze.txt path
        motion_weight: relative weight of motion vs center prior
        smooth_window: temporal smoothing window in frames
        saliency_map_output: if set, save saliency PNG for first frame
    """
    center_prior = make_center_prior(height, width)

    gaze_points = []
    frame_num = 0
    prev_frame = None

    with open(yuv_path, 'rb') as f:
        while True:
            frame = read_yuv_frame(f, width, height)
            if frame is None:
                break

            saliency = compute_saliency(frame, prev_frame, center_prior, motion_weight)

            # Gaze = argmax of saliency map
            flat_idx = np.argmax(saliency)
            gy, gx = np.unravel_index(flat_idx, saliency.shape)

            gaze_points.append((float(gx), float(gy)))
            prev_frame = frame

            if verbose and frame_num % 25 == 0:
                print(f"  Frame {frame_num:4d}: gaze=({gx:5.1f}, {gy:5.1f})")

            # Save first-frame saliency map for debugging
            if frame_num == 0 and saliency_map_output:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(12, 7))
                    ax.imshow(saliency, cmap='hot', origin='upper')
                    ax.plot(gx, gy, 'g+', markersize=15, markeredgewidth=2, label='Gaze')
                    ax.set_title('Saliency map (frame 0)')
                    ax.legend()
                    plt.savefig(saliency_map_output, dpi=100, bbox_inches='tight')
                    plt.close()
                    print(f"Saliency map saved: {saliency_map_output}")
                except Exception as e:
                    print(f"Warning: could not save saliency map: {e}")

            frame_num += 1

    # Temporal smoothing
    if smooth_window > 1:
        gaze_points = smooth_gaze_path(gaze_points, smooth_window)

    # Write output
    with open(output_path, 'w') as out:
        for i, (gx, gy) in enumerate(gaze_points):
            out.write(f"{i} {gx:.1f} {gy:.1f}\n")

    print(f"Gaze path written: {output_path} ({frame_num} frames)")
    return gaze_points


def detect_saccades(gaze_points, frame_width, threshold_frac=0.05):
    """
    Detect saccades: frames where gaze jumps > threshold_frac * frame_width.
    Returns list of frame indices where saccade occurs.
    """
    threshold = frame_width * threshold_frac
    saccades = []
    for i in range(1, len(gaze_points)):
        dx = gaze_points[i][0] - gaze_points[i-1][0]
        dy = gaze_points[i][1] - gaze_points[i-1][1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > threshold:
            saccades.append(i)
    return saccades


def main():
    parser = argparse.ArgumentParser(description='Saliency-based gaze path extractor')
    parser.add_argument('--input',   required=True, help='Input YUV420p file')
    parser.add_argument('--width',   type=int, default=1920)
    parser.add_argument('--height',  type=int, default=1080)
    parser.add_argument('--fps',     type=float, default=25.0)
    parser.add_argument('--output',  default='gaze.txt', help='Output gaze path file')
    parser.add_argument('--motion-weight', type=float, default=2.0,
                        help='Weight of motion energy vs center prior (default: 2.0)')
    parser.add_argument('--smooth',  type=int, default=5,
                        help='Temporal smoothing window in frames (default: 5)')
    parser.add_argument('--saliency-map', type=str, default=None,
                        help='Save first-frame saliency visualization PNG')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(f"Extracting gaze path from {args.input} ({args.width}x{args.height} @ {args.fps}fps)")
    gaze = extract_gaze_path(
        args.input, args.width, args.height, args.output,
        motion_weight=args.motion_weight,
        smooth_window=args.smooth,
        saliency_map_output=args.saliency_map,
        verbose=args.verbose
    )

    saccades = detect_saccades(gaze, args.width)
    print(f"Detected {len(saccades)} saccades (threshold: {args.width*0.05:.0f}px)")
    if args.verbose and saccades:
        print(f"  Saccade frames: {saccades[:10]}{'...' if len(saccades) > 10 else ''}")


if __name__ == '__main__':
    main()
