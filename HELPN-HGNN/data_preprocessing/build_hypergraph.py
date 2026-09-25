import os
import argparse
import json

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from skimage.segmentation import slic
from skimage.filters import threshold_otsu
from scipy.stats import zscore
from scipy.signal import find_peaks
import pywt


def load_fmri(path):
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine


def extract_superpixels(mean_vol, n_segments=500, compactness=0.1):
    segments = slic(
        mean_vol,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=None,
        start_label=0
    )
    return segments


def compute_region_timeseries(data, segments):
    T = data.shape[-1]
    labels = np.unique(segments)
    ts_list = []
    for lbl in labels:
        mask = (segments == lbl)
        ts = zscore(data[mask, ...].reshape(-1, T), axis=1)
        ts_mean = np.nanmean(ts, axis=0)
        ts_list.append(ts_mean)
    return np.vstack(ts_list), labels


def select_dynamic_regions(timeseries, pctl=90):
    variances = np.var(timeseries, axis=1)
    thr = np.percentile(variances, pctl)
    return np.where(variances >= thr)[0]


def get_region_coords(segments, labels, affine):
    coords = []
    for lbl in labels:
        pts = np.column_stack(np.where(segments == lbl))
        center_voxel = pts.mean(axis=0)
        xyz = affine.dot(np.append(center_voxel, 1))[:3]
        coords.append(xyz)
    return np.array(coords)


def deconvolve_series(ts):
    """Wavelet denoising to smooth signal and highlight neural activity peaks."""
    coeffs = pywt.wavedec(ts, 'db4', level=3)
    smooth = pywt.waverec(coeffs, 'db4')
    return smooth[:len(ts)]


def detect_hyper_events(ts_mat, window=1):
    """Detect hyper-events: when multiple ROIs have peaks within ±window time points,
    they form a hyper-event at that time step."""
    n_rois, T = ts_mat.shape
    peaks = [find_peaks(ts_mat[i], height=np.std(ts_mat[i]))[0] for i in range(n_rois)]
    events = []
    last_set = None
    for t in range(T):
        coactive = {i for i in range(n_rois)
                    if np.any(np.abs(peaks[i] - t) <= window)}
        if len(coactive) > 1 and coactive != last_set:
            events.append((t, coactive))
            last_set = coactive
    return events


def build_hypergraph(events, ts_mat):
    """Build hypergraph: hyperedges are named by event index, weights are the mean
    signal intensity of co-active ROIs at that time step."""
    H, W = {}, {}
    for t, rois in events:
        eid = f"e{t:03d}"
        H[eid] = rois
        W[eid] = float(np.mean([ts_mat[i, t] for i in rois]))
    return H, W


def main():
    parser = argparse.ArgumentParser(description="fMRI Superpixel Hypergraph Construction")
    parser.add_argument('--input', '-i', required=True,
                        help="4D fMRI NIfTI file (.nii/.nii.gz)")
    parser.add_argument('--output_dir', '-o', default='.',
                        help="Output directory (default: current directory)")
    parser.add_argument('--output_file', '-f', default=None,
                        help="Output filename prefix (without extension, default: same as input)")
    parser.add_argument('--n_segments', type=int, default=500,
                        help="Number of SLIC superpixels")
    parser.add_argument('--pctl', type=float, default=90,
                        help="Variance selection percentile")
    parser.add_argument('--corr_thr', type=float, default=0.7,
                        help="Correlation coefficient threshold")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.output_file:
        out_prefix = os.path.join(output_dir, args.output_file)
    else:
        out_prefix = os.path.join(output_dir, os.path.splitext(os.path.basename(args.input))[0])

    # Load fMRI data
    data, affine = load_fmri(args.input)

    # Compute mean volume and brain mask
    mean_vol = np.mean(data, axis=-1)
    otsu_thr = threshold_otsu(mean_vol)
    brain_mask = mean_vol > otsu_thr

    # Extract superpixel regions and time series
    segments = extract_superpixels(mean_vol, args.n_segments)
    ts, labels = compute_region_timeseries(data, segments)
    idx_dyn = select_dynamic_regions(ts, args.pctl)
    ts_sel = ts[idx_dyn]
    labels_sel = labels[idx_dyn]

    # Compute ROI world coordinates and filter out brain-exterior regions
    coords = get_region_coords(segments, labels_sel, affine)
    inv_aff = np.linalg.inv(affine)
    voxel_coords = apply_affine(inv_aff, coords)
    inside_idx = [
        i for i, vc in enumerate(voxel_coords)
        if (0 <= int(round(vc[0])) < brain_mask.shape[0]
            and 0 <= int(round(vc[1])) < brain_mask.shape[1]
            and 0 <= int(round(vc[2])) < brain_mask.shape[2]
            and brain_mask[tuple(np.round(vc).astype(int))])
    ]
    inside_idx = np.array(inside_idx, dtype=int)
    coords = coords[inside_idx]
    ts_sel = ts_sel[inside_idx]
    labels_sel = labels_sel[inside_idx]
    if ts_sel.shape[0] == 0:
        raise ValueError("No dynamic regions selected. Check your input data or selection threshold.")

    # Detect hyper-events and build hypergraph
    ts_mat = np.vstack([deconvolve_series(ts_sel[i]) for i in range(ts_sel.shape[0])])
    events = detect_hyper_events(ts_mat, window=1)
    H, W = build_hypergraph(events, ts_mat)

    # Save hypergraph and hyper-events to JSON
    hypergraph_data = {
        "hyperedge": {eid: [coords[i].tolist() for i in rois] for eid, rois in H.items()},
        "weights": W,
    }
    hypergraph_path = f"{out_prefix}_hypergraph.json"
    with open(hypergraph_path, "w") as f:
        json.dump(hypergraph_data, f, indent=4)
    print(f"Hypergraph and hyper-events saved to {hypergraph_path}")


if __name__ == '__main__':
    main()