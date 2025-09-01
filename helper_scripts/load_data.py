# beam_helpers/io.py
import json
import numpy as np
import utm
from pathlib import Path
from typing import Tuple

def load_config(config_path: str | Path):
    """Load JSON configuration."""
    config_path = Path(config_path)
    print("---------------------------------------------------------")
    print(f"Loading config from: {config_path}")
    try:
        with config_path.open('r') as f:
            cfg = json.load(f)
        print("Config loaded.")
        return cfg
    except FileNotFoundError:
        raise FileNotFoundError(f"Config not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON at {config_path}: {e}")

def min_max(arr: np.ndarray, axis=0, eps=1e-9) -> np.ndarray:
    """Min-max normalize along axis with numerical safety."""
    arr = np.asarray(arr)
    mn = np.min(arr, axis=axis, keepdims=True)
    mx = np.max(arr, axis=axis, keepdims=True)
    denom = np.maximum(mx - mn, eps)
    return (arr - mn) / denom

def xy_from_latlong(lat_long: np.ndarray) -> np.ndarray:
    """
    Convert [lat, lon] pairs (in degrees) to UTM planar [x, y] (meters).
    lat_long: shape (N, 2) columns = [lat, lon]
    """
    lat = lat_long[:, 0]
    lon = lat_long[:, 1]
    x, y, *_ = utm.from_latlon(lat, lon)  # vectorized
    return np.stack((x, y), axis=1)

def add_pos_noise(pos: np.ndarray, noise_variance_in_m: float = 1.0) -> np.ndarray:
    """
    Add isotropic Gaussian noise (meters) in UTM, then convert back to [lat, lon].
    pos: (N, 2) [lat, lon]
    """
    n_samples = pos.shape[0]
    dist = np.random.normal(0, noise_variance_in_m, n_samples)
    ang = np.random.uniform(0, 2*np.pi, n_samples)
    xy_noise = np.stack((dist * np.cos(ang), dist * np.sin(ang)), axis=1)

    x, y, zn, zl = utm.from_latlon(pos[:, 0], pos[:, 1])
    xy_pos = np.stack((x, y), axis=1)
    xy_pos_noise = xy_pos + xy_noise

    lat, lon = utm.to_latlon(xy_pos_noise[:, 0], xy_pos_noise[:, 1], zn, zl)
    return np.stack((lat, lon), axis=1)

def normalize_pos(pos1: np.ndarray,
                  pos2: np.ndarray,
                  norm_type: int) -> np.ndarray:
    """
    Normalizations from Morais et al. (adapted).
    pos1: (N, 2) or (1, 2) BS/Unit1 positions [lat, lon]
    pos2: (N, 2) UE/Unit2 positions [lat, lon]
    Returns normalized 2D features.
    """
    # If pos1 provided per-sample, use its mean as BS reference.
    if pos1.ndim == 2 and pos1.shape[0] > 1:
        pos1_ref = pos1.mean(axis=0, keepdims=True)
    else:
        pos1_ref = pos1.reshape(1, 2)

    if norm_type == 1:
        return min_max(pos2)

    if norm_type == 2:
        pos_norm = min_max(pos2)
        avg_pos2 = np.mean(pos2, axis=0)
        # Flip axes depending on BS vs avg UE
        if pos1_ref[0, 0] > avg_pos2[0]:
            pos_norm[:, 0] = 1 - pos_norm[:, 0]
        if pos1_ref[0, 1] > avg_pos2[1]:
            pos_norm[:, 1] = 1 - pos_norm[:, 1]
        return pos_norm

    if norm_type == 3:
        return min_max(xy_from_latlong(pos2))

    if norm_type == 4:
        # Rotate **cartesian** coords so BS→UE mean aligns with +x, then min-max
        pos2_cart = xy_from_latlong(pos2)
        pos_bs_cart = xy_from_latlong(pos1_ref)
        avg_pos2 = np.mean(pos2_cart, axis=0)
        vect_bs_to_ue = avg_pos2 - pos_bs_cart[0]
        theta = np.arctan2(vect_bs_to_ue[1], vect_bs_to_ue[0])
        rot = np.array([[ np.cos(theta),  np.sin(theta)],
                        [-np.sin(theta),  np.cos(theta)]])
        pos_rot = (rot @ pos2_cart.T).T
        return min_max(pos_rot)

    if norm_type == 5:
        # Polar features relative to BS; distance normalized, angle centered to pi/2 and mapped to [0,1]
        pos2_cart = xy_from_latlong(pos2)
        pos_bs_cart = xy_from_latlong(pos1_ref)
        diff = pos2_cart - pos_bs_cart  # broadcast (N,2) - (1,2)
        dist = np.linalg.norm(diff, axis=1)
        ang = np.arctan2(diff[:, 1], diff[:, 0])

        dist_norm = dist / max(np.max(dist), 1e-9)

        avg_diff = diff.mean(axis=0)
        avg_ang = np.arctan2(avg_diff[1], avg_diff[0])

        # unwrap angle to [0, 2pi)
        ang_un = np.where(ang >= 0, ang, ang + 2*np.pi)
        avg_ang_un = avg_ang if avg_ang >= 0 else avg_ang + 2*np.pi

        offset = np.pi/2 - avg_ang_un
        ang_final = ang_un + offset
        # Map any angle to [0, 2pi), then compress 0..pi to 0..1 (as in original idea)
        ang_final = np.mod(ang_final, 2*np.pi)
        ang_norm = (np.clip(ang_final, 0, np.pi)) / np.pi

        return np.stack((dist_norm, ang_norm), axis=1)

    raise ValueError(f"Unknown norm_type: {norm_type}")


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_scenario2(noise_m=1.0, n_beams=64, norm_type=4, max_samples=2974):
    """
    Load scenario 2 DeepSense files, add GPS noise, normalize positions, and select beams.
    """
    # Hard-coded files
    pos1_file = DATA_DIR / "scenario2_unit1_loc_1-2974.npy"
    pos2_file = DATA_DIR / "scenario2_unit2_loc_1-2974.npy"
    pwr1_file = DATA_DIR / "scenario2_unit1_pwr_60ghz_1-2974.npy"
    seq_file  = DATA_DIR / "scenario2_seq_index_1-2974.npy"

    pos1 = np.load(pos1_file)[:, :2]
    pos2 = np.load(pos2_file)[:, :2]
    pwr1 = np.load(pwr1_file)
    seq  = np.load(seq_file)

    # Limit samples
    N = min(max_samples, len(pos2))
    pos1, pos2, pwr1, seq = pos1[:N], pos2[:N], pwr1[:N], seq[:N]

    # Add noise
    pos2_noisy = add_pos_noise(pos2, noise_variance_in_m=noise_m)

    # Normalize positions
    gps_positions = normalize_pos(pos1, pos2_noisy, norm_type)

    # Beam selection
    max_beams = pwr1.shape[-1]
    divider = max_beams // n_beams
    beam_idxs = np.arange(0, divider * n_beams, divider)
    pwrs = pwr1[:, beam_idxs]

    best_beam = np.argmax(pwrs, axis=1)

    print(f"[Scenario2] N={N} | beams={n_beams}/{max_beams} | norm={norm_type} | noise={noise_m}m")
    print("gps_positions:", gps_positions.shape, "powers:", pwrs.shape, "labels:", best_beam.shape)

    return best_beam, pwrs, gps_positions, seq

