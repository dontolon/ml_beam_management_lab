# beam_helpers/io.py
import json
import numpy as np
import utm
from pathlib import Path
from typing import Tuple
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

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

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim

def build_mlp(
    input_dim=2,
    output_dim=64,
    hidden_dim=128,
    hidden_layers=2,
    activation="ReLU",
    lr=1e-2,
    milestones=[20, 40],
    gamma=0.2,
):
    """
    Build a simple feed-forward neural network (MLP) and 
    initialize optimizer, criterion, and learning rate scheduler.

    Args:
        input_dim (int): number of input features (default=2 for UE positions).
        output_dim (int): number of output classes (default=64 for beams).
        hidden_dim (int): hidden layer width.
        hidden_layers (int): number of hidden layers.
        activation (str): activation function ("ReLU", "Tanh", "Sigmoid", ...).
        lr (float): learning rate for Adam optimizer.
        milestones (list[int]): epochs at which to reduce LR.
        gamma (float): factor to reduce LR at milestones.

    Returns:
        model (nn.Module): the neural network.
        criterion (nn.Module): loss function.
        optimizer (torch.optim.Optimizer): optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler): learning rate scheduler.
    """
    # --- Build layers ---
    layers = []
    in_dim = input_dim
    act_layer = getattr(nn, activation)

    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_layer())
        in_dim = hidden_dim

    layers.append(nn.Linear(hidden_dim, output_dim))

    model = nn.Sequential(*layers)

    # --- Training components ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    return model, criterion, optimizer, scheduler


def evaluate(loader, model, k_values):
    """
    Evaluate top-k accuracies.
    Returns dict {k: acc}.
    """
    model.eval()
    correct_topk = {k: 0 for k in k_values}
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)                       # logits
            _, pred_topk = out.topk(max(k_values), dim=1)  # top-k predictions
            total += yb.size(0)
            for k in k_values:
                correct_topk[k] += (pred_topk[:, :k] == yb.unsqueeze(1)).any(dim=1).sum().item()

    return {k: correct_topk[k]/total for k in k_values}

# Training loop
def run_epoch(loader, model, optimizer, criterion, train=True):
    losses, correct, total = [], 0, 0
    if train: model.train()
    else: model.eval()

    for xb, yb in loader:
        if train: optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)
    return np.mean(losses), correct/total

#######################FEDERATED################################

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

# --- small utility: normalize together for visualization so BS + UEs share same [0,1] scale
def _normalize_for_plot(BS_location, UE_locations):
    both = np.vstack([BS_location, UE_locations])  # (1+N, 2)
    mn = both.min(axis=0, keepdims=True)
    mx = both.max(axis=0, keepdims=True)
    eps = 1e-9
    both_n = (both - mn) / np.maximum(mx - mn, eps)
    return both_n[0:1], both_n[1:]  # BS_norm, UE_norm

# ========== SPLITTING FUNCTIONS ==========

def split_iid_equal_indices(N, num_clients=8, seed=42):
    """
    IID split: shuffle indices and split into equal (or near-equal) shards.
    Returns: list of np.ndarray indices, one per client.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    shards = np.array_split(idx, num_clients)
    return [np.asarray(s, dtype=int) for s in shards]

def split_noniid_label_dirichlet(
    y, num_clients=10, alpha_labels=0.5, alpha_sizes=1.0, min_per_client=10, seed=42
):
    """
    Non-IID split using Dirichlet BOTH for:
    - label distributions (alpha_labels)
    - client sizes (alpha_sizes)

    Args:
      y: label vector (N,)
      num_clients: number of clients
      alpha_labels: Dirichlet concentration for label proportions (class skew)
      alpha_sizes: Dirichlet concentration for client sizes (size skew)
      min_per_client: ensure each client has at least this many samples
      seed: RNG seed

    Returns:
      list of np.ndarray indices (one per client)
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    classes = np.unique(y)

    # Step 1: decide total size per client (size skew)
    size_props = rng.dirichlet([alpha_sizes] * num_clients)
    size_counts = np.floor(size_props * N).astype(int)

    # ensure everyone has min_per_client
    for i in range(num_clients):
        if size_counts[i] < min_per_client:
            size_counts[i] = min_per_client
    while size_counts.sum() > N:
        j = rng.integers(0, num_clients)
        if size_counts[j] > min_per_client:
            size_counts[j] -= 1
    while size_counts.sum() < N:
        j = rng.integers(0, num_clients)
        size_counts[j] += 1

    # Step 2: allocate per-class samples using Dirichlet
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idxs = idx_by_class[c]
        n_c = len(idxs)

        # how much of this class goes to each client
        class_props = rng.dirichlet([alpha_labels] * num_clients)
        class_counts = (class_props * n_c).astype(int)

        # adjust to exact n_c
        while class_counts.sum() < n_c:
            class_counts[rng.integers(0, num_clients)] += 1
        while class_counts.sum() > n_c:
            j = rng.integers(0, num_clients)
            if class_counts[j] > 0:
                class_counts[j] -= 1

        # assign slices to each client
        start = 0
        for j in range(num_clients):
            take = class_counts[j]
            if take > 0:
                client_indices[j].extend(idxs[start:start+take])
                start += take

    # Step 3: enforce overall client sizes (truncate/extend if needed)
    final_clients = []
    for j in range(num_clients):
        idxs = rng.permutation(client_indices[j])
        if len(idxs) > size_counts[j]:
            idxs = idxs[:size_counts[j]]
        elif len(idxs) < size_counts[j]:
            # randomly duplicate to meet quota
            extra = rng.choice(idxs, size_counts[j] - len(idxs), replace=True)
            idxs = np.concatenate([idxs, extra])
        final_clients.append(np.array(idxs, dtype=int))

    return final_clients


# ========== PLOTTING ==========

def plot_client_split(UE_locations, client_splits, BS_location=None, normalize=True, title=None):
    """
    Visualize client partitions over the UE positions.
    - UE_locations: (N, 2) in lat/lon (or any 2D) for plotting
    - client_splits: list of np.ndarray indices (one per client)
    - BS_location: (1,2) optional; if provided and normalize=True it’s normalized with UEs
    - normalize: if True, min-max normalize BS+UEs together for a clean [0,1] plot
    """
    if normalize and BS_location is not None:
        BS_norm, UE_norm = _normalize_for_plot(BS_location, UE_locations)
        Xplot, Bplot = UE_norm, BS_norm[0]
    else:
        Xplot = UE_locations
        Bplot = BS_location[0] if BS_location is not None else None

    plt.figure(figsize=(10,6))
    cmap = plt.cm.get_cmap('tab20', len(client_splits))

    # plot each client with a distinct color
    for i, idxs in enumerate(client_splits):
        plt.scatter(Xplot[idxs,0], Xplot[idxs,1], s=8, color=cmap(i), label=f"Client {i} ({len(idxs)})", alpha=0.8)

    # BS marker
    if BS_location is not None:
        plt.scatter(Bplot[0], Bplot[1], c="red", marker="*", s=100, label="BS")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title if title else "Client split visualization")
    if len(client_splits) <= 12:
        plt.legend(markerscale=2, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.show()

# ========== BUILD CLIENT LOADERS FOR FL LOOP ==========

def make_client_loaders(X, y, client_splits, batch_size=32, shuffle=True):
    """
    Turn a client split (list of index arrays) into DataLoaders.
    Returns: list of (loader, num_samples).
    """
    clients = []
    for idxs in client_splits:
        Xt = torch.tensor(X[idxs], dtype=torch.float32)
        yt = torch.tensor(y[idxs], dtype=torch.long)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)
        clients.append((loader, len(idxs)))
    return clients

# ===== FedAvg helpers that work with `build_mlp` =====
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_state_dict(model, state):
    model.load_state_dict(state, strict=True)

def local_train_on_client(
    model_ctor,          # must return (model, criterion, optimizer, scheduler)
    global_state,        # server's current weights
    client_loader,       # DataLoader for this client
    local_epochs=1,
    device=DEVICE,
):
    """
    Train a local model starting from `global_state` on one client's data.
    Uses the (criterion, optimizer, scheduler) returned by your model_ctor (build_mlp wrapper).
    """
    model, criterion, optimizer, scheduler = model_ctor()
    model = model.to(device)
    set_state_dict(model, global_state)   # start from server weights

    model.train()
    for _ in range(local_epochs):
        for xb, yb in client_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return get_state_dict(model)

def fedavg(states, sizes):
    total = float(np.sum(sizes))
    avg = {k: torch.zeros_like(v) for k, v in states[0].items()}
    for st, n in zip(states, sizes):
        w = n / total
        for k in avg:
            avg[k] += st[k] * w
    return avg

def sample_clients_indices(num_clients, fraction, seed=None):
    rng = np.random.default_rng(seed)
    m = max(1, int(round(fraction * num_clients)))
    return rng.choice(num_clients, size=m, replace=False).tolist()

def federated_round(
    model,               # global model (updated in-place)
    model_ctor,          # returns (model, criterion, optimizer, scheduler)
    clients,             # list of (loader, num_samples)
    client_fraction=1,
    local_epochs=1,
    seed=None,
):
    """
    One FedAvg round:
      - sample clients
      - each client trains locally from current global state
      - server aggregates via weighted average
    """
    C = len(clients)
    selected = sample_clients_indices(C, client_fraction, seed=seed)

    global_state = get_state_dict(model)
    local_states, sizes = [], []

    for cid in selected:
        loader, n = clients[cid]
        new_state = local_train_on_client(
            model_ctor=model_ctor,
            global_state=global_state,
            client_loader=loader,
            local_epochs=local_epochs,
            device=DEVICE,
        )
        local_states.append(new_state)
        sizes.append(n)

    new_global = fedavg(local_states, sizes)
    set_state_dict(model, new_global)
    return selected
