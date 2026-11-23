"""Matched-backbone comparison algorithms."""

import torch
import torch.nn.functional as F


def standardized(values):
    return (values - values.mean(dim=-1, keepdim=True)) / values.std(
        dim=-1, correction=0, keepdim=True
    ).clamp_min(1e-6)


def spectral_bonus(features, clusters):
    """Spectral embedding, deterministic farthest-first k-means and sqrt(K/n)."""
    norm = F.normalize(features.double())
    distances = (1 - norm @ norm.T).clamp_min(0)
    upper = distances.triu(1)
    nonzero = upper[upper > 1e-10]
    k = len(features)
    if not nonzero.numel() or clusters == 1:
        return torch.ones(k)
    affinity = torch.exp(-0.5 * (distances / nonzero.median()).square())
    deg = affinity.sum(1).rsqrt()
    _, vectors = torch.linalg.eigh(deg[:, None] * affinity * deg[None, :])
    x = F.normalize(vectors[:, -min(k, clusters) :], dim=1)
    centers = x[:1]
    for _ in range(1, min(k, clusters)):
        nearest = torch.cdist(x, centers).amin(1)
        if nearest.max() < 1e-10:
            break
        centers = torch.cat((centers, x[nearest.argmax()].unsqueeze(0)))
    for _ in range(50):
        labels = torch.cdist(x, centers).argmin(1)
        new = torch.stack(
            [x[labels == j].mean(0) if (labels == j).any() else c for j, c in enumerate(centers)]
        )
        if torch.allclose(new, centers):
            break
        centers = new
    labels = torch.cdist(x, centers).argmin(1)
    return (k / labels.bincount()[labels].float()).sqrt()


def concentrated_subset(features, k):
    """DRIFT appendix F. Select a center and its k-1 closest candidates."""
    distances = torch.cdist(features.double(), features.double()).square()
    distances.fill_diagonal_(float("inf"))
    nearest = torch.argsort(distances, dim=1, stable=True)[:, : k - 1]
    center = distances.gather(1, nearest).sum(1).argmin()
    return torch.cat([center.reshape(1), nearest[center]]).tolist()


def drift_advantages(rewards, features, cfg):
    bonuses = []
    for x in features:
        distances = torch.cdist(x.double(), x.double()).square()
        bonuses.append((distances.sum(1) / (len(x) - 1)).float().clamp(0, cfg.drift_cap))
    return standardized(rewards) + cfg.drift_weight * standardized(torch.stack(bonuses))
