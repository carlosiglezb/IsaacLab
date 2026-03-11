import torch
import omni.usd
from pxr import UsdGeom, UsdPhysics
from isaaclab.assets import Articulation



def segment_to_segment_dist(p1, p2, p3, p4):
    """
    Computes the minimum distance between line segment (p1-p2) and (p3-p4).
    """
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3

    a = torch.sum(u * u, dim=-1)
    b = torch.sum(u * v, dim=-1)
    c = torch.sum(v * v, dim=-1)
    d = torch.sum(u * w, dim=-1)
    e = torch.sum(v * w, dim=-1)

    D = a * c - b * b
    sc, tc = D, D

    # compute the line parameters of the two closest points
    mask_D_small = D < 1e-7
    sc = torch.where(mask_D_small, torch.zeros_like(D), (b * e - c * d))
    tc = torch.where(mask_D_small, e, (a * e - b * d))

    # clamp parameters to be within segment bounds [0, 1]
    # use sc/D and tc/D logic
    sc = torch.clamp(sc / torch.clamp(D, min=1e-7), 0.0, 1.0)
    tc = torch.clamp(tc / torch.clamp(D, min=1e-7), 0.0, 1.0)

    # closest points
    closest_p1 = p1 + sc.unsqueeze(-1) * u
    closest_p2 = p3 + tc.unsqueeze(-1) * v

    return torch.norm(closest_p1 - closest_p2, dim=-1)