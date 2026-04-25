import numpy as np
from typing import Optional
from ruamel.yaml import YAML

from extensions.polytope_math import extract_plane_eqn_from_coeffs


def update_plane_offset_from_root(_origin_pos, H, d):
    return d + H @ _origin_pos


class TraversableRegions:
    def __init__(self, IRIS_lst: list[dict[str, np.ndarray]],
                 IRIS_seq: dict[str, list[int]],
                 safe_points_lst: list[dict[str, np.ndarray]],
                 convex_hull_halfspace_path_dict: Optional[dict[str, str]] = None,
                 root_to_torso_pos: Optional[np.ndarray] = None,):
        self.regions = IRIS_lst
        self.IRIS_seq = IRIS_seq
        self.safe_points_lst = safe_points_lst
        if convex_hull_halfspace_path_dict is not None:
            self.reach = {}
        else:
            self.reach = None

        # Retrieve halfspace coefficients defining the convex hull
        if convex_hull_halfspace_path_dict is not None:
            for frame_name, chull_path in convex_hull_halfspace_path_dict.items():
                plane_coeffs = []
                with open(chull_path, 'r') as f:
                    yml = YAML().load(f)
                    for i_plane in range(len(yml)):
                        plane_coeffs.append(yml[i_plane])
                H, d = extract_plane_eqn_from_coeffs(plane_coeffs)
                if frame_name != 'torso':
                    root_to_torso_pos = np.zeros((3, 1)) if root_to_torso_pos is None else root_to_torso_pos
                    d = update_plane_offset_from_root(root_to_torso_pos, H, d)
                self.reach[frame_name] = {'H': H, 'd': d}
        else:
            print(f"Convex hulls not specified")

