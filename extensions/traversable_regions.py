import numpy as np
from ruamel.yaml import YAML

from extensions.polytope_math import extract_plane_eqn_from_coeffs


class TraversableRegions:
    def __init__(self, IRIS_lst: list[dict[str: np.array]],
                 IRIS_seq: dict[str: list[int]],
                 safe_points_lst: list[dict[str: np.array]],
                 convex_hull_halfspace_path_dict: dict[str: str] | None = None):
        self.regions = IRIS_lst
        self.IRIS_seq = IRIS_seq
        self.safe_points_lst = safe_points_lst
        self.reach: dict | None = None

        # Retrieve halfspace coefficients defining the convex hull
        if convex_hull_halfspace_path_dict is not None:
            for frame_name, chull_path in convex_hull_halfspace_path_dict.items():
                plane_coeffs = []
                with open(chull_path, 'r') as f:
                    yml = YAML().load(f)
                    for i_plane in range(len(yml)):
                        plane_coeffs.append(yml[i_plane])
                H, d = extract_plane_eqn_from_coeffs(plane_coeffs)
                self.reach[frame_name] = {'H': H}
                self.reach[frame_name] = {'d': -d}
        else:
            print(f"Convex hulls not specified")