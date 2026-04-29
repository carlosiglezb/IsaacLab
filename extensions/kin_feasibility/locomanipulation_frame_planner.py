import os
from typing import List

from .multiframe_fpp.multiframe_fpp import plan_multiple_iris

import numpy as np
from ruamel.yaml import YAML


class LocomanipulationFramePlanner:
    def __init__(self,
                 traversable_regions_list,
                 aux_frames_path=None,
                 fixed_frames=None,
                 motion_frames_seq=None,
                 sca_robot_geom=None,
                 b_use_knees_in_smooth_plan=True):

        self.traversable_regions = traversable_regions_list
        self.fixed_frames = fixed_frames
        self.motion_frames_seq = motion_frames_seq
        self.b_use_knees_in_smooth_plan = b_use_knees_in_smooth_plan
        self.sca_robot_geom = sca_robot_geom
        self.env_geometry = None
        self.solver_stats = {}
        self.path = []
        self.points = None
        self.safe_points = None
        self.box_seq = []

        if aux_frames_path is not None:
            self.aux_frames = self._load_aux_frames(aux_frames_path)
        else:
            self.aux_frames = None

    def plan_iris(self,
                  T: float,
                  alpha: List[float],
                  w_rigid: np.ndarray,
                  w_rigid_poly: np.ndarray = None,
                  b_final_vel_constraint: bool = False,
                  verbose: bool = False):
        self.path, self.box_seq, self.points, self.safe_points, self.solver_stats = plan_multiple_iris(
            traversable_regions=self.traversable_regions,
            T=T,
            alpha=alpha,
            verbose=verbose,
            A=self.aux_frames,
            fixed_frames=self.fixed_frames,
            motion_frames_seq=self.motion_frames_seq,
            sca_robot_geometry=self.sca_robot_geom,
            env_geometry=self.env_geometry,
            w_rigid=w_rigid,
            w_rigid_poly=w_rigid_poly,
            b_use_knees_in_smooth_plan=self.b_use_knees_in_smooth_plan,
            b_final_vel_constr=b_final_vel_constraint,
        )

    def set_env_geometry(self, env_geometry):
        self.env_geometry = env_geometry

    def plot(self, visualizer, static_html=False):
        frame_names = list(self.traversable_regions.IRIS_seq.keys())
        for i, p in enumerate(self.path):
            for seg in range(len(p.beziers)):
                bezier_curve = [p.beziers[seg]]
                fr_name = frame_names[i]
                self.visualize_bezier_points(visualizer.viewer, fr_name, bezier_curve, seg)

        for seq, sp_dict in enumerate(self.safe_points):
            for k, v in sp_dict.items():
                v = v.reshape(1, 3)
                self.visualize_simple_points(
                    visualizer.viewer["traversable_regions"]["inputs"],
                    k + "/" + str(seq), v, color=[0.1, 0.1, 0.1, 0.5])

        visualizer.viewer["traversable_regions"].set_property("visible", False)
        if static_html:
            res = visualizer.viewer.static_html()
            path = os.getcwd() + '/data'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_file = './data/multi-contact-plan.html'
            with open(save_file, "w") as f:
                f.write(res)

    @staticmethod
    def _load_aux_frames(path):
        aux_frames = []
        with open(path, 'r') as f:
            yml = YAML().load(f)
            for fr in range(len(yml)):
                aux_frames.append(yml[fr])
        return aux_frames
