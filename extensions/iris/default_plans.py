import numpy as np
from extensions.kin_feasibility.planner_surface_contact import MotionFrameSequencer, PlannerSurfaceContact


def get_on_knocker_balanced_contact_sequence(starting_frame_pos: dict[str: np.array]):
    """
    Assumes the environment is located 0.35m ahead of robot's initial
    position and has a base step height if 0.4
    """

    door_l_inner_location = np.array([0.34, 0.35, 1.0])
    door_r_inner_location = np.array([0.34, -0.35, 1.0])
    step_length_offset = np.array([0.45, 0., 0.])
    ft_kn_offset = np.array([0.15, 0., 0.28])

    starting_lh_pos = starting_frame_pos['LH']
    starting_rh_pos = starting_frame_pos['RH']
    starting_torso_pos = starting_frame_pos['torso']
    final_lf_pos = starting_frame_pos['LF'] + step_length_offset
    final_lkn_pos = starting_frame_pos['L_knee'] + step_length_offset
    final_rkn_pos = starting_frame_pos['R_knee'] + step_length_offset
    final_rf_pos = starting_frame_pos['RF'] + step_length_offset
    final_torso_pos = starting_frame_pos['torso'] + step_length_offset
    final_rh_pos = starting_frame_pos['RH'] + step_length_offset
    final_lh_pos = starting_frame_pos['LH'] + step_length_offset
    intermediate_rf_pos = np.array([0.35, final_rf_pos[1], 0.44])

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L hand to frame
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                                        'LH': door_l_inner_location,
                                        'RH': door_r_inner_location,
                                        })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    rh_contact_front = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
    motion_frames_seq.add_contact_surfaces([lh_contact_front, rh_contact_front])

    # ---- Step 2: step on knee-knocker with right foot
    fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        # 'RH': intermediate_rh_pos,      # added for ergoCub
                        'RF': intermediate_rf_pos,
                        'R_knee': intermediate_rf_pos + ft_kn_offset})
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_contact_over])

    # ---- Step 3: step through door with left foot
    fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'L_knee': final_lf_pos + ft_kn_offset,
                        'LF': final_lf_pos})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_contact_over])

    # ---- Step 4: balance + return to zero configuration
    fixed_frames.append(['LF', 'L_knee', 'RH'])
    motion_frames_seq.add_motion_frame({
        'torso': final_torso_pos,
        'RF': final_rf_pos,
        'R_knee': final_rf_pos + ft_kn_offset,
        'LH': final_lh_pos
    })
    rf_square_up = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_square_up])

    # ---- Step 7: balance + return to zero configuration
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'])
    motion_frames_seq.add_motion_frame({'RH': final_rh_pos})

    return fixed_frames, motion_frames_seq
