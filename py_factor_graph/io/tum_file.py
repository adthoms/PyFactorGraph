import os
import numpy as np
from typing import List

from py_factor_graph.utils.matrix_utils import (
    get_quat_from_rotation_matrix,
    get_rotation_matrix_from_transformation_matrix,
    get_translation_from_transformation_matrix,
)
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.name_utils import get_time_idx_from_frame_name
from py_factor_graph.utils.logging_utils import logger, F_PREC


def save_robot_trajectories_to_tum_file(
    fg: FactorGraphData, dir: str, use_ground_truth: bool = True
) -> List[str]:
    """Save robot trajectories to TUM format. Each file corresponds to a robot's trajectory.

    Args:
      fg (FactorGraphData): The factor graph data.
      dir (str): The base folder to write the files to.
      use_ground_truth (bool, optional): Whether to use ground truth odometry. Defaults to True.

    Returns:
      List[str]: The list of file paths written.
    """
    # set file prefix and trajectory dictionary
    file_prefix = "odom_gt_robot_" if use_ground_truth else "odom_meas_robot_"
    trajectory_dict = (
        fg.robot_true_trajectories_dict
        if use_ground_truth
        else fg.robot_odometry_trajectories_dict
    )

    # initialize list of odometry files
    odom_files = []

    # write to separate files for each robot
    for robot_char in fg.all_robot_chars:
        # create file
        file_name = f"{file_prefix}{robot_char}.txt"
        file_path = os.path.join(dir, file_name)

        # open file for writing
        with open(file_path, "w") as f:
            # write header
            f.write("# time_stamp x y z qx qy qz qw\n")

            # iterate over trajectory for each robot
            for robot_and_pose_id, T in trajectory_dict[robot_char].items():
                # get timestamped transform in TUM format
                ts = (
                    fg.pose_variables_dict[robot_and_pose_id].timestamp
                    if fg.pose_variables_dict[robot_and_pose_id].timestamp is not None
                    else get_time_idx_from_frame_name(robot_and_pose_id)
                )
                qx, qy, qz, qw = get_quat_from_rotation_matrix(
                    get_rotation_matrix_from_transformation_matrix(T)
                )
                t = get_translation_from_transformation_matrix(T)
                if len(t) == 2:
                    t = np.append(t, 0)

                # write to file
                f.write(
                    f"{ts:.{F_PREC}f} "
                    f"{t[0]:.{F_PREC}f} {t[1]:.{F_PREC}f} {t[2]:.{F_PREC}f} "
                    f"{qx:.{F_PREC}f} {qy:.{F_PREC}f} {qz:.{F_PREC}f} {qw:.{F_PREC}f}\n"
                )

        # close file
        f.close()

        # log and append to list
        logger.info(f"Saved factor graph trajectory in TUM file format to: {file_path}")
        odom_files.append(file_path)

    return odom_files
