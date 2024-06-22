import os
import numpy as np
from typing import List

from py_factor_graph.utils.matrix_utils import (
    get_quat_from_rotation_matrix,
    get_rotation_matrix_from_theta,
)
from py_factor_graph.io.pyfg_file import (
    POSE_TYPE_2D,
    POSE_TYPE_3D,
    read_from_pyfg_file,
    save_to_pyfg_file,
)
from py_factor_graph.io.tum_file import save_robot_trajectories_to_tum_file
from py_factor_graph.utils.logging_utils import F_PREC

# get current directory and directory containing data
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cur_dir, "data")

# create temporary folder for saving factor graph to file
tmp_dir = os.path.join(cur_dir, "tmp")
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)


def check_file_equality(file_path_1: str, file_path_2: str) -> bool:
    """Checks if two files are equal by comparing their contents line by line.

    Args:
      file_path_1 (str): The first file path.
      file_path_2 (str): The second file path.

    Returns:
      bool: True if the files are equal, False otherwise.
    """
    with open(file_path_1, "r") as f1, open(file_path_2, "r") as f2:
        for line1, line2 in zip(f1, f2):
            if line1.strip() != line2.strip():
                return False
    return True


def check_tum_line_closeness(
    tum_line_list_1: List[str], tum_line_list_2: List[str]
) -> bool:
    """Checks that two lists containing TUM formatted strings are within tolerance as defined by F_PREC.

    Args:
      tum_line_list_1 (List[str]): The first list of lines.
      tum_line_list_2 (List[str]): The second list of lines.

    Returns:
      bool: True if the lines are within tolerance.
    """
    if len(tum_line_list_1) != len(tum_line_list_2):
        return False
    for line1, line2 in zip(tum_line_list_1, tum_line_list_2):
        line1_array = np.fromstring(line1, sep=" ", dtype=float)
        line2_array = np.fromstring(line2, sep=" ", dtype=float)
        if not np.allclose(line1_array, line2_array, atol=F_PREC):
            return False
    return True


def check_read_write_pyfg_file(file_type: str) -> None:
    """Checks the functionality of reading and writing factor graph data to a PyFg file.

    Args:
      file_type (str): The type of the PyFG file (either "se2" or "se3").

    Returns:
      None
    """
    # read factor graph data
    data_file = os.path.join(data_dir, f"pyfg_{file_type}_test_data.pyfg")
    factor_graph = read_from_pyfg_file(data_file)

    # write factor graph data
    write_file = os.path.join(tmp_dir, f"pyfg_{file_type}_test_tmp.pyfg")
    save_to_pyfg_file(factor_graph, write_file)

    # assert read and write files are equal
    assert check_file_equality(data_file, write_file)

    # remove temporary file
    os.remove(write_file)


def check_write_tum_file(file_type: str) -> None:
    """Checks the functionality of writing factor graph data to a TUM file.

    Args:
      file_type (str): The type of the PyFG file (either "se2" or "se3").

    Returns:
      None
    """
    # read factor graph data
    data_file = os.path.join(data_dir, f"pyfg_{file_type}_test_data.pyfg")
    factor_graph = read_from_pyfg_file(data_file)
    robot_chars = factor_graph.all_robot_chars

    # set flags
    use_ground_truth_flags = [True, False]
    use_pose_idx_flags = [True, False]

    # write factor graph data
    for use_ground_truth in use_ground_truth_flags:
        for use_pose_idx in use_pose_idx_flags:
            # retain same file prefix from save_robot_trajectories_to_tum_file
            file_prefix = "odom_gt_robot_" if use_ground_truth else "odom_meas_robot_"

            # save robot trajectories to TUM file
            save_robot_trajectories_to_tum_file(
                factor_graph, tmp_dir, use_ground_truth, use_pose_idx
            )

            # iterate over all TUM files
            for robot_char in robot_chars:
                write_file = os.path.join(tmp_dir, f"{file_prefix}{robot_char}.txt")

                # TUM lines
                saved_tum_lines = []
                with open(write_file, "r") as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        saved_tum_lines.append(line.strip())

                # PyFG lines
                pyfg_equivalent_tum_lines = []
                with open(data_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        columns = line.strip().split()
                        if (
                            columns[0] in [POSE_TYPE_2D, POSE_TYPE_3D]
                            and robot_char in columns[2]
                        ):
                            # get pose index as timestamp of pose ID
                            state_idx_str = columns[1]
                            if use_pose_idx:
                                state_idx_str = "".join(filter(str.isdigit, columns[2]))

                            if file_type == "se2":
                                # get theta from quat to avoid floating point rounding errors
                                qx, qy, qz, qw = get_quat_from_rotation_matrix(
                                    get_rotation_matrix_from_theta(float(columns[5]))
                                )
                                quat_string = f"{qx:.{F_PREC}f} {qy:.{F_PREC}f} {qz:.{F_PREC}f} {qw:.{F_PREC}f}"

                                # add pyfg data in tum format
                                tum_columns = (
                                    [state_idx_str] + columns[3:5] + [f"{0:.{F_PREC}f}"]
                                )
                                pyfg_equivalent_tum_lines.append(
                                    " ".join(tum_columns) + " " + quat_string
                                )
                            elif file_type == "se3":
                                # add pyfg data in tum format
                                tum_columns = [state_idx_str] + columns[3:]
                                pyfg_equivalent_tum_lines.append(" ".join(tum_columns))
                            else:
                                raise ValueError(
                                    f"PyFG test file type {file_type} is invalid"
                                )

                # check written TUM lines against PyFG lines
                if use_ground_truth:
                    # ground truth poses must be identical for fprec
                    assert saved_tum_lines == pyfg_equivalent_tum_lines
                else:
                    # measured odom will accumulate error through pre-multiplication
                    # pyfg test files assume zero-noise odometry measurements
                    assert check_tum_line_closeness(
                        saved_tum_lines, pyfg_equivalent_tum_lines
                    )

                # remove temporary file
                os.remove(write_file)


def test_pyfg_se3_file() -> None:
    check_read_write_pyfg_file("se3")


def test_pyfg_se2_file() -> None:
    check_read_write_pyfg_file("se2")


def test_tum_se3_file() -> None:
    check_write_tum_file("se3")


def test_tum_se2_file() -> None:
    check_write_tum_file("se2")
