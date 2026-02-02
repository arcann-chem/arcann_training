"""
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2026 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
Created: 2023/09/04
Last modified: 2026/02/02

Unit tests for the slurm module.

Classes
-------
TestReplaceInSlurmFileGeneral():
    Test case for the 'replace_in_slurm_file_general' function.
"""

# Standard library modules
import unittest

# Local imports
from arcann_training.common.slurm import replace_in_slurm_file_general


class TestReplaceInSlurmFileGeneral(unittest.TestCase):
    """
    Test case for the 'replace_in_slurm_file_general' function.

    Methods
    -------
    test_replace_in_slurm_file_general_qosA():
        Test checking replacing values for QoS function of walltime (small).
    test_replace_in_slurm_file_general_qosB():
        Test checking replacing values for QoS function of walltime (long).
    test_replace_in_slurm_file_general_toolong():
        Test checking replacing values for max QoS when walltime is too long.
    test_replace_in_slurm_file_general_no_email():
        Test checking replacing values when no email is provided.
    test_replace_in_slurm_file_general_no_partition_subpartition():
        Test checking replacing values when partition and subpartition are not provided.
    """

    def setUp(self):
        self.slurm_file_master = [
            "Sample",
            "text",
            "_R_PROJECT_",
            "_R_ALLOC_",
            "_R_PARTITION_",
            "_R_SUBPARTITION_",
            "_R_QOS_",
            "_R_WALLTIME_",
            "_R_EMAIL_",
        ]
        self.machine_spec = {
            "project_name": "project123",
            "allocation_name": "allocation456",
            "partition": "partitionA",
            "subpartition": "subpartitionX",
            "qos": {"qosA": 3600, "qosB": 7200},
        }
        self.machine_walltime_format = "hours"
        self.slurm_email = "test@example.com"

    def test_replace_in_slurm_file_general_qosA(self):
        """
        Test checking replacing values for QoS function of walltime (small).
        """
        self.walltime_approx_s = 1800
        expected_result = [
            "Sample",
            "text",
            "project123",
            "allocation456",
            "partitionA",
            "subpartitionX",
            "qosA",
            "0:30:00",
            "test@example.com",
        ]
        result = replace_in_slurm_file_general(
            self.slurm_file_master,
            self.machine_spec,
            self.walltime_approx_s,
            self.machine_walltime_format,
            self.slurm_email,
        )
        self.assertEqual(result, expected_result)

    def test_replace_in_slurm_file_general_qosB(self):
        """
        Test checking replacing values for QoS function of walltime (long).
        """
        self.walltime_approx_s = 5400
        expected_result = [
            "Sample",
            "text",
            "project123",
            "allocation456",
            "partitionA",
            "subpartitionX",
            "qosB",
            "1:30:00",
            "test@example.com",
        ]
        result = replace_in_slurm_file_general(
            self.slurm_file_master,
            self.machine_spec,
            self.walltime_approx_s,
            self.machine_walltime_format,
            self.slurm_email,
        )
        self.assertEqual(result, expected_result)

    def test_replace_in_slurm_file_general_toolong(self):
        """
        Test checking replacing values for max QoS when walltime is too long.
        """
        self.walltime_approx_s = 8000
        expected_result = [
            "Sample",
            "text",
            "project123",
            "allocation456",
            "partitionA",
            "subpartitionX",
            "qosB",
            "2:00:00",
            "test@example.com",
        ]
        result = replace_in_slurm_file_general(
            self.slurm_file_master,
            self.machine_spec,
            self.walltime_approx_s,
            self.machine_walltime_format,
            self.slurm_email,
        )
        self.assertEqual(result, expected_result)

    def test_replace_in_slurm_file_general_no_email(self):
        """
        Test checking replacing values when no email is provided.
        """
        self.walltime_approx_s = 3600
        self.slurm_email = ""
        expected_result = [
            "Sample",
            "text",
            "project123",
            "allocation456",
            "partitionA",
            "subpartitionX",
            "qosA",
            "1:00:00",
        ]
        result = replace_in_slurm_file_general(
            self.slurm_file_master,
            self.machine_spec,
            self.walltime_approx_s,
            self.machine_walltime_format,
            self.slurm_email,
        )
        self.assertEqual(result, expected_result)

    def test_replace_in_slurm_file_general_no_partition_subpartition(self):
        """
        Test checking replacing values when partition and subpartition are not provided.
        """
        self.walltime_approx_s = 7200
        self.machine_spec["partition"] = None
        self.machine_spec["subpartition"] = None
        expected_result = [
            "Sample",
            "text",
            "project123",
            "allocation456",
            "qosB",
            "2:00:00",
            "test@example.com",
        ]
        result = replace_in_slurm_file_general(
            self.slurm_file_master,
            self.machine_spec,
            self.walltime_approx_s,
            self.machine_walltime_format,
            self.slurm_email,
        )
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
