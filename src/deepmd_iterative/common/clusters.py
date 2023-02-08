from pathlib import Path
import logging
import sys
import socket

#### deepmd_iterative imports
from deepmd_iterative.common.json import json_read


def get_hostname() -> str:
    """_summary_

    Returns:
        str: _description_
    """
    if socket.gethostname().find(".") >= 0:
        return socket.gethostname()
    else:
        return socket.gethostbyaddr(socket.gethostname())[0]


def clusterize(
        deepmd_iterative_apath: Path,
        training_iterative_apath: Path,
        step: str,
        input_cluster=None,
        user_keyword=None,
        check_only=False,
):
    """_summary_

    Args:
        deepmd_iterative_apath (Path): _description_
        training_iterative_apath (Path): _description_
        step (str): _description_
        input_cluster (str, optional): _description_. Defaults to None.
        user_keyword (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple: cluster_short, cluster_spec, cluster_walltime_format, error_code
        :param check_only:
    """
    clusters_files = []
    if (training_iterative_apath / "user_inputs" / "clusters.json").is_file():
        clusters_files.append(
            json_read(training_iterative_apath / "user_inputs" / "clusters.json")
        )
    if (deepmd_iterative_apath / "data" / "clusters.json").is_file():
        clusters_files.append(
            json_read(deepmd_iterative_apath / "data" / "clusters.json")
        )
    if len(clusters_files) <= 0:
        logging.error(f"No clusters.json found. Wrong installation")
        logging.error(f"Aborting...")
        sys.exit(1)

    cluster = -1
    if input_cluster is None:
        cluster_hostname = get_hostname()
        print(cluster_hostname)
        for clusters_file in clusters_files:
            for zzz in clusters_file.keys():
                if clusters_file[zzz]["hostname"] in cluster_hostname:
                    cluster = zzz
    else:
        cluster = input_cluster

    if cluster == -1:
        logging.error(f"The cluster {get_hostname()} has no spec in clusters.json")
        logging.error(f"Aborting...")
        sys.exit(1)

    if check_only:
        return cluster

    for clusters_file in clusters_files:
        if user_keyword is None:
            for zzz in clusters_file[cluster].keys():
                if (
                        zzz != "hostname"
                        and zzz != "walltime_format"
                        and zzz != "launch_command"
                ):
                    if "default" in clusters_file[cluster][zzz].keys():
                        for yyy in clusters_file[cluster][zzz]["default"]:
                            if step in yyy:
                                return (
                                    cluster,
                                    clusters_file[cluster][zzz],
                                    clusters_file[cluster]["walltime_format"],
                                    clusters_file[cluster]["launch_command"],
                                    0,
                                )
                            else:
                                pass
            if clusters_file == clusters_files[-1]:
                logging.error(f"Couldn't find any default for this step/phase/cluster")
                logging.error(f"Aborting...")
                sys.exit(1)
            else:
                pass
        elif type(user_keyword) == list and len(user_keyword) == 3:
            for zzz in clusters_file[cluster].keys():
                if (
                        zzz != "hostname"
                        and zzz != "walltime_format"
                        and zzz != "launch_command"
                ):
                    if (
                            (clusters_file[cluster][zzz]["project_name"] == user_keyword[0])
                            and (
                            clusters_file[cluster][zzz]["allocation_name"]
                            == user_keyword[1]
                    )
                            and (clusters_file[cluster][zzz]["arch_name"] == user_keyword[2])
                            and (step in clusters_file[cluster][zzz]["valid_for"])
                    ):
                        return (
                            cluster,
                            clusters_file[cluster][zzz],
                            clusters_file[cluster]["walltime_format"],
                            clusters_file[cluster]["launch_command"],
                            0,
                        )
            if clusters_file == clusters_files[-1]:
                return "", [], "", "", 5
            else:
                pass
        elif type(user_keyword) == str:
            if (user_keyword in clusters_file[cluster].keys()) and (
                    step in clusters_file[cluster][user_keyword]["valid_for"]
            ):
                return (
                    cluster,
                    clusters_file[cluster][user_keyword],
                    clusters_file[cluster]["walltime_format"],
                    clusters_file[cluster]["launch_command"],
                    0,
                )
            else:
                if clusters_file == clusters_files[-1]:
                    return "", [], "", "", 4
                else:
                    pass
        else:
            if clusters_file == clusters_files[-1]:
                return "", [], "", "", 3
            else:
                pass


def check_same_cluster(cluster: str, _json: dict):
    """_summary_

    Args:
        cluster (str): _description_
        _json (dict): _description_
    """
    if _json["cluster"] != cluster:
        logging.critical(
            f"Different cluster {cluster} than the one for prep {_json['cluster']}"
        )
        logging.critical("Aborting...")
        sys.exit(1)
