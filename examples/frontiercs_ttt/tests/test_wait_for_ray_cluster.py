from examples.frontiercs_ttt.wait_for_ray_cluster import (
    ClusterSnapshot,
    cluster_snapshot,
    missing_paths,
)


def test_cluster_snapshot_counts_only_alive_nodes_with_enough_gpus():
    nodes = [
        {"Alive": True, "Resources": {"GPU": 8.0}},
        {"Alive": True, "Resources": {"GPU": 8.0}},
        {"Alive": True, "Resources": {"GPU": 2.0}},
        {"Alive": False, "Resources": {"GPU": 8.0}},
    ]

    assert cluster_snapshot(nodes, {"GPU": 18.0}, 8.0) == ClusterSnapshot(
        alive_nodes=3,
        eligible_nodes=2,
        total_gpus=18.0,
    )


def test_cluster_snapshot_handles_missing_resource_fields():
    nodes = [{"Alive": True}, {"Alive": False, "Resources": {"GPU": 4.0}}]

    assert cluster_snapshot(nodes, {}, 4.0) == ClusterSnapshot(
        alive_nodes=1,
        eligible_nodes=0,
        total_gpus=0.0,
    )


def test_missing_paths_reports_only_absent_entries(tmp_path):
    present = tmp_path / "present"
    present.mkdir()
    absent = tmp_path / "absent"

    assert missing_paths([str(present), str(absent)]) == [str(absent)]
