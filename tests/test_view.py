import unittest

from gpuwatch.sampler import sample_once
from gpuwatch.training.status import TrainingStatus
from gpuwatch.view import ViewOptions, apply_view, parse_int_list


class ViewTests(unittest.TestCase):
    def test_parse_int_list_supports_ranges(self):
        self.assertEqual(parse_int_list("0,2-4"), (0, 2, 3, 4))

    def test_apply_view_filters_gpu_pid_and_state(self):
        snapshot = sample_once(backend="fake")
        process_pid = snapshot.gpus[1].processes[0].pid
        statuses = [
            TrainingStatus(pid=process_pid, gpu_index=1, run_name="a", state="running"),
            TrainingStatus(pid=999999, gpu_index=3, run_name="b", state="failed"),
        ]
        filtered_snapshot, filtered_statuses = apply_view(
            snapshot,
            statuses,
            ViewOptions(gpu_indices=(1,), pids=(process_pid,), states=("running",)),
        )
        self.assertEqual([gpu.index for gpu in filtered_snapshot.gpus], [1])
        self.assertEqual([status.run_name for status in filtered_statuses], ["a"])

    def test_apply_view_sorts_gpu_util_stably(self):
        snapshot = sample_once(backend="fake")
        filtered_snapshot, _statuses = apply_view(snapshot, [], ViewOptions(sort="util", reverse=True))
        utils = [gpu.utilization_gpu_percent for gpu in filtered_snapshot.gpus]
        self.assertEqual(utils, sorted(utils, reverse=True))

    def test_apply_view_filters_and_sorts_dataset(self):
        snapshot = sample_once(backend="fake")
        statuses = [
            TrainingStatus(pid=1, gpu_index=0, run_name="a", dataset="zara2", state="running"),
            TrainingStatus(pid=2, gpu_index=1, run_name="b", dataset="nba", state="running"),
        ]
        _snapshot, filtered_statuses = apply_view(snapshot, statuses, ViewOptions(dataset="nba", sort="dataset"))
        self.assertEqual([status.run_name for status in filtered_statuses], ["b"])


if __name__ == "__main__":
    unittest.main()
