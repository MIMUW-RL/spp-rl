import logging
import pickle as pkl

from rltoolkit.buffer import Memory

logger = logging.getLogger(__name__)


class StatsLogger:
    def __init__(self, alpha: float = 0.9):
        self.running_return = None
        self.test_return = None
        self._alpha = 0.9
        self.frames = 0
        self.rollouts = 0
        self.time_list = []
        self.stats = []
        self.success_rate = None

    def calc_running_return(self, buffer: Memory) -> float:
        new_mean_return = buffer.average_returns_per_rollout
        if self.running_return is None:
            self.running_return = new_mean_return
        else:
            self.running_return *= self._alpha
            self.running_return += (1 - self._alpha) * new_mean_return
        return self.running_return


    def log_stats(self, iteration: int) -> None:
        logger.info(
            f"Iteration {iteration:4}\t Running return: {self.running_return:20.10}"
        )
        if self.test_return is not None:
            logger.info(
                f"Iteration {iteration:4}\t Test return: {self.test_return:23.10}"
            )
        if self.success_rate is not None:
            logger.info(
                f"Iteration {iteration:4}\t success rate: {self.success_rate}"
            )
        average_time = sum(self.time_list) / len(self.time_list)
        logger.info(f"Average iteration is {average_time:8} seconds")

    def task_done(self, i: int) -> None:
        if str(i)[-1] == "1":
            iteration = str(i) + "st"
        elif str(i)[-1] == "2":
            iteration = str(i) + "nd"
        elif str(i)[-1] == "3":
            iteration = str(i) + "rd"
        else:
            iteration = str(i) + "th"

        logger.info(
            f"Task finished at {iteration} iteration. "
            f"Running return is {self.running_return}"
            f"Test return is {self.test_return}"
        )

    def reset_time_list(self):
        self.time_list = []

    def dump_stats(self, file_name):
        with open(str(file_name) + "_logs.pkl", "wb") as f:
            pkl.dump(self.stats, f)
