import os
import tempfile

from rltoolkit.algorithms.ddpg import DDPG


def test_ddpg_continuous():
    env_name = "Pendulum-v0"
    iterations = 10
    stats_freq = 3
    return_done = -1000
    test_episodes = 1
    model = DDPG(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
        test_episodes=test_episodes,
    )
    model.train()


def test_save_and_load():
    env_name = "Pendulum-v0"
    iterations = 3
    stats_freq = 3
    return_done = -1000
    tmpdir = tempfile.TemporaryDirectory()
    log_dir = tmpdir.name
    model = DDPG(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
        log_dir=log_dir,
    )
    model.train()

    path_to_model = os.path.join(log_dir, model.filename + ".pkl")
    loaded_model = DDPG(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
        log_dir=log_dir,
    )
    loaded_model.load(path_to_model)

    tmpdir.cleanup()
