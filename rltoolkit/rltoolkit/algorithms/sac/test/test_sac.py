from rltoolkit.algorithms.sac import SAC
import tempfile
import os


def test_sac_continuous():
    env_name = "Pendulum-v0"
    iterations = 10
    stats_freq = 3
    return_done = -1000
    model = SAC(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
    )
    model.train()


def test_save_and_load():
    env_name = "Pendulum-v0"
    iterations = 3
    stats_freq = 3
    return_done = -1000
    tmpdir = tempfile.TemporaryDirectory()
    log_dir = tmpdir.name
    model = SAC(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
        log_dir=log_dir,
    )
    model.train()

    path_to_model = os.path.join(log_dir, model.filename + ".pkl")
    loaded_model = SAC(
        env_name=env_name,
        iterations=iterations,
        stats_freq=stats_freq,
        return_done=return_done,
        log_dir=log_dir,
    )
    loaded_model.load(path_to_model)

    tmpdir.cleanup()
