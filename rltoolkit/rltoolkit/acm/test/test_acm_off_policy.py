from rltoolkit.acm.off_policy import DDPG_AcM, SAC_AcM

ENV_NAME = "Pendulum-v0"
ACM_PRE_TRAIN_SAMPLES = 100
ACM_PRE_TRAIN_EPOCHS = 1
ITERATIONS = 5
STEPS_PER_EPOCH = 100
STATS_FREQ = 5
ACM_UPDATE_FREQ = 100
ACM_EPOCHS = 1



def test_ddpg_acm():
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
    )
    model.pre_train()
    model.train()


def test_ddpg_acm_min_max():
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        denormalize_actor_out=True,
        min_max_denormalize=True,
    )
    model.pre_train()
    model.train()


def test_sac_acm():
    model = SAC_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
    )
    model.pre_train()
    model.train()


def test_sac_acm_drop_pretrain():
    acm_keep_pretrain = False
    model = SAC_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_keep_pretrain=acm_keep_pretrain,
    )
    model.pre_train()
    model.train()


def test_ddpg_acm_batches():
    acm_update_batches = 50
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_update_batches=acm_update_batches,
    )
    model.pre_train()
    model.train()


def test_ddpg_custom_loss():
    custom_loss = 0.1
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        custom_loss=custom_loss,
    )
    model.pre_train()
    model.train()


def test_ddpg_custom_loss_min_max():
    custom_loss = 0.1
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        custom_loss=custom_loss,
        denormalize_actor_out=True,
        min_max_denormalize=True,
    )
    model.pre_train()
    model.train()


def test_ddpg_acm_critic():
    acm_critic = True
    model = DDPG_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_critic=acm_critic,
    )
    model.pre_train()
    model.train()


def test_sac_acm_critic():
    acm_critic = True
    model = SAC_AcM(
        env_name=ENV_NAME,
        acm_pre_train_samples=ACM_PRE_TRAIN_SAMPLES,
        acm_pre_train_epochs=ACM_PRE_TRAIN_EPOCHS,
        iterations=ITERATIONS,
        steps_per_epoch=STEPS_PER_EPOCH,
        stats_freq=STATS_FREQ,
        acm_update_freq=ACM_UPDATE_FREQ,
        acm_epochs=ACM_EPOCHS,
        acm_critic=acm_critic,
    )
    model.pre_train()
    model.train()
