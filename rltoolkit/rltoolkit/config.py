# RL config
ENV_NAME = "CartPole-v0"
ITERATIONS = 2000
GAMMA = 0.95
STEPS_PER_EPOCH = 200
STATS_FREQ = 20
TEST_EPISODES = None
RETURN_DONE = None
LOG_DIR = None
USE_GPU = False
TENSORBOARD_DIR = None
TENSORBOARD_COMMENT = ""
VERBOSE = 1
RENDER = False
DEBUG_MODE = True

# A2C config
A_LR = 3e-3
C_LR = 3e-4
NUM_TARGET_UPDATES = 10
NUM_CRITIC_UPDATES = 10
NORMALIZE_ADV = True

# AcM config
ACM_EPOCHS = 1
ACM_BATCH_SIZE = 128
ACM_UPDATE_FREQ = 1
ACM_OB_IDX = None  # For reacher: [0, 1, 2, 3, 6, 7]
BUFFER_BATCHES = 10
ACM_LR = 0.001
ACM_PRE_TRAIN_SAMPLES = 25000
ACM_PRE_TRAIN_N_EPOCHS = 5
ACM_SCHEDULER_STEP = 25
ACM_SCHEDULER_GAMMA = 0.5
ACM_UPDATE_BATCHES = 200
DENORMALIZE_ACTOR_OUT = True
ACM_KEEP_PRE_TRAIN = True
ACM_CRITIC = True
MIN_MAX_DENORMALIZE = True
NORM_CLOSS = True

# PPO config
PPO_EPSILON = 0.2
GAE_LAMBDA = 0.95
PPO_MAX_KL_DIV = 0.15
PPO_MAX_EPOCHS = 50
PPO_BATCH_SIZE = 1000
PPO_ENTROPY = 0.00

# DDPG
DDPG_LR = 1e-3
TAU = 0.005
UPDATE_BATCH_SIZE = 100
BUFFER_SIZE = int(1e6)
RANDOM_FRAMES = 100
UPDATE_FREQ = 50
GRAD_STEPS = 50
ACT_NOISE = 0.1

# SAC config
ALPHA_LR = 1e-3
ALPHA = 0.2
PI_UPDATE_FREQ = 1

# Norm config
MAX_ABS_OBS_VALUE = 10
NORM_ALPHA = 0.99
OBS_NORM = False

#Lagrangian config


# hparam/{name}: [short_string, default value]
SHORTNAMES = {
    "hparams/type": ["", None],
    "hparams/gamma": ["g", GAMMA],
    "hparams/steps_per_epoch": ["spe", STEPS_PER_EPOCH],
    "hparams/actor_lr": ["a_lr", A_LR],
    "hparams/critic_lr": ["c_lr", C_LR],
    "hparams/critic_num_target_updates": ["c_tar_u", NUM_TARGET_UPDATES],
    "hparams/num_critic_updates_per_target": ["c_up_pt", NUM_CRITIC_UPDATES],
    "hparams/normalize_adv": ["nor", NORMALIZE_ADV],
    "hparams/acm_epochs": ["acm_e", ACM_EPOCHS],
    "hparams/acm_batch_size": ["acm_bs", ACM_BATCH_SIZE],
    "hparams/acm_update_freq": ["acm_ufr", ACM_UPDATE_FREQ],
    "hparams/acm_lr": ["acm_lr", ACM_LR],
    "hparams/buffer_batches": ["bb", BUFFER_BATCHES],
    "hparams/acm_pre_train_epochs": ["pe", ACM_PRE_TRAIN_N_EPOCHS],
    "hparams/acm_pre_train_samples": ["ps", ACM_PRE_TRAIN_SAMPLES],
    "hparams/ppo_epsilon": ["po_eps", PPO_EPSILON],
    "hparams/gae_lambda": ["gae_l", GAE_LAMBDA],
    "hparams/kl_div_threshold": ["kl_thr", PPO_MAX_KL_DIV],
    "hparams/max_ppo_epochs": ["po_e", PPO_MAX_EPOCHS],
    "hparams/ppo_batch_size": ["po_bs", PPO_BATCH_SIZE],
    "hparams/alpha": ["al", ALPHA],
    "hparams/tau": ["tau", TAU],
    "hparams/update_batch_size": ["ubs", UPDATE_BATCH_SIZE],
    "hparams/buffer_size": ["bu_s", BUFFER_SIZE],
    "hparams/update_after": ["ua", False],
    "hparams/random_frames": ["rf", RANDOM_FRAMES],
    "hparams/update_freq": ["ufr", UPDATE_FREQ],
    "hparams/pi_update_freq": ["pi_ufr", PI_UPDATE_FREQ],
    "hparams/grad_steps": ["gs", GRAD_STEPS],
    "hparams/act_noise": ["noi", ACT_NOISE],
    "hparams/acm_update_batches": ["acm_ub", ACM_UPDATE_BATCHES],
    "hparams/unbiased_update": ["acm_unb", False],
    "hparams/custom_loss": ["acm_cl", False],
    "hparams/denormalize_actor_out": ["acm_dno_act", DENORMALIZE_ACTOR_OUT],
    "hparams/acm_keep_pretrain": ["use_pretr", ACM_KEEP_PRE_TRAIN],
    "hparams/acm_critic": ["acm_c", ACM_CRITIC],
    "hparams/min_max_denormalize": ["m_m_den", MIN_MAX_DENORMALIZE],
    "hparams/norm_closs": ["n_cl", NORM_CLOSS],
    "hparams/max_frames": ["mf", 1000000 ],
    "hparams/lagrangian_cl": ["lagr_cl", True],
    "hparams/cl_target": ["cl_t", 0.2],
    "hparams/custom_loss_target_decay": ["cltd", None],
    "hparams/custom_loss_target_dfactor": ["cltdf", None]
}
