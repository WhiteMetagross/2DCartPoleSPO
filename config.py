#This is the centralized configuration for the CartPole-v1 environment.
#This configuration is used by the training, evaluation, and optimization programs.

import os

try:
    from SPOinPyTorch import Config
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from SPOinPyTorch import Config

#This function returns the default configuration for the CartPole-v1 environment.
def get_default_config() -> Config:
    config = Config()
    config.update({
        'env_name': 'CartPole-v1',
        'total_timesteps': 200_000,
        'steps_per_batch': 2048,
        'update_epochs': 10,
        'num_minibatches': 64,
        'learning_rate': 0.0007282976901767067,
        'gamma': 0.9919377900986864,
        'gae_lambda': 0.9519200113582851,
        'epsilon': 0.29947656712638776,
        'entropy_coeff': 0.004378183225974978,
        'value_loss_coeff': 0.5067698532530341,
        'max_grad_norm': 1.6841379686217262,
        'actor_hidden_dims': [128, 128, 128, 128],
        'critic_hidden_dims': [128, 128, 128, 128],
        'normalize_advantages': False,
        'eval_interval': 2,
        'save_interval': 20,
        'log_interval': 5,
        'target_reward': 500.0,
        'early_stopping_patience': 20,
    })
    return config

