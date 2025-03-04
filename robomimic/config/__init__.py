from robomimic.config.config import Config
from robomimic.config.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig
