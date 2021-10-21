import gym

# Register custom envs
#import utils.import_envs  # noqa: F401 pytype: disable=import-error

registered_envs = set(gym.envs.registry.env_specs.keys())
print('HopperBulletEnv-v0' in registered_envs)
