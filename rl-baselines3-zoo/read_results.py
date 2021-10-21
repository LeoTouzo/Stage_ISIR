import numpy as np

res=np.load('logs/sac/HalfCheetahBulletEnv-v0_2/evaluations.npz')
print(res.files)
print(np.shape(res['timesteps']))
print(np.shape(res['results']))
print(np.shape(res['ep_lengths']))

#['timesteps', 'results', 'ep_lengths']
#(100,) (100, 10) (100, 10)

