import os

import torch
import wandb

if __name__ == '__main__':

    # get data from API
    api = wandb.Api()
    url = 'koulanurag/minimal-marl'
    runs = api.runs(url)

    # extract relevant data
    data = {}
    for runs in [api.runs(url)]:
        for run in runs:
            env_name = run.config['env_name']

            if run.config['algo'] == 'vdn' and env_name not in data:
                data[env_name] = None
                env_root = os.path.join('models/{}'.format(env_name))
                wandb.restore(name='model.p', run_path='/'.join(run.path), replace=True, root=env_root)

                from vdn import QNet, test
                import numpy as np

                # create env.
                import gym

                env = gym.make(env_name)
                q = QNet(env.observation_space, env.action_space, recurrent=True)
                q.load_state_dict(torch.load(os.path.join(env_root, 'model.p'), map_location=torch.device('cpu')))
                test_score, obs_images = test(env, 1, q, render_first=True)

                with wandb.init(project='minimal-marl', job_type='render', config=run.config) as run_1:
                    run_1.log({"test/video": wandb.Video(np.array(obs_images).swapaxes(3, 1).swapaxes(3, 2),
                                                         fps=32, format="gif")})

                break
        if len(data.keys()) > 0:
            break
