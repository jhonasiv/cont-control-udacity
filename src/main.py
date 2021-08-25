import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from unityagents import UnityEnvironment

from agent import DDPGAgent

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def evaluate(env, brain_name, agent):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    rewards = []
    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        state = env_info.vector_observations[0]
        rewards.append(reward)
        
        if done:
            break
    print(f'Achieved reward of {np.array(rewards).sum()}!')


def train(env: UnityEnvironment, brain_name: str, num_eps: int, agent: DDPGAgent):
    eps = 0
    next_goal = 30
    last_goal_reached = 0
    hist_rewards = []
    ep_rewards = []
    while eps < num_eps:
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        ep_rewards.clear()
        agent.reset_noise()
        while True:
            actions = agent.act(state)
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            dones = env_info.local_done[0]
            agent.step(state, actions, rewards, next_state, dones)
            state = next_state
            ep_rewards.append(rewards)
            if dones:
                break
        eps += 1
        hist_rewards.append(sum(ep_rewards))
        moving_avg = np.mean(np.array(hist_rewards)[-100:])
        print(f'\rEpisode {eps}/{num_eps} \tMoving Avg: {moving_avg}',
              end='')
        if moving_avg > next_goal:
            print(f'Achieved {next_goal} after {eps - 100} episodes!')
            last_goal_reached = eps
            agent.save('../ckpt', f'ckpt_{next_goal}')
            next_goal += 5
        
        # Stop threshold
        if eps - last_goal_reached >= 400:
            break
    return np.array(hist_rewards)


def plot_scores(scores) -> None:
    """
    Plot scores from trained agent.
    :param scores: scores obtained after training
    """
    fig = plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('../resources/scores-plot.png')
    plt.show()


def main(gamma, a_lr, c_lr, seed, buffer_size, batch_size, num_eps, alpha, eval, tau, beta_init,
         update_every):
    env = UnityEnvironment(file_name="../world/single/Reacher.x86", no_graphics=False)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    
    num_steps = num_eps * 1002
    
    agent = DDPGAgent(action_size=action_size, state_size=state_size, alpha=alpha,
                      update_every=update_every, optimizers_cls={'actor': Adam, 'critic': Adam},
                      optimizers_kwargs={'actor' : {'lr': a_lr},
                                         'critic': {'lr': c_lr, 'weight_decay': 0}},
                      buffer_size=buffer_size, batch_size=batch_size, device=device, gamma=gamma,
                      soft_update_step=tau, seed=seed, beta_init=beta_init, num_steps=num_steps)
    if not eval:
        scores = train(env, brain_name, num_eps, agent)
        plot_scores(scores)
    else:
        agent.load('../ckpt', 'ckpt_35')
        evaluate(env, brain_name, agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_lr", default=1e-5, type=float, help='Actor learning rate')
    parser.add_argument("--c_lr", default=1e-5, type=float, help='Critic learning rate')
    parser.add_argument("--gamma", default=0.99, type=float, help='Discount rate')
    parser.add_argument("--seed", default=55321, type=int, help='Random seed')
    parser.add_argument("--buffer_size", default=1e5, type=float, help='Experience Replay buffer '
                                                                       'size')
    parser.add_argument("--batch_size", default=32, type=int, help='Minibatch sample size')
    parser.add_argument("--num_eps", default=2000, type=int, help='Maximum number of episodes')
    parser.add_argument("--alpha", default=0.6, type=float, help='Alpha constant for PRE')
    parser.add_argument("--tau", default=1e-3, type=float, help='Soft update step-size')
    parser.add_argument("--beta_init", default=0.4, type=float, help='Initial value for Beta '
                                                                     'constant for PRE')
    parser.add_argument("--update_every", default=1, type=int, help='Update models every ... '
                                                                    'time steps')
    parser.add_argument("--eval", const=True, nargs='?', help='Evaluate a trained model')
    
    args = parser.parse_args()
    
    main(**vars(args))
