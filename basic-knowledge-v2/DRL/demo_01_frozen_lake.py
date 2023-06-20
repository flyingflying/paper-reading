# -*- coding:utf-8 -*-
# Author: lqxu

# reference: https://huggingface.co/learn/deep-rl-course/unit2/hands-on?fw=pt 

# Gym 是 OpenAI 开源的, 提供了大量可以用于 `强化学习` 的 `环境程序`
# 这样的库很重要, 我们不需要再去封装 `环境程序` 了, 而是更多地关注 `agent 程序` 的编写
# 后来, 这个项目独立出去了, 成立了 Farama Foundation 公司, 并将项目名字改为 Gymnasium
import gymnasium as gym
from gymnasium import Env

import os 
import random
from dataclasses import dataclass

import imageio
import numpy as np
from tqdm import tqdm
from numpy import ndarray
import matplotlib.pyplot as plt

# env 是 `环境程序`
# 关于 frozen lake 相关的内容可以参考: https://gymnasium.farama.org/environments/toy_text/frozen_lake/ 
env: Env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")


def test_env():
    # 初始化环境, 并获得 初始状态
    init_state_id, info = env.reset()
    print(init_state_id, info)
    
    def show_cur_ui():
        # 返回当前的程序 UI 图
        image = env.render()  # ndarray 的形式
        
        plt.imshow(image)
        plt.show()
    
    show_cur_ui()
    
    # state 集合就是每一个格子, 自然是用数字来表示的
    print(env.observation_space)
    # action 集合就是 上下左右
    print(env.action_space)

    def move(action_id: int):
        next_state_id, reward, terminated, truncated, info = env.step(action_id)
        # 只有走到终点 reward 才是 1, 其它时候 reward 都是 0
        print(next_state_id, reward, terminated, truncated, info)
        
        show_cur_ui()

    action_ids = [1, 1, 2, 1, 2, 2]

    for action_id in action_ids:
        move(action_id)


def init_Qtable() -> ndarray:

    n_states, n_actions = env.observation_space.n, env.action_space.n
    Qtable = np.zeros((n_states, n_actions))

    return Qtable


def greedy_policy(Qtable: ndarray, state_id: int) -> int:
    # policy: 根据当前的 state, 生成下一步的 state
    # Exploitation (利用): take the action with the highest state, action value
    action_id = np.argmax(Qtable[state_id][:])
    return action_id


def epsilon_greedy_policy(Qtable: ndarray, state_id: int, epsilon: float) -> int:
    if random.uniform(0, 1) > epsilon:
        # Exploitation 利用
        action_id = greedy_policy(Qtable, state_id)
    else:
        # Exploration 探索
        action_id = env.action_space.sample()

    return action_id


@dataclass
class HyperParameters:
    # Training parameters
    n_training_episodes = 10000  # Total training episodes
    learning_rate = 0.7  # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100  # Total number of test episodes

    # Environment parameters
    env_id = "FrozenLake-v1"  # Name of the environment
    max_steps = 99  # Max steps per episode
    gamma = 0.95  # Discounting rate
    eval_seed = []  # The evaluation seed of the environment

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.0005  # Exponential decay rate for exploration prob


def train(Qtable: ndarray, hyper_params: HyperParameters, ):
    for episode in tqdm(range(hyper_params.n_training_episodes)):  # epoches
        # Reduce epsilon (because we need less and less exploration)
        epsilon = hyper_params.min_epsilon + (
            (hyper_params.max_epsilon - hyper_params.min_epsilon) * np.exp(-hyper_params.decay_rate * episode)
        )
        # Reset the environment
        state_id, _ = env.reset()

        # repeat
        for _ in range(hyper_params.max_steps):  # steps
            # Choose the action At using epsilon greedy policy
            action_id = epsilon_greedy_policy(Qtable, state_id, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state_id, reward, terminated, truncated, info = env.step(action_id)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state_id][action_id] = Qtable[state_id][action_id] + hyper_params.learning_rate * (
                reward + hyper_params.gamma * np.max(Qtable[new_state_id]) - Qtable[state_id][action_id]
            )

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state_id = new_state_id
    return Qtable


def evaluate_agent(Qtable: ndarray, hyper_params: HyperParameters, seed: list[int] = None):

    episode_rewards = []
    for episode in tqdm(range(hyper_params.n_eval_episodes)):
        if seed:
            state_id, _ = env.reset(seed=seed[episode])
        else:
            state_id, _ = env.reset()

        total_rewards_ep = 0

        for _ in range(hyper_params.max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action_id = greedy_policy(Qtable, state_id)
            new_state_id, reward, terminated, truncated, info = env.step(action_id)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state_id = new_state_id
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward


def record_video(Qtable: ndarray, out_directory: str):
    """
    利用 imageio 将多个图片合成一个视频
    """
    images = []
    terminated = False
    truncated = False
    state_id, _ = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not terminated or truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action_id = np.argmax(Qtable[state_id][:])
        state_id, reward, terminated, truncated, info = env.step(
            action_id
        )  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for img in images], fps=1)


if __name__ == "__main__":
    # test_env()
    
    Qtable = init_Qtable()
    
    print(Qtable)
    
    hyper_params = HyperParameters()
    
    train(Qtable, hyper_params)
    
    evaluate_agent(Qtable, hyper_params)
    
    print(Qtable)
    
    # record_video(Qtable, os.path.join(os.path.dirname(__file__), "game_video.mp4"))
    
