"""Hindsight Experience Replay (HER) utilities for goal-based augmentation.

This module provides a simple 'future' strategy HER augmentation function
that works with the 18D state format in direct joint control:
- State: [joints(6), robot_xyz(3), target_xyz(3), dist_xyz(3), dist_3d(1), vel(2)]
- achieved goal: indices 6:9 (end-effector XYZ, i.e., robot_xyz)
- desired goal (target): indices 9:12 (target_xyz)

Usage:
    from utils.her import her_augmentation
    her_augmentation(agent, obs_list, actions_list, next_obs_list, k=4, ...)

The function will call `agent.remember()` to add augmented transitions.
"""
import random
import numpy as np


def her_augmentation(agent, obs_list, actions_list, next_obs_list,
                     k=4,
                     strategy='future',
                     achieved_idx=slice(6, 9),   # robot_xyz in 18D state
                     desired_idx=slice(9, 12),   # target_xyz in 18D state
                     goal_threshold=0.01,
                     success_reward=0.0,  # kaymen99: pure sparse (0 success, -1 failure)
                     step_reward=-1.0):
    """Augment replay buffer using the 'future' HER strategy.

    For each time step t in the episode, sample k future time steps
    and replace the desired goal (target) in state and next_state with
    the future achieved_goal. Recompute reward/done and store transitions
    via `agent.remember()`.

    Args:
        agent: agent object exposing `remember(state, action, reward, next_state, done)`
        obs_list: list of states (numpy arrays) before actions
        actions_list: list of actions taken
        next_obs_list: list of states after actions
        k: number of HER samples per timestep
        achieved_idx: slice or index for achieved goal within state
        desired_idx: slice or index for desired goal within state
        goal_threshold: distance threshold for success
        success_reward: reward when success
        step_reward: reward otherwise
    """
    if not (len(obs_list) and len(actions_list) and len(next_obs_list)):
        return

    T = len(obs_list)
    added = 0
    for t in range(T):
        if strategy == 'final':
            future_indices = [T - 1] if T > 0 else []
        else:
            # default to 'future' behaviour
            future_indices = list(range(t, T))

        if not future_indices:
            continue

        for _ in range(k):
            if strategy == 'final':
                future_idx = future_indices[0]
            else:
                future_idx = random.choice(future_indices)

            # future achieved goal (from next state at future_idx)
            future_achieved = np.array(next_obs_list[future_idx])[achieved_idx].copy()

            # create augmented state and next_state by replacing desired goal
            state = np.array(obs_list[t], dtype=np.float32).copy()
            next_state = np.array(next_obs_list[t], dtype=np.float32).copy()
            state[desired_idx] = future_achieved
            next_state[desired_idx] = future_achieved

            # recompute reward/done using achieved goal in next_state
            achieved_next = np.array(next_state)[achieved_idx]
            dist = np.linalg.norm(achieved_next - future_achieved)
            if dist <= goal_threshold:
                reward = float(success_reward)
                done = True
            else:
                reward = float(step_reward)
                done = False

            action = np.array(actions_list[t], dtype=np.float32)
            agent.store_transition(state, action, reward, next_state, done)
            added += 1

    return added
