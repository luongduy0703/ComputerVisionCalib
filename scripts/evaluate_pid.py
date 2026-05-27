#!/usr/bin/env python3
"""
PID Tuning Evaluation Script

Compares trajectory tracking performance across different PID strategies:
1. No PID (direct position commands — baseline)
2. Fixed PID (manually tuned gains)
3. RL-optimized PID (trained model)

Usage:
    # Inside Gazebo simulation:
    ros2 run visual_servoing evaluate_pid.py
    
    # Standalone:
    python3 evaluate_pid.py --n-targets 20
"""

import os
import sys
import numpy as np
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controllers.pid_joint_controller import PIDJointController
from controllers.trajectory_generator import TrajectoryGenerator
from controllers.pid_gain_predictor import PIDGainPredictor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Joint limits (matching rl_environment.py)
JOINT_LIMITS_LOW = np.array([-1.5708, -1.0472, -1.5708, -1.5708, -1.5708, -1.5708])
JOINT_LIMITS_HIGH = np.array([1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708])


def generate_test_targets(n_targets: int, seed: int = 42) -> list:
    """Generate reproducible test targets."""
    rng = np.random.RandomState(seed)
    targets = []
    center = (JOINT_LIMITS_LOW + JOINT_LIMITS_HIGH) / 2.0
    half_range = (JOINT_LIMITS_HIGH - JOINT_LIMITS_LOW) / 2.0 * 0.7
    
    for _ in range(n_targets):
        q = rng.uniform(center - half_range, center + half_range)
        q = np.clip(q, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        targets.append(q)
    
    return targets


def evaluate_no_pid(base_env, targets: list, traj_gen: TrajectoryGenerator) -> dict:
    """Evaluate baseline: direct position commands, no PID."""
    print("\n📊 Strategy 1: No PID (Direct Position Commands)")
    print("-" * 50)
    
    import rclpy
    
    results = {'iae': [], 'final_error': [], 'strategy': 'No PID'}
    
    for i, q_goal in enumerate(targets):
        # Move to home
        base_env._move_to_joint_positions(np.zeros(6), duration=2.0)
        time.sleep(0.5)
        for _ in range(5):
            rclpy.spin_once(base_env, timeout_sec=0.1)
        
        q_start = np.array(base_env.joint_positions)
        trajectory = traj_gen.linear(q_start, q_goal, n_steps=100)
        
        # Execute without PID
        total_error = 0.0
        for q_desired in trajectory:
            base_env._move_to_joint_positions(q_desired, duration=0.02)
            rclpy.spin_once(base_env, timeout_sec=0.01)
            q_actual = np.array(base_env.joint_positions)
            total_error += np.sum(np.abs(q_desired - q_actual))
        
        time.sleep(0.3)
        for _ in range(5):
            rclpy.spin_once(base_env, timeout_sec=0.1)
        
        q_final = np.array(base_env.joint_positions)
        final_err = np.linalg.norm(q_goal - q_final)
        
        results['iae'].append(total_error)
        results['final_error'].append(final_err)
        
        print(f"  Target {i+1}/{len(targets)}: IAE={total_error:.4f}, "
              f"Final={np.degrees(final_err):.2f}°")
    
    return results


def evaluate_fixed_pid(base_env, targets: list, traj_gen: TrajectoryGenerator,
                        kp: float = 1.0, ki: float = 0.0, kd: float = 0.05) -> dict:
    """Evaluate with manually-tuned fixed PID gains."""
    print(f"\n📊 Strategy 2: Fixed PID (Kp={kp}, Ki={ki}, Kd={kd})")
    print("-" * 50)
    
    import rclpy
    
    pid = PIDJointController(n_joints=6)
    pid.set_gains(
        Kp=np.ones(6) * kp,
        Ki=np.ones(6) * ki,
        Kd=np.ones(6) * kd
    )
    
    results = {'iae': [], 'final_error': [], 'strategy': f'Fixed PID (Kp={kp})'}
    
    for i, q_goal in enumerate(targets):
        base_env._move_to_joint_positions(np.zeros(6), duration=2.0)
        time.sleep(0.5)
        for _ in range(5):
            rclpy.spin_once(base_env, timeout_sec=0.1)
        
        q_start = np.array(base_env.joint_positions)
        trajectory = traj_gen.linear(q_start, q_goal, n_steps=100)
        
        pid.reset()
        
        for q_desired in trajectory:
            q_actual = np.array(base_env.joint_positions)
            q_command = pid.compute(q_desired, q_actual, dt=0.01)
            q_command = np.clip(q_command, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
            base_env._move_to_joint_positions(q_command, duration=0.02)
            rclpy.spin_once(base_env, timeout_sec=0.01)
        
        time.sleep(0.3)
        for _ in range(5):
            rclpy.spin_once(base_env, timeout_sec=0.1)
        
        q_final = np.array(base_env.joint_positions)
        final_err = np.linalg.norm(q_goal - q_final)
        metrics = pid.get_episode_metrics()
        
        results['iae'].append(metrics['iae'])
        results['final_error'].append(final_err)
        
        print(f"  Target {i+1}/{len(targets)}: IAE={metrics['iae']:.4f}, "
              f"Final={np.degrees(final_err):.2f}°")
    
    return results


def evaluate_rl_pid(base_env, targets: list, traj_gen: TrajectoryGenerator,
                     checkpoint_dir: str = None) -> dict:
    """Evaluate with RL-optimized PID gains."""
    print("\n📊 Strategy 3: RL-Optimized PID")
    print("-" * 50)
    
    import rclpy
    
    predictor = PIDGainPredictor(checkpoint_dir=checkpoint_dir)
    
    if not predictor.has_model() and not predictor.has_fixed_gains():
        print("  ⚠️ No trained model found! Skipping RL-PID evaluation.")
        return {'iae': [], 'final_error': [], 'strategy': 'RL-PID (not available)'}
    
    pid = predictor.get_pid_controller()
    results = {'iae': [], 'final_error': [], 'strategy': 'RL-Optimized PID'}
    
    for i, q_goal in enumerate(targets):
        base_env._move_to_joint_positions(np.zeros(6), duration=2.0)
        time.sleep(0.5)
        for _ in range(5):
            rclpy.spin_once(base_env, timeout_sec=0.1)
        
        q_actual = np.array(base_env.joint_positions)
        q_vel = np.array(base_env.joint_velocities)
        
        # RL predicts optimal gains for this target
        gains = predictor.predict(q_actual, q_vel, q_goal)
        
        q_start = q_actual.copy()
        trajectory = traj_gen.linear(q_start, q_goal, n_steps=100)
        
        pid.reset()
        
        for q_desired in trajectory:
            q_actual = np.array(base_env.joint_positions)
            q_command = pid.compute(q_desired, q_actual, dt=0.01)
            q_command = np.clip(q_command, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
            base_env._move_to_joint_positions(q_command, duration=0.02)
            rclpy.spin_once(base_env, timeout_sec=0.01)
        
        time.sleep(0.3)
        for _ in range(5):
            rclpy.spin_once(base_env, timeout_sec=0.1)
        
        q_final = np.array(base_env.joint_positions)
        final_err = np.linalg.norm(q_goal - q_final)
        metrics = pid.get_episode_metrics()
        
        results['iae'].append(metrics['iae'])
        results['final_error'].append(final_err)
        
        print(f"  Target {i+1}/{len(targets)}: IAE={metrics['iae']:.4f}, "
              f"Final={np.degrees(final_err):.2f}°, "
              f"Kp̄={np.mean(gains['Kp']):.2f}")
    
    return results


def plot_comparison(all_results: list, output_dir: str):
    """Plot comparison chart between strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('PID Strategy Comparison — Trajectory Tracking', 
                 fontsize=16, fontweight='bold')
    
    strategies = [r['strategy'] for r in all_results]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # IAE comparison
    ax = axes[0]
    iae_data = [r['iae'] for r in all_results if r['iae']]
    strategy_labels = [r['strategy'] for r in all_results if r['iae']]
    bp = ax.boxplot(iae_data, labels=strategy_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(iae_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('IAE (rad·steps)')
    ax.set_title('Integral Absolute Error')
    ax.grid(True, alpha=0.3)
    
    # Final error comparison
    ax = axes[1]
    fe_data = [[np.degrees(e) for e in r['final_error']] for r in all_results if r['final_error']]
    bp = ax.boxplot(fe_data, labels=strategy_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(fe_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Final Error (°)')
    ax.set_title('Final Position Error')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(output_dir, f'pid_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Comparison plot saved: {plot_path}")


def print_summary(all_results: list):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<28} {'Mean IAE':>10} {'Mean Final°':>12} {'Best Final°':>12}")
    print("-" * 70)
    
    for r in all_results:
        if not r['iae']:
            print(f"{r['strategy']:<28} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
            continue
        
        mean_iae = np.mean(r['iae'])
        mean_fe = np.mean([np.degrees(e) for e in r['final_error']])
        best_fe = np.min([np.degrees(e) for e in r['final_error']])
        
        print(f"{r['strategy']:<28} {mean_iae:10.4f} {mean_fe:12.2f} {best_fe:12.2f}")
    
    print("=" * 70)


def main():
    """Run PID evaluation comparison."""
    parser = argparse.ArgumentParser(description='Evaluate PID Strategies')
    parser.add_argument('--n-targets', type=int, default=20,
                        help='Number of test targets (default: 20)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to RL-PID checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible targets')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎛️ PID Strategy Evaluation")
    print("=" * 70)
    
    # Generate test targets (reproducible)
    targets = generate_test_targets(args.n_targets, seed=args.seed)
    print(f"\n📌 {len(targets)} test targets generated (seed={args.seed})")
    
    import rclpy
    from rl.rl_environment import RLEnvironment
    
    rclpy.init()
    
    try:
        # Create base environment
        base_env = RLEnvironment(max_episode_steps=200, goal_tolerance=0.01)
        time.sleep(2.0)
        for _ in range(10):
            rclpy.spin_once(base_env, timeout_sec=0.1)
        
        traj_gen = TrajectoryGenerator(n_joints=6, dt=0.01, default_duration=1.0)
        
        all_results = []
        
        # Strategy 1: No PID
        r1 = evaluate_no_pid(base_env, targets, traj_gen)
        all_results.append(r1)
        
        # Strategy 2: Fixed PID
        r2 = evaluate_fixed_pid(base_env, targets, traj_gen, kp=1.0, ki=0.0, kd=0.05)
        all_results.append(r2)
        
        # Strategy 3: RL-PID
        r3 = evaluate_rl_pid(base_env, targets, traj_gen, checkpoint_dir=args.checkpoint)
        all_results.append(r3)
        
        # Summary and plots
        print_summary(all_results)
        
        output_dir = os.path.join(os.path.dirname(__file__), 'training_results', 'png')
        plot_comparison(all_results, output_dir)
        
        base_env.destroy_node()
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
