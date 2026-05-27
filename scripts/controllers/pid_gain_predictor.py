#!/usr/bin/env python3
"""
PID Gain Predictor — Inference Wrapper for Deployment

Loads a trained SAC actor model and predicts optimal PID gains
for a given robot state and target configuration.

Usage:
    # In Python:
    predictor = PIDGainPredictor(checkpoint_path='checkpoints/sac_pid_tuning/')
    gains = predictor.predict(q_actual, q_vel, q_goal)
    
    # Command-line test:
    python3 pid_gain_predictor.py --checkpoint checkpoints/sac_pid_tuning/
"""

import os
import sys
import json
import numpy as np
import argparse
from typing import Dict, Tuple, Optional

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from controllers.pid_joint_controller import PIDJointController


class PIDGainPredictor:
    """
    Inference-only wrapper for predicting optimal PID gains.
    
    Loads a trained SAC actor and uses it to predict PID gains
    given the current state and target configuration.
    
    Can also fall back to a fixed best-gains JSON if no model is available.
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None, n_joints: int = 6):
        """
        Initialize predictor.
        
        Args:
            checkpoint_dir: Path to SAC checkpoint directory
            n_joints: Number of joints
        """
        self.n_joints = n_joints
        self.state_dim = 4 * n_joints  # 24D
        self.action_dim = 3 * n_joints  # 18D
        self.pid = PIDJointController(n_joints=n_joints)
        
        self.actor = None
        self.best_gains_fixed = None
        
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                '..', 'checkpoints', 'sac_pid_tuning'
            )
        
        self.checkpoint_dir = checkpoint_dir
        
        # Try loading SAC actor
        self._load_model()
        
        # Try loading fixed best gains as fallback
        self._load_best_gains()
    
    def _load_model(self):
        """Load trained SAC actor for inference."""
        try:
            import torch
            
            actor_path = os.path.join(self.checkpoint_dir, 'actor_sac_best.pth')
            if not os.path.exists(actor_path):
                print(f"[PIDGainPredictor] No actor model at {actor_path}")
                return
            
            # Import SAC agent to get actor architecture
            from agents.sac_agent import SACAgentGazebo
            
            agent = SACAgentGazebo(
                state_dim=self.state_dim,
                n_actions=self.action_dim,
                max_action=np.ones(self.action_dim),
                min_action=-np.ones(self.action_dim),
            )
            agent.load_models(actor_path)
            self.actor = agent
            
            print(f"[PIDGainPredictor] ✅ Loaded actor from {actor_path}")
            
        except Exception as e:
            print(f"[PIDGainPredictor] ⚠️ Could not load model: {e}")
    
    def _load_best_gains(self):
        """Load fixed best gains from JSON as fallback."""
        gains_path = os.path.join(self.checkpoint_dir, 'best_gains.json')
        if os.path.exists(gains_path):
            try:
                with open(gains_path, 'r') as f:
                    gains = json.load(f)
                self.best_gains_fixed = {
                    'Kp': np.array(gains['Kp']),
                    'Ki': np.array(gains['Ki']),
                    'Kd': np.array(gains['Kd']),
                }
                print(f"[PIDGainPredictor] ✅ Loaded best gains from {gains_path}")
            except Exception as e:
                print(f"[PIDGainPredictor] ⚠️ Could not load best gains: {e}")
    
    def predict(self, q_actual: np.ndarray, q_vel: np.ndarray, 
                q_goal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict optimal PID gains for the given state.
        
        Args:
            q_actual: Current joint positions [n_joints]
            q_vel: Current joint velocities [n_joints]
            q_goal: Target joint positions [n_joints]
        
        Returns:
            Dict with 'Kp', 'Ki', 'Kd' arrays
        """
        q_actual = np.array(q_actual, dtype=np.float32)
        q_vel = np.array(q_vel, dtype=np.float32)
        q_goal = np.array(q_goal, dtype=np.float32)
        error = q_goal - q_actual
        
        # Build 24D state
        state = np.concatenate([q_actual, q_vel, q_goal, error])
        
        if self.actor is not None:
            # Use trained model
            action = self.actor.select_action(state, evaluate=True)
            self.pid.set_gains_from_normalized(action)
        elif self.best_gains_fixed is not None:
            # Use fixed best gains (fallback)
            self.pid.set_gains(
                self.best_gains_fixed['Kp'],
                self.best_gains_fixed['Ki'],
                self.best_gains_fixed['Kd']
            )
        else:
            # Use defaults
            print("[PIDGainPredictor] ⚠️ No model or gains found, using defaults")
        
        return self.pid.get_gains_dict()
    
    def get_pid_controller(self) -> PIDJointController:
        """Return the PID controller instance (with current gains set)."""
        return self.pid
    
    def has_model(self) -> bool:
        """Check if a trained model is loaded."""
        return self.actor is not None
    
    def has_fixed_gains(self) -> bool:
        """Check if fixed best gains are available."""
        return self.best_gains_fixed is not None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line test for PID gain prediction."""
    parser = argparse.ArgumentParser(description='PID Gain Predictor')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to SAC checkpoint directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎛️ PID Gain Predictor — Test")
    print("=" * 60)
    
    predictor = PIDGainPredictor(checkpoint_dir=args.checkpoint)
    
    # Test prediction with random state
    q_actual = np.zeros(6)
    q_vel = np.zeros(6)
    q_goal = np.array([0.5, -0.3, 0.8, 0.0, -0.5, 0.2])
    
    print(f"\nTest input:")
    print(f"  q_actual: {q_actual}")
    print(f"  q_goal:   {np.round(q_goal, 2)}")
    
    gains = predictor.predict(q_actual, q_vel, q_goal)
    
    print(f"\nPredicted PID gains:")
    print(f"  Kp: {np.round(gains['Kp'], 3)}")
    print(f"  Ki: {np.round(gains['Ki'], 4)}")
    print(f"  Kd: {np.round(gains['Kd'], 4)}")
    print(f"\n  Model loaded: {predictor.has_model()}")
    print(f"  Fixed gains: {predictor.has_fixed_gains()}")


if __name__ == '__main__':
    main()
