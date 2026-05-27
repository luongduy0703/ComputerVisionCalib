"""
SAC (Soft Actor-Critic) agent implementation for Gazebo 4DOF Robot.

Implements a PyTorch-based SAC with the API expected by `train_robot.py`:
- __init__(state_dim, n_actions, max_action, min_action, ...)
- choose_action(state, evaluate=False)
- remember(state, action, reward, next_state, done)
- learn() -> (actor_loss, critic_loss)
- save_models(episode=None)
- load_models(actor_path, critic_path)

SAC is an off-policy actor-critic algorithm that maximizes both expected return
and entropy, leading to better exploration and more robust policies.
"""
from collections import deque
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# ============================================================================
# SAC HYPERPARAMETERS & CONFIGURATION
# ============================================================================

# Learning rates
SAC_ACTOR_LR = 3e-4
SAC_CRITIC_LR = 3e-4
SAC_ALPHA_LR = 3e-4         # Temperature parameter learning rate

# Discount and update rates
SAC_GAMMA = 0.99
SAC_TAU = 0.05  # 0.05 for faster adaptation

# SAC-specific: Entropy regularization
SAC_ALPHA = 0.2             # Initial temperature for entropy regularization
SAC_AUTO_ENTROPY_TUNING = True  # Automatically adjust alpha

# Replay buffer and batch size
SAC_BUFFER_SIZE = int(1e6)
SAC_BATCH_SIZE = 256


def _to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class GaussianActor(nn.Module):
    """
    Stochastic actor that outputs a Gaussian distribution over actions.
    Uses reparameterization trick for backpropagation through sampling.
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dims=(256, 256)):
        super().__init__()
        self.max_action = max_action
        
        # Shared feature layers
        self.l1 = nn.Linear(state_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # Mean and log_std heads
        self.mean_linear = nn.Linear(hidden_dims[1], action_dim)
        self.log_std_linear = nn.Linear(hidden_dims[1], action_dim)
        
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action using reparameterization trick.
        Returns: action, log_prob, mean
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        noise = Normal(0, 1).sample(mean.shape).to(state.device) # Sample noise from standard normal
        
        # Squash action to [-1, 1] using tanh
        y_t = torch.tanh(mean + std * noise)
        action = y_t * _to_tensor(self.max_action, y_t.device)
        
        # Compute log probability
        log_prob = normal.log_prob(mean + std * noise)
        log_prob -= torch.log(_to_tensor(self.max_action, y_t.device) * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * _to_tensor(self.max_action, mean.device)
        
        return action, log_prob, mean
    
    def get_action(self, state, evaluate=False):
        """Get action for inference (no gradient)."""
        mean, log_std = self.forward(state)
        
        if evaluate:
            # Deterministic action for evaluation
            return torch.tanh(mean) * self.max_action
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t) * self.max_action


class SoftQNetwork(nn.Module):
    """
    Soft Q-Network (Critic) for SAC.
    Takes state and action as input, outputs Q-value.
    """
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.storage = deque(maxlen=int(max_size))

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward.reshape(-1, 1), next_state, done.reshape(-1, 1)

    def size(self):
        return len(self.storage)
    
    def save(self, filepath):
        """Save replay buffer to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.storage), f)
    
    def load(self, filepath):
        """Load replay buffer from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.storage = deque(data, maxlen=self.storage.maxlen)


class SACAgentGazebo:
    """
    Soft Actor-Critic (SAC) Agent for Gazebo robot control.
    
    SAC is an off-policy maximum entropy RL algorithm that:
    - Uses a stochastic policy (Gaussian) for better exploration
    - Maximizes both expected return AND entropy
    - Uses twin Q-networks to reduce overestimation bias
    - Automatically adjusts temperature (alpha) for entropy regularization
    """
    
    def __init__(
        self,
        state_dim,
        n_actions,
        max_action=1.0,
        min_action=-1.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.05,
        alpha=0.2,
        auto_entropy_tuning=True,
        buffer_size=int(1e6),
        batch_size=256,
        device=None,
        seed=0,
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            n_actions: Dimension of action space
            max_action: Maximum action value
            min_action: Minimum action value
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critics
            alpha_lr: Learning rate for temperature parameter
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Initial temperature for entropy regularization
            auto_entropy_tuning: Whether to automatically adjust alpha
            buffer_size: Replay buffer size
            batch_size: Mini-batch size for training
            device: Torch device (cuda/cpu)
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_dim = state_dim
        self.action_dim = n_actions
        
        # Handle array or scalar max/min action
        if isinstance(max_action, (list, np.ndarray)):
            self.max_action = np.array(max_action, dtype=np.float32)
        else:
            self.max_action = float(max_action)
        
        if isinstance(min_action, (list, np.ndarray)):
            self.min_action = np.array(min_action, dtype=np.float32)
        else:
            self.min_action = float(min_action)
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning
        
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        # Actor (Gaussian policy)
        self.actor = GaussianActor(state_dim, n_actions, self.max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Twin Q-networks (critics)
        self.critic1 = SoftQNetwork(state_dim, n_actions).to(self.device)
        self.critic2 = SoftQNetwork(state_dim, n_actions).to(self.device)
        self.critic1_target = SoftQNetwork(state_dim, n_actions).to(self.device)
        self.critic2_target = SoftQNetwork(state_dim, n_actions).to(self.device)
        
        # Copy weights to targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Entropy temperature (alpha)
        if auto_entropy_tuning:
            # Target entropy: -dim(A) * 0.5 for MORE exploration (breaks plateau)
            # Standard SAC uses -dim(A), but lower target = higher entropy maintained
            self.target_entropy = -torch.tensor(n_actions * 0.5, dtype=torch.float32, device=self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        
        # Training stats
        self.total_it = 0
        
        # Checkpoint directory
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'sac_gazebo')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"✅ SAC Agent initialized:")
        print(f"   State dim: {state_dim}, Actions: {n_actions}")
        print(f"   Device: {self.device}")
        print(f"   Gamma: {gamma}, Tau: {tau}")
        print(f"   Auto entropy tuning: {auto_entropy_tuning}")
        print(f"   Initial alpha: {self.alpha:.4f}")
    
    def select_action(self, state, evaluate=False):
        """
        Select action based on current policy.
        
        Args:
            state: Current state
            evaluate: If True, use deterministic (mean) action
            
        Returns:
            action: Action array
        """
        if isinstance(state, np.ndarray):
            state = state.reshape(1, -1)
        state_t = _to_tensor(state, self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.get_action(state_t, evaluate=evaluate)
        self.actor.train()
        
        action = action.cpu().numpy().flatten()
        return np.clip(action, self.min_action, self.max_action)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = float(reward)
        done = float(done)
        self.replay_buffer.add((state, action, reward, next_state, done))
    
    def train(self):
        """
        Update actor and critics using SAC algorithm.
        
        Returns:
            (actor_loss, critic_loss) or (None, None) if not enough samples
        """
        # Must have at least a full batch to start training
        if self.replay_buffer.size() < self.batch_size:
            return None, None
        
        self.total_it += 1
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = _to_tensor(state, self.device)
        action = _to_tensor(action, self.device)
        reward = _to_tensor(reward, self.device)
        next_state = _to_tensor(next_state, self.device)
        done = _to_tensor(done, self.device)
        
        # ==================== Update Critics ====================
        with torch.no_grad():
            # Sample next action from current policy
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            
            # Compute target Q-value (minimum of two targets for stability)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Soft Bellman backup with entropy term
            target_q = reward + (1 - done) * self.gamma * (target_q - self.alpha * next_log_prob)
        
        # Current Q estimates
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        # Critic losses (MSE)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # ==================== Update Actor ====================
        # Sample action from current policy
        new_action, log_prob, _ = self.actor.sample(state)
        
        # Compute Q-values for new actions
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss: maximize Q - alpha * log_prob (equivalently minimize negative)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ==================== Update Temperature (Alpha) ====================
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ==================== Soft Update Target Networks ====================
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()
    
    def save_models(self, episode=None):
        """Save actor and critic networks."""
        fname = "sac_best.pth" if episode is None else f"sac_ep_{episode}.pth"
        
        path_actor = os.path.join(self.checkpoint_dir, f'actor_{fname}')
        path_critic1 = os.path.join(self.checkpoint_dir, f'critic1_{fname}')
        path_critic2 = os.path.join(self.checkpoint_dir, f'critic2_{fname}')
        
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic1.state_dict(), path_critic1)
        torch.save(self.critic2.state_dict(), path_critic2)
        
        # Save alpha if using auto entropy tuning
        if self.auto_entropy_tuning:
            path_alpha = os.path.join(self.checkpoint_dir, f'alpha_{fname}')
            torch.save({'log_alpha': self.log_alpha, 'alpha': self.alpha}, path_alpha)
        
        print(f"💾 SAC models saved: {fname}")
    
    def load_models(self, actor_path, critic_path=None):
        """
        Load actor and critic networks.
        
        Args:
            actor_path: Path to actor checkpoint
            critic_path: Path to critic checkpoint (optional, will infer if not provided)
        """
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        
        # Try to load critics
        if critic_path:
            self.critic1.load_state_dict(torch.load(critic_path, map_location=self.device))
        else:
            # Infer critic paths from actor path
            critic1_path = actor_path.replace('actor_', 'critic1_')
            critic2_path = actor_path.replace('actor_', 'critic2_')
            
            if os.path.exists(critic1_path):
                self.critic1.load_state_dict(torch.load(critic1_path, map_location=self.device))
            if os.path.exists(critic2_path):
                self.critic2.load_state_dict(torch.load(critic2_path, map_location=self.device))
        
        # Try to load alpha
        alpha_path = actor_path.replace('actor_', 'alpha_')
        if os.path.exists(alpha_path) and self.auto_entropy_tuning:
            alpha_data = torch.load(alpha_path, map_location=self.device)
            self.log_alpha = alpha_data['log_alpha']
            self.alpha = alpha_data['alpha']
        
        # Set to eval mode
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        
        print(f"✅ SAC models loaded from: {actor_path}")
