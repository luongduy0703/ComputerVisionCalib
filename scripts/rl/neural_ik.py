#!/usr/bin/env python3
"""
Neural Inverse Kinematics for 6-DOF Robot Arm (v2 - Position Loss)

KEY INSIGHT: IK is non-unique - multiple joint configurations reach the same position.
Standard MSE on joints fails because it averages all solutions.
SOLUTION: Use position-based loss - FK(predicted_joints) vs target_position

Usage:
  python3 neural_ik.py  # Train
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk_ik_utils import fk, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH


class NeuralIKNetwork(nn.Module):
    """Neural Network for IK with larger capacity"""
    
    def __init__(self, input_dim=3, hidden_dim=512, output_dim=6):
        super().__init__()
        
        # Larger network for better representational capacity
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Store joint limits as buffers (move with model to device)
        self.register_buffer('joint_low', torch.tensor(JOINT_LIMITS_LOW, dtype=torch.float32))
        self.register_buffer('joint_high', torch.tensor(JOINT_LIMITS_HIGH, dtype=torch.float32))
    
    def forward(self, x):
        normalized = self.network(x)
        # Denormalize to actual joint angles
        joints = (normalized + 1) / 2 * (self.joint_high - self.joint_low) + self.joint_low
        return joints


def fk_batch_torch(joints, device):
    """
    Batch FK computation using PyTorch for gradient flow.
    
    Exactly replicates the fk() chain from fk_ik_utils.py:
      base_link → T_r6 → T_r18 → T_r19 → T_j20(Rz) → T_r21 → T_j22(Ry)
      → T_j23(Ry) → T_r24 → T_r25 → T_j26(Rz) → T_r27 → T_j28(Ry)
      → T_r29 → T_j30(Ry) → T_r32 → T_r33 → bibut_1 (EE)
    
    Joint axes (from URDF):
      J1 (Rev20): axis=(0,0,-1)  → Rz(-q[0])
      J2 (Rev22): axis=(0,-1,0) → Ry(-q[1])
      J3 (Rev23): axis=(0,-1,0) → Ry(-q[2])
      J4 (Rev26): axis=(0,0,-1) → Rz(-q[3])
      J5 (Rev28): axis=(0,-1,0) → Ry(-q[4])
      J6 (Rev30): axis=(0,1,0)  → Ry(+q[5])
    """
    batch_size = joints.shape[0]
    
    def make_Rz(theta):
        """Rz rotation matrix (batch)"""
        c, s = torch.cos(theta), torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        return torch.stack([
            torch.stack([c, -s, zeros], dim=-1),
            torch.stack([s,  c, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1)
        ], dim=-2)
    
    def make_Ry(theta):
        """Ry rotation matrix (batch)"""
        c, s = torch.cos(theta), torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        return torch.stack([
            torch.stack([c,  zeros, s], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-s, zeros, c], dim=-1)
        ], dim=-2)
    
    # Start at origin with identity rotation
    pos = torch.zeros(batch_size, 3, device=device)
    R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    def apply_translation(pos, R, tx, ty, tz):
        """Apply a fixed translation in the current local frame"""
        t = torch.tensor([tx, ty, tz], device=device, dtype=torch.float32)
        return pos + torch.bmm(R, t.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)).squeeze(-1)
    
    # Fixed: base_link → old_component__6__1
    pos = apply_translation(pos, R, -0.046528, 0.031724, 0.748891)
    # Fixed: → old_component__14__1
    pos = apply_translation(pos, R, -0.093, 0.0, -0.01)
    # Fixed: → old_component__15__1
    pos = apply_translation(pos, R, 0.04889, -0.028138, -0.00625)
    
    # Rev 20: axis=(0,0,-1) → Rz(-q[0])
    pos = apply_translation(pos, R, -0.034687, -0.0039, -0.0162)
    R = torch.bmm(R, make_Rz(-joints[:, 0]))
    
    # Fixed: → old_component__17__1
    pos = apply_translation(pos, R, -0.048931, -0.007, -0.033724)
    
    # Rev 22: axis=(0,-1,0) → Ry(-q[1])
    pos = apply_translation(pos, R, 0.034687, -0.0192, -0.0039)
    R = torch.bmm(R, make_Ry(-joints[:, 1]))
    
    # Rev 23: axis=(0,-1,0) → Ry(-q[2])
    pos = apply_translation(pos, R, 0.0, 0.0, -0.155)
    R = torch.bmm(R, make_Ry(-joints[:, 2]))
    
    # Fixed: → old_component__20__1
    pos = apply_translation(pos, R, -0.0039, 0.0192, -0.034687)
    # Fixed: → old_component__21__1
    pos = apply_translation(pos, R, 0.03375, 0.0362, -0.042816)
    
    # Rev 26: axis=(0,0,-1) → Rz(-q[3])
    pos = apply_translation(pos, R, 0.0, -0.00995, -0.0148)
    R = torch.bmm(R, make_Rz(-joints[:, 3]))
    
    # Fixed: → old_component__23__1
    pos = apply_translation(pos, R, 0.0152, -0.023, -0.0425)
    
    # Rev 28: axis=(0,-1,0) → Ry(-q[4])
    pos = apply_translation(pos, R, -0.00995, -0.0148, 0.0)
    R = torch.bmm(R, make_Ry(-joints[:, 4]))
    
    # Fixed: → old_component__25__1
    pos = apply_translation(pos, R, -0.0152, 0.0075, -0.075)
    
    # Rev 30: axis=(0,1,0) → Ry(+q[5])
    pos = apply_translation(pos, R, 0.02045, 0.015, 0.0)
    R = torch.bmm(R, make_Ry(joints[:, 5]))
    
    # Fixed: → but_1
    pos = apply_translation(pos, R, 0.0, 0.01225, -0.01)
    # Fixed: → bibut_1 (end-effector)
    pos = apply_translation(pos, R, 0.0, 0.0, -0.045)
    
    return pos


class NeuralIK:
    """Neural IK with position-based training"""
    
    def __init__(self, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = NeuralIKNetwork().to(self.device)
        
        # Workspace bounds (base_link frame)
        # Board at base_link X≈-0.50, Y≈0, Z≈0.56
        # These defaults are overwritten by generate_training_data()
        self.pos_min = np.array([-0.55, -0.43, 0.26])
        self.pos_max = np.array([ 0.30,  0.43, 0.94])
        
        print(f"✅ Neural IK v2 (base_link workspace) initialized on {self.device}")
        print(f"   Workspace: X=[{self.pos_min[0]:.2f}, {self.pos_max[0]:.2f}], "
              f"Y=[{self.pos_min[1]:.2f}, {self.pos_max[1]:.2f}], "
              f"Z=[{self.pos_min[2]:.2f}, {self.pos_max[2]:.2f}]")

    
    def generate_training_data(self, n_samples=500000):
        """Generate FK samples"""
        print(f"📊 Generating {n_samples:,} FK samples...")
        
        valid_positions = []
        valid_joints = []
        
        attempts = 0
        while len(valid_joints) < n_samples and attempts < n_samples * 2:
            attempts += 1
            joints = np.random.uniform(JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
            
            try:
                x, y, z = fk(joints)
                pos = np.array([x, y, z])
                if np.isfinite(pos).all() and pos[2] > 0.02:
                    valid_positions.append(pos)
                    valid_joints.append(joints)
            except:
                continue
            
            if len(valid_joints) % 100000 == 0 and len(valid_joints) > 0:
                print(f"   Generated {len(valid_joints):,} samples...")
        
        positions = np.array(valid_positions, dtype=np.float32)
        joints = np.array(valid_joints, dtype=np.float32)
        
        self.pos_min = positions.min(axis=0)
        self.pos_max = positions.max(axis=0)
        
        print(f"✅ Generated {len(joints):,} valid FK samples")
        print(f"   Position range: X=[{self.pos_min[0]:.3f}, {self.pos_max[0]:.3f}], "
              f"Y=[{self.pos_min[1]:.3f}, {self.pos_max[1]:.3f}], "
              f"Z=[{self.pos_min[2]:.3f}, {self.pos_max[2]:.3f}]")
        
        return positions, joints
    
    def normalize_position(self, pos):
        return 2 * (pos - self.pos_min) / (self.pos_max - self.pos_min + 1e-8) - 1
    
    def train(self, positions, joints, epochs=200, batch_size=1024, lr=1e-3):
        """Train using POSITION-BASED LOSS (not joint MSE!)"""
        print(f"\n🎓 Training Neural IK v2 (Position Loss)...")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
        
        # Normalize positions for input
        positions_norm = self.normalize_position(positions)
        
        X = torch.tensor(positions_norm, dtype=torch.float32)
        target_pos = torch.tensor(positions, dtype=torch.float32)  # Keep unnormalized for FK comparison
        
        n = len(X)
        n_train = int(0.9 * n)
        indices = np.random.permutation(n)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        
        X_train, X_val = X[train_idx], X[val_idx]
        pos_train, pos_val = target_pos[train_idx], target_pos[val_idx]
        
        train_loader = DataLoader(
            TensorDataset(X_train, pos_train),
            batch_size=batch_size,
            shuffle=True
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_error = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_errors = []
            
            for batch_x, batch_target_pos in train_loader:
                batch_x = batch_x.to(self.device)
                batch_target_pos = batch_target_pos.to(self.device)
                
                optimizer.zero_grad()
                
                # Predict joints
                pred_joints = self.model(batch_x)
                
                # FK to get predicted position
                pred_pos = fk_batch_torch(pred_joints, self.device)
                
                # Position loss (in mm for better gradients)
                loss = torch.mean(torch.sum((pred_pos - batch_target_pos) ** 2, dim=1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                error_mm = torch.sqrt(torch.sum((pred_pos - batch_target_pos) ** 2, dim=1)).mean() * 1000
                train_errors.append(error_mm.item())
            
            train_error = np.mean(train_errors)
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                X_val_d = X_val.to(self.device)
                pos_val_d = pos_val.to(self.device)
                
                val_pred_joints = self.model(X_val_d)
                val_pred_pos = fk_batch_torch(val_pred_joints, self.device)
                val_error = torch.sqrt(torch.sum((val_pred_pos - pos_val_d) ** 2, dim=1)).mean() * 1000
                val_error = val_error.item()
            
            scheduler.step(val_error)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: Train Error={train_error:.2f}mm, Val Error={val_error:.2f}mm")
            
            if val_error < best_val_error:
                best_val_error = val_error
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        print(f"\n✅ Training complete! Best Val Error: {best_val_error:.2f}mm")
        
        # Final evaluation
        self._evaluate_accuracy(positions)
    
    def _evaluate_accuracy(self, positions, n_samples=1000):
        """Evaluate accuracy"""
        print("\n📏 Evaluating Neural IK accuracy...")
        
        indices = np.random.choice(len(positions), min(n_samples, len(positions)), replace=False)
        errors = []
        
        self.model.eval()
        with torch.no_grad():
            for i in indices:
                target_pos = positions[i]
                pred_joints = self.predict(target_pos)
                pred_x, pred_y, pred_z = fk(pred_joints)
                pred_pos = np.array([pred_x, pred_y, pred_z])
                error = np.linalg.norm(target_pos - pred_pos) * 1000
                errors.append(error)
        
        errors = np.array(errors)
        print(f"   Mean error:   {errors.mean():.2f} mm")
        print(f"   Median error: {np.median(errors):.2f} mm")
        print(f"   95th %ile:    {np.percentile(errors, 95):.2f} mm")
        print(f"   < 10mm:       {(errors < 10).mean()*100:.1f}%")
        print(f"   < 5mm:        {(errors < 5).mean()*100:.1f}%")
    
    def predict(self, target_position, current_joints=None, refine=False, max_jump=0.5):
        """
        Predict joint angles for target position
        
        Args:
            target_position: [x, y, z] target end-effector position
            current_joints: Optional current joint angles for smoother motion
            refine: If True, apply Jacobian refinement for higher accuracy
            max_jump: Maximum allowed joint angle change (rad) before warning
            
        Returns:
            joints: [j1, j2, j3, j4, j5, j6] predicted joint angles
        """
        pos_norm = self.normalize_position(np.array(target_position))
        pos_tensor = torch.tensor(pos_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            joints = self.model(pos_tensor)
        
        joints = joints.cpu().numpy()[0]
        
        # Safety: Clamp to joint limits
        joints = np.clip(joints, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        
        # Multi-solution: Check if motion is too large
        if current_joints is not None:
            delta = np.abs(joints - current_joints)
            if delta.max() > max_jump:
                pass  # Could add warning or alternative solution search here
        
        # Jacobian refinement for higher accuracy (optional)
        if refine:
            joints = self._jacobian_refine(joints, target_position, n_steps=2)
        
        return joints
    
    def _jacobian_refine(self, joints, target_pos, n_steps=2, step_size=0.5):
        """
        Jacobian-based refinement to improve Neural IK accuracy
        
        Iteratively corrects joint angles to reduce position error:
        Δθ = J⁺ × Δp where J is the Jacobian matrix
        """
        target_pos = np.array(target_pos)
        joints = np.array(joints, dtype=np.float64)
        
        for step in range(n_steps):
            # Get current end-effector position
            current_pos = np.array(fk(joints))
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < 0.001:  # < 1mm, good enough
                break
            
            # Compute numerical Jacobian (3x6)
            J = self._compute_jacobian(joints)
            
            # Pseudo-inverse for underdetermined system (6 joints, 3 outputs)
            J_pinv = np.linalg.pinv(J)
            
            # Compute joint correction
            delta_joints = J_pinv @ error
            
            # Apply with conservative step size
            joints = joints + step_size * delta_joints
            
            # Clamp to limits after each step
            joints = np.clip(joints, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        
        return joints.astype(np.float32)
    
    def _compute_jacobian(self, joints, epsilon=1e-5):
        """
        Compute numerical Jacobian matrix (3x6)
        J[i,j] = ∂position_i / ∂joint_j
        """
        J = np.zeros((3, 6))
        base_pos = np.array(fk(joints))
        
        for j in range(6):
            # Perturb joint j
            joints_plus = joints.copy()
            joints_plus[j] += epsilon
            pos_plus = np.array(fk(joints_plus))
            
            # Numerical derivative
            J[:, j] = (pos_plus - base_pos) / epsilon
        
        return J
    
    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'pos_min': self.pos_min,
            'pos_max': self.pos_max
        }
        torch.save(checkpoint, path)
        print(f"💾 Neural IK saved to: {path}")
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.pos_min = checkpoint['pos_min']
        self.pos_max = checkpoint['pos_max']
        self.model.eval()
        print(f"✅ Neural IK loaded from: {path}")


def main():
    print("=" * 60)
    print("🧠 Neural IK Training v2 (Position-Based Loss)")
    print("=" * 60)
    
    nik = NeuralIK()
    positions, joints = nik.generate_training_data(n_samples=500000)
    nik.train(positions, joints, epochs=200)
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'neural_ik.pth')
    nik.save(save_path)
    
    print("\n" + "=" * 60)
    print("✅ Neural IK v2 training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
