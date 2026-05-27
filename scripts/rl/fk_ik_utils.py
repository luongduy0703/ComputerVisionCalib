"""
Forward Kinematics for the 6-DOF UAV Robot Arm (hop_description(1) — robot_arm2)
Pure Python (NO numpy) to prevent tkinter segfaults in ROS nodes.

Exact 4x4 homogeneous transforms from the URDF joint origins.

Kinematic chain (from new_arm.xacro):
  base_link
    → Rigid 6:  xyz=(-0.046528, 0.031724, 0.748891)  → old_component__6__1
    → Rigid 18: xyz=(-0.093, 0.0, -0.01)              → old_component__14__1
    → Rigid 19: xyz=(0.04889, -0.028138, -0.00625)    → old_component__15__1
    → Rev 20:   xyz=(-0.034687, -0.0039, -0.0162)     axis=(0 0 -1)  → old_component__16__1
    → Rigid 21: xyz=(-0.048931, -0.007, -0.033724)    → old_component__17__1
    → Rev 22:   xyz=(0.034687, -0.0192, -0.0039)      axis=(0 1 0)   → old_component__18__1
    → Rev 23:   xyz=(0.0, 0.0, -0.155)                axis=(0 1 0)   → old_component__19__1
    → Rigid 24: xyz=(-0.0039, 0.0192, -0.034687)      → old_component__20__1
    → Rigid 25: xyz=(0.03375, 0.0362, -0.042816)      → old_component__21__1
    → Rev 26:   xyz=(0.0, -0.00995, -0.0148)          axis=(0 0 -1)  → old_component__22__1
    → Rigid 27: xyz=(0.0152, -0.023, -0.0425)         → old_component__23__1
    → Rev 28:   xyz=(-0.00995, -0.0148, 0.0)          axis=(0 1 0)   → old_component__24__1
    → Rigid 29: xyz=(-0.0152, 0.0075, -0.075)         → old_component__25__1
    → Rev 30:   xyz=(0.02045, 0.015, 0.0)             axis=(0 1 0)   → giabut_1
    → Rigid 32: xyz=(0.0, 0.01225, -0.01)             → but_1
    → Rigid 33: xyz=(0.0, 0.0, -0.045)                → bibut_1 (end-effector)

Rotation convention: axis=(ax,ay,az) with angle q means
  The joint rotates q radians about the specified axis vector.
  axis=(0,0,-1) → Rz(-q)
  axis=(0,1,0)  → Ry(q)
"""

import math
from typing import Tuple

JOINT_LIMITS_LOW  = (0.0, 0.5236, 0.0, 0.0, 0.0, 0.0)
JOINT_LIMITS_HIGH = (3.14159, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159)

# ── helpers ──────────────────────────────────────────────────────────────────

def _I():
    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

def _T(x,y,z):
    return [[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]]

def _Rz(t):
    c,s=math.cos(t),math.sin(t)
    return [[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]]

def _Ry(t):
    c,s=math.cos(t),math.sin(t)
    return [[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]]

def _mul(A,B):
    R=[[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                R[i][j]+=A[i][k]*B[k][j]
    return R

def _chain(*ms):
    r=_I()
    for m in ms: r=_mul(r,m)
    return r

def _pos(M): return (M[0][3],M[1][3],M[2][3])

# ── FK ───────────────────────────────────────────────────────────────────────

def fk(q) -> Tuple[float,float,float]:
    """
    Compute end-effector (bibut_1) position from base_link.
    q: 6 angles in radians [Rev20, Rev22, Rev23, Rev26, Rev28, Rev30]
    Returns (x, y, z) in base_link frame.
    """
    if len(q)!=6: raise ValueError(f"Expected 6 joint angles, got {len(q)}")
    
    # Map input from positive agent space [0, pi] down to internal URDF space [-pi/2, pi/2]
    offsets = [1.570796, 1.570796, 1.570796, 3.141592, 1.570796, 1.570796]
    q_int = [q[i] - offsets[i] for i in range(6)]

    # Fixed: base_link → old_component__6__1
    T_r6  = _T(-0.046528, 0.031724, 0.748891)
    # Fixed: → old_component__14__1
    T_r18 = _T(-0.093, 0.0, -0.01)
    # Fixed: → old_component__15__1
    T_r19 = _T(0.04889, -0.028138, -0.00625)

    # Rev 20: axis=(0, 0, -1) → rotation = Rz(-q[0])
    T_j20 = _chain(_T(-0.034687, -0.0039, -0.0162), _Rz(-q_int[0]))

    # Fixed: → old_component__17__1
    T_r21 = _T(-0.048931, -0.007, -0.033724)

    # Rev 22: axis=(0, -1, 0) → rotation = Ry(-q[1])
    T_j22 = _chain(_T(0.034687, -0.0192, -0.0039), _Ry(-q_int[1]))

    # Rev 23: axis=(0, 1, 0) → rotation = Ry(q[2])
    T_j23 = _chain(_T(0.0, 0.0, -0.155), _Ry(q_int[2]))

    # Fixed: → old_component__20__1
    T_r24 = _T(-0.0039, 0.0192, -0.034687)
    # Fixed: → old_component__21__1
    T_r25 = _T(0.03375, 0.0362, -0.042816)

    # Rev 26: axis=(0, 0, 1) → rotation = Rz(q[3])
    T_j26 = _chain(_T(0.0, -0.00995, -0.0148), _Rz(q_int[3]))

    # Fixed: → old_component__23__1
    T_r27 = _T(0.0152, -0.023, -0.0425)

    # Rev 28: axis=(0, -1, 0) → rotation = Ry(-q[4])
    T_j28 = _chain(_T(-0.00995, -0.0148, 0.0), _Ry(-q_int[4]))

    # Fixed: → old_component__25__1
    T_r29 = _T(-0.0152, 0.0075, -0.075)

    # Rev 30: axis=(0, 1, 0) → rotation = Ry(q[5])
    T_j30 = _chain(_T(0.02045, 0.015, 0.0), _Ry(q_int[5]))

    # Fixed: → but_1
    T_r32 = _T(0.0, 0.01225, -0.01)
    # Fixed: → bibut_1
    T_r33 = _T(0.0, 0.0, -0.045)

    T_ee = _chain(
        T_r6, T_r18, T_r19, T_j20, T_r21, T_j22, T_j23,
        T_r24, T_r25, T_j26, T_r27, T_j28, T_r29, T_j30,
        T_r32, T_r33
    )
    return _pos(T_ee)

def test_fk():
    import sys
    home = fk([0,0,0,0,0,0])
    print(f"Home (all 0s): x={home[0]:.4f}, y={home[1]:.4f}, z={home[2]:.4f}")
    # Test a few angles
    for label, angles in [
        ("J1=40°", [0.6981, 0, 0, 0, 0, 0]),
        ("J2=40°", [0, 0.6981, 0, 0, 0, 0]),
        ("J6=40°", [0, 0, 0, 0, 0, 0.6981]),
    ]:
        pos = fk(angles)
        print(f"{label}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")

if __name__=="__main__":
    test_fk()
