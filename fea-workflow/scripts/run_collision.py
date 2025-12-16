
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solver.explicit_dynamics import ExplicitDynamicsSolver

def main():
    # Hardcoded config for dynamics
    cfg = {
        'geometry': {'Lx': 1.0, 'Ly': 1.0, 'H': 0.1},
        'material': {'E': 1.0, 'nu': 0.3} # GPa / Normalized
    }
    
    solver = ExplicitDynamicsSolver(cfg)
    
    print("Running Explicit Dynamics Collision (1000 steps)...")
    trajectory = solver.run(steps=1000)
    print(f"Simulation Complete. Frames: {len(trajectory)}")
    
    # Visualization: Plot Initial vs Final state
    x_init = trajectory[0]
    x_final = trajectory[-1]
    
    # Scatter plot of nodes
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x_init[:,0], x_init[:,1], x_init[:,2], c='b', s=1)
    ax.set_title("Frame 0 (t=0)")
    ax.set_zlim(-0.5, 0.5)
    
    ax2 = fig.add_subplot(122, projection='3d')
    # Color by z to see layers?
    # Or collision deformation
    ax2.scatter(x_final[:,0], x_final[:,1], x_final[:,2], c='r', s=1)
    ax2.set_title("Frame End (t=1.0s)")
    ax2.set_zlim(-0.5, 0.5)
    
    plt.savefig(os.path.join(os.path.dirname(__file__), "collision_result.png"))
    print("Saved collision_result.png")
    
    # Save trajectory for potential animation
    np.save(os.path.join(os.path.dirname(__file__), "trajectory.npy"), np.array(trajectory))
    print("Saved trajectory.npy")

if __name__ == "__main__":
    main()
