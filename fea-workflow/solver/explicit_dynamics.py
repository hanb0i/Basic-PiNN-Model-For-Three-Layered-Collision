
import numpy as np

class ExplicitDynamicsSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.Lx = cfg['geometry']['Lx']
        self.Ly = cfg['geometry']['Ly']
        self.H = cfg['geometry']['H']
        
        # Mesh
        self.ne_x = 10 # Coarser for dynamic speed
        self.ne_y = 10
        self.ne_z = 6  # 2 per layer?
        
        self.dx = self.Lx / self.ne_x
        self.dy = self.Ly / self.ne_y
        self.dz = self.H / self.ne_z
        
        self.nx = self.ne_x + 1
        self.ny = self.ne_y + 1
        self.nz = self.ne_z + 1
        self.n_nodes = self.nx * self.ny * self.nz
        
        # Physics
        self.E = cfg['material']['E'] * 1000 # Scaling for stiffer dynamics? Keep 1.0 for consistency.
        self.nu = cfg['material']['nu']
        self.density = 10.0 # Arbitrary density
        
        # Sim parameters
        self.dt = 0.0001 # Reduced from 0.001 to ensure stability (CFL condition)
        self.total_time = 0.5 # Reduced total time for testing
        
        # State
        self.x = np.zeros((self.n_nodes, 3)) # Positions
        self.v = np.zeros((self.n_nodes, 3)) # Velocities
        self.f = np.zeros((self.n_nodes, 3)) # Forces
        self.mass = np.zeros(self.n_nodes)   # Lumped Mass
        
        # Elements
        self.elements = [] # List of node indices
        
        self._initialize_mesh()
        self._compute_lumped_mass()
        
        # Safety Check
        if np.any(self.mass <= 0):
            raise ValueError("Zero or negative mass detected! Mesh or density error.")
            
        self._set_initial_conditions()
        
    def _get_node_idx(self, i, j, k):
        return i + j*self.nx + k*self.nx*self.ny
        
    def _initialize_mesh(self):
        # 1. Generate Nodes (Grid)
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        z = np.linspace(0, self.H, self.nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.x[:, 0] = X.ravel()
        self.x[:, 1] = Y.ravel()
        self.x[:, 2] = Z.ravel()
        
        # 2. Generate Elements (Hex)
        for k in range(self.ne_z):
            for j in range(self.ne_y):
                for i in range(self.ne_x):
                    # 8 corner nodes
                    n0 = self._get_node_idx(i, j, k)
                    n1 = self._get_node_idx(i+1, j, k)
                    n2 = self._get_node_idx(i+1, j+1, k)
                    n3 = self._get_node_idx(i, j+1, k)
                    n4 = self._get_node_idx(i, j, k+1)
                    n5 = self._get_node_idx(i+1, j, k+1)
                    n6 = self._get_node_idx(i+1, j+1, k+1)
                    n7 = self._get_node_idx(i, j+1, k+1)
                    
                    self.elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
                    
        self.elements = np.array(self.elements, dtype=int)
        print(f"Mesh Initialized: {len(self.elements)} elements")

    def _compute_lumped_mass(self):
        # Mass of one element
        vol = self.dx * self.dy * self.dz
        elem_mass = vol * self.density
        node_share = elem_mass / 8.0
        
        for elem in self.elements:
            for n_idx in elem:
                self.mass[n_idx] += node_share
                
    def _set_initial_conditions(self):
        # Initial downward velocity
        self.v[:, 2] = -5.0 # m/s (Slower than -10 to see bounce in short time)
        
    def _compute_internal_forces(self):
        # Linear Elasticity on Hex Elements (Simplified Co-rotational or Small Strain)
        # For explicit dynamics, true deformation gradient F is best, but costly.
        # We'll use a simplified stiffness matrix approach for "Internal Force = K * u"
        # F_int = sum (Ke * u_element)
        
        # Computing Ke on the fly or pre-stored? 
        # Since mesh is regular: one Ke for all.
        lam = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        
        # ... Reuse Ke generation logic from fem_solver ...
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        Ke = np.zeros((24, 24))
        C_diag = [lam+2*mu, lam+2*mu, lam+2*mu, mu, mu, mu]
        C = np.zeros((6, 6))
        C[0:3, 0:3] = lam
        np.fill_diagonal(C, C_diag)
        
        dx, dy, dz = self.dx, self.dy, self.dz
        
        for r in gp:
            for s in gp:
                for t in gp:
                    invJ = np.diag([2/dx, 2/dy, 2/dz])
                    detJ = dx * dy * dz / 8.0
                    B = np.zeros((6, 24))
                    node_signs = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
                    for i_n in range(8):
                        xi, eta, zeta = node_signs[i_n]
                        dN_dxi = 0.125 * xi * (1 + eta * s) * (1 + zeta * t)
                        dN_deta = 0.125 * eta * (1 + xi * r) * (1 + zeta * t)
                        dN_dzeta = 0.125 * zeta * (1 + xi * r) * (1 + eta * s)
                        d_global = invJ @ np.array([dN_dxi, dN_deta, dN_dzeta])
                        nx_val, ny_val, nz_val = d_global
                        col = 3 * i_n
                        B[0, col] = nx_val; B[1, col+1] = ny_val; B[2, col+2] = nz_val
                        B[3, col+1] = nz_val; B[3, col+2] = ny_val
                        B[4, col] = nz_val; B[4, col+2] = nx_val
                        B[5, col] = ny_val; B[5, col+1] = nx_val
                    Ke += B.T @ C @ B * detJ
        
        # Initial positions X0
        # u = x_current - X_initial
        # To avoid storing X0 for all, we can assume regular grid X0 logic
        # But simpler: assume small strain, F = K * (x - x0)
        
        # Vectorized accumulation?
        # Loop over elements is slow in python.
        # But for 600 elements (10x10x6), it's fine.
        
        # Let's reconstruct x0 for an element on the fly
        # Or store x0 globally? We initialized self.x, let's copy it as x0
        if not hasattr(self, 'x0'):
            self.x0 = self.x.copy()

        # Displacement
        u = self.x - self.x0 
        
        # This loop is the bottleneck.
        # Optimized:
        # u_elem shape: (n_elem, 24)
        # nodes shape: (n_elem, 8)
        
        nodes = self.elements
        # Gather displacements: (n_elem, 8, 3) -> (n_elem, 24)
        u_nodes = u[nodes] # (n_elem, 8, 3)
        u_flat = u_nodes.reshape(len(nodes), 24)
        
        # F_elem = u_flat @ Ke.T ? Ke is symmetric. Ke @ u
        # Ke (24, 24). u_flat (N, 24).
        # F_elem (N, 24)
        
        f_elem = u_flat @ Ke.T # (N, 24)
        
        # Scatter add to global force
        # Slow in pure python loop?
        # np.add.at is faster
        
        f_flat = f_elem.reshape(len(nodes), 8, 3)
        
        self.f[:] = 0 # Reset forces
        
        # Scatter
        # We need to flatten indices and values
        np.add.at(self.f, nodes, -f_flat) # Internal force resists displacement -> -K*u
        
    def _compute_external_forces_collision(self):
        # Gravity
        self.f[:, 2] -= 9.81 * self.mass
        
        # Ground Penalty (z < 0)
        # If z < 0, F_pen = k * |z|
        k_pen = 1e4 # Stiffness
        
        penetration_mask = self.x[:, 2] < 0
        penetration_depth = -self.x[penetration_mask, 2]
        
        self.f[penetration_mask, 2] += k_pen * penetration_depth

    def step(self):
        # 1. Internal Forces
        self._compute_internal_forces()
        
        # 2. External (Collision)
        self._compute_external_forces_collision()
        
        # 3. Integration (Semi-implicit Euler or Velocity Verlet)
        # a = F / m
        a = self.f / self.mass[:, None]
        
        # v += a * dt
        self.v += a * self.dt
        
        # x += v * dt
        self.x += self.v * self.dt
        
        return self.x.copy()

    def run(self, steps=100):
        trajectory = []
        for i in range(steps):
            x_curr = self.step()
            if i % 5 == 0:
                trajectory.append(x_curr)
        return trajectory
