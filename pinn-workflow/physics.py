
import torch
import torch.autograd as autograd
import pinn_config as config

def gradient(u, x):
    # u: (N, 3), x: (N, 3)
    # Returns du/dx: (N, 3, 3)
    # [ [dux/dx, dux/dy, dux/dz],
    #   [duy/dx, duy/dy, duy/dz],
    #   [duz/dx, duz/dy, duz/dz] ]
    
    grad_u = torch.zeros(x.shape[0], 3, 3, device=x.device)
    
    for i in range(3): # u_x, u_y, u_z
        u_i = u[:, i].unsqueeze(1)
        grad_i = autograd.grad(
            u_i, x, 
            grad_outputs=torch.ones_like(u_i),
            create_graph=True, 
            retain_graph=True
        )[0]
        grad_u[:, i, :] = grad_i
        
    return grad_u

def strain(grad_u):
    # epsilon = 0.5 * (grad_u + grad_u^T)
    return 0.5 * (grad_u + grad_u.transpose(1, 2))

def stress(eps, lm, mu):
    # sigma = lambda * tr(eps) * I + 2 * mu * eps
    trace_eps = torch.einsum('bii->b', eps).unsqueeze(1).unsqueeze(2) # (N, 1, 1)
    eye = torch.eye(3, device=eps.device).unsqueeze(0).repeat(eps.shape[0], 1, 1)
    
    sigma = lm * trace_eps * eye + 2 * mu * eps
    return sigma

def divergence(sigma, x):
    # sigma: (N, 3, 3), x: (N, 3)
    # div_sigma: (N, 3) vector
    # We need d(sigma_ij)/dx_j
    
    div = torch.zeros(x.shape[0], 3, device=x.device)
    
    # Row 0: d(sig_xx)/dx + d(sig_xy)/dy + d(sig_xz)/dz
    # etc.
    
    for i in range(3): # For each component of force equilibrium
        # We need d(sigma_i0)/dx + d(sigma_i1)/dy + d(sigma_i2)/dz
        div_i = 0
        for j in range(3):
            sig_ij = sigma[:, i, j].unsqueeze(1)
            grad_sig_ij = autograd.grad(
                sig_ij, x,
                grad_outputs=torch.ones_like(sig_ij),
                create_graph=True,
                retain_graph=True
            )[0]
            div_i += grad_sig_ij[:, j]
        div[:, i] = div_i
        
    return div

def compute_loss(model, data, device):
    total_loss = 0
    losses = {}
    
    # --- 1. PDE Residuals (Interior) ---
    pde_loss = 0
    for i in range(3): # For each layer
        x_int = data['interior'][i].to(device)
        x_int.requires_grad = True
        
        lm, mu = config.Lame_Params[i]
        
        u = model(x_int, i)
        grad_u = gradient(u, x_int)
        eps = strain(grad_u)
        sig = stress(eps, lm, mu)
        div_sigma = divergence(sig, x_int)
        
        # Equilibrium: -div(sigma) = 0
        residual = -div_sigma
        
        loss_i = torch.mean(residual**2)
        pde_loss += loss_i
        
    losses['pde'] = pde_loss
    total_loss += config.WEIGHTS['pde'] * pde_loss
    
    # --- 2. Dirichlet BCs (Clamped Sides) ---
    bc_loss = 0
    for i in range(3):
        x_side = data['sides'][i].to(device)
        u_side = model(x_side, i)
        # u = 0
        loss_side = torch.mean(u_side**2)
        bc_loss += loss_side
        
    losses['bc_sides'] = bc_loss
    total_loss += config.WEIGHTS['bc'] * bc_loss
    
    # --- 3. Traction BCs (Top & Bottom) ---
    # Top Loaded
    x_top_load = data['top_load'].to(device)
    x_top_load.requires_grad = True
    
    lm3, mu3 = config.Lame_Params[2] # Layer 3
    u_top = model(x_top_load, 2)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm3, mu3)
    
    # n = (0, 0, 1)
    # Traction T = sigma * n
    # T_z = sigma_zz * 1 + ...
    # n = [0, 0, 1]^T
    # T = [sigma_x3, sigma_y3, sigma_z3]^T
    T = sig_top[:, :, 2] 
    
    # Target: (0, 0, -p0)
    target = torch.tensor([0.0, 0.0, -config.p0], device=device).repeat(x_top_load.shape[0], 1)
    
    loss_load = torch.mean((T - target)**2)
    losses['load'] = loss_load
    total_loss += config.WEIGHTS['load'] * loss_load
    
    # Top Free
    x_top_free = data['top_free'].to(device)
    x_top_free.requires_grad = True
    u_top_free = model(x_top_free, 2)
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm3, mu3)
    T_free = sig_top_free[:, :, 2]
    
    loss_free = torch.mean(T_free**2)
    losses['free_top'] = loss_free
    total_loss += config.WEIGHTS['bc'] * loss_free # Use BC weight
    
    # Bottom Free (Layer 1)
    x_bot = data['bottom'].to(device)
    x_bot.requires_grad = True
    
    lm1, mu1 = config.Lame_Params[0]
    u_bot = model(x_bot, 0)
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm1, mu1)
    
    # n = (0, 0, -1) -> T = sigma * n = - sigma_3j
    # We want T = 0 -> sigma_3j = 0
    T_bot = -sig_bot[:, :, 2]
    
    loss_bot = torch.mean(T_bot**2)
    losses['free_bot'] = loss_bot
    total_loss += config.WEIGHTS['bc'] * loss_bot
    
    # --- 4. Interface Continuity (u matching) ---
    # Layer 1-2
    x_if12 = data['if_12'].to(device)
    x_if12.requires_grad = True
    u1_if = model(x_if12, 0)
    u2_if = model(x_if12, 1)
    
    loss_if12 = torch.mean((u1_if - u2_if)**2)
    
    # Layer 2-3
    x_if23 = data['if_23'].to(device)
    x_if23.requires_grad = True
    u2_if_23 = model(x_if23, 1)
    u3_if = model(x_if23, 2)
    
    loss_if23 = torch.mean((u2_if_23 - u3_if)**2)
    
    losses['interface'] = loss_if12 + loss_if23
    total_loss += config.WEIGHTS['interface_u'] * (loss_if12 + loss_if23)

    # --- 5. Interface Stress Continuity (Force Transmission) ---
    # We need sigma_z continuity: sigma_xz, sigma_yz, sigma_zz must match
    # Layer 1-2 Interface
    # Layer 1 side
    lm1, mu1 = config.Lame_Params[0]
    grad_u1_if = gradient(u1_if, x_if12)
    sig1_if = stress(strain(grad_u1_if), lm1, mu1)
    # Layer 2 side
    lm2, mu2 = config.Lame_Params[1]
    grad_u2_if_12 = gradient(u2_if, x_if12)
    sig2_if = stress(strain(grad_u2_if_12), lm2, mu2)
    
    # Traction vector on z-plane is simply the 3rd column of stress tensor
    traction1_12 = sig1_if[:, :, 2] # (N, 3)
    traction2_12 = sig2_if[:, :, 2]
    
    loss_stress_12 = torch.mean((traction1_12 - traction2_12)**2)
    
    # Layer 2-3 Interface
    # Layer 2 side
    # u2_if_23 already computed
    grad_u2_if_23 = gradient(u2_if_23, x_if23)
    sig2_if_23 = stress(strain(grad_u2_if_23), lm2, mu2)
    
    # Layer 3 side
    lm3, mu3 = config.Lame_Params[2]
    grad_u3_if = gradient(u3_if, x_if23)
    sig3_if_23 = stress(strain(grad_u3_if), lm3, mu3)
    
    traction2_23 = sig2_if_23[:, :, 2]
    traction3_23 = sig3_if_23[:, :, 2]
    
    loss_stress_23 = torch.mean((traction2_23 - traction3_23)**2)
    
    losses['interface_stress'] = loss_stress_12 + loss_stress_23
    total_loss += config.WEIGHTS['interface_stress'] * (loss_stress_12 + loss_stress_23)
    
    losses['total'] = total_loss
    return total_loss, losses
