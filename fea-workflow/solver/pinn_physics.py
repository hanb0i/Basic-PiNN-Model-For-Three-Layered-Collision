
import torch
import torch.autograd as autograd

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

def gradient(u, x):
    grad_u = torch.zeros(x.shape[0], 3, 3, device=x.device)
    for i in range(3):
        u_i = u[:, i].unsqueeze(1)
        grad_i = autograd.grad(u_i, x, grad_outputs=torch.ones_like(u_i), create_graph=True, retain_graph=True)[0]
        grad_u[:, i, :] = grad_i
    return grad_u

def strain(grad_u):
    return 0.5 * (grad_u + grad_u.transpose(1, 2))

def stress(eps, lm, mu):
    trace_eps = torch.einsum('bii->b', eps).unsqueeze(1).unsqueeze(2)
    eye = torch.eye(3, device=eps.device).unsqueeze(0).repeat(eps.shape[0], 1, 1)
    return lm * trace_eps * eye + 2 * mu * eps

def divergence(sigma, x):
    div = torch.zeros(x.shape[0], 3, device=x.device)
    for i in range(3):
        div_i = 0
        for j in range(3):
            sig_ij = sigma[:, i, j].unsqueeze(1)
            grad_sig_ij = autograd.grad(sig_ij, x, grad_outputs=torch.ones_like(sig_ij), create_graph=True, retain_graph=True)[0]
            div_i += grad_sig_ij[:, j]
        div[:, i] = div_i
    return div

def compute_loss(model, data, device, config):
    total_loss = 0
    losses = {}
    weights = config['pinn']['weights']
    
    # Material Params (Assuming uniform E, nu for now based on config, or we can look up per layer if wanted)
    E = config['material']['E']
    nu = config['material']['nu']
    lm, mu = get_lame_params(E, nu)
    
    # Interior PDE
    pde_loss = 0
    for i in range(3):
        x_int = data['interior'][i].to(device)
        x_int.requires_grad = True
        u = model(x_int, i)
        div_sigma = divergence(stress(strain(gradient(u, x_int)), lm, mu), x_int)
        pde_loss += torch.mean((-div_sigma)**2)
        
    losses['pde'] = pde_loss
    total_loss += weights['pde'] * pde_loss
    
    # BC: Clamped Sides
    bc_loss = 0
    for i in range(3):
        x_s = data['sides'][i].to(device)
        u_s = model(x_s, i)
        bc_loss += torch.mean(u_s**2)
    losses['bc'] = bc_loss
    total_loss += weights['bc'] * bc_loss
    
    # BC: Top Loaded
    x_tl = data['top_load'].to(device)
    x_tl.requires_grad = True
    u_tl = model(x_tl, 2) # Layer 3
    sig_tl = stress(strain(gradient(u_tl, x_tl)), lm, mu)
    T_z = sig_tl[:, :, 2] # n=(0,0,1)
    target = torch.tensor([0.0, 0.0, -config['load_patch']['pressure']], device=device).repeat(x_tl.shape[0], 1)
    load_loss = torch.mean((T_z - target)**2)
    losses['load'] = load_loss
    total_loss += weights['load'] * load_loss
    
    # BC: Top Free
    if len(data['top_free']) > 0:
        x_tf = data['top_free'].to(device)
        x_tf.requires_grad = True
        sig_tf = stress(strain(gradient(model(x_tf, 2), x_tf)), lm, mu)
        free_loss = torch.mean(sig_tf[:, :, 2]**2)
        total_loss += weights['bc'] * free_loss
        
    # BC: Bottom Free
    x_b = data['bottom'].to(device)
    x_b.requires_grad = True
    sig_b = stress(strain(gradient(model(x_b, 0), x_b)), lm, mu)
    bot_loss = torch.mean(sig_b[:, :, 2]**2) # n=(0,0,-1) -> T = -sigma_3j. Square matches.
    total_loss += weights['bc'] * bot_loss
    
    # Interfaces
    # 1-2
    x_if12 = data['if_12'].to(device)
    loss_if1 = torch.mean((model(x_if12, 0) - model(x_if12, 1))**2)
    # 2-3
    x_if23 = data['if_23'].to(device)
    loss_if2 = torch.mean((model(x_if23, 1) - model(x_if23, 2))**2)
    
    losses['interface'] = loss_if1 + loss_if2
    total_loss += weights['interface'] * (loss_if1 + loss_if2)
    
    losses['total'] = total_loss
    return total_loss, losses
