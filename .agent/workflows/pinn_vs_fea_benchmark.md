---
description: Workflow to compare the 3-Layer PINN model against a Finite Element Analysis (FEA) benchmark.
---

This workflow outlines the steps to validate the PINN's accuracy using a standard FEM solution. It does not require building the FEM solver now, but defines the process.

## 1. Define the Benchmark Problem
Ensure the FEA model uses the **exact same configuration** as `config.py`:
- **Geometry**: $L_x=1.0, L_y=1.0, H=0.1$.
- **Mesh**: Use a structured hexahedral mesh or fine tetrahedral mesh.
    - Recommended resolution: At least $50 \times 50 \times 15$ elements to capture the thin gradients.
- **Materials**: 3 Layers with $E=[1,1,1]$ and $\nu=[0.3,0.3,0.3]$ (or as defined in config).
- **Boundary Conditions**:
    - **Sides ($x=0, L_x, y=0, L_y$)**: Fixed ($u=0$).
    - **Bottom ($z=0$)**: Free (Traction=0).
    - **Top ($z=H$)**:
        - Load Patch ($x \in [L_x/3, 2L_x/3], y \in [L_y/3, 2L_y/3]$): Traction $\sigma \cdot n = (0, 0, -p_0)$.
        - Rest: Traction-free.

## 2. Generate FEA Solution
Use a standard solver like **FEniCS** (open-source) or **Ansys/Abaqus** (commercial).
1.  **Setup**: Implement the linear elasticity weak form.
    $$ \int_{\Omega} \sigma(u) : \epsilon(v) dx = \int_{\Gamma_{load}} T \cdot v ds $$
2.  **Solve**: Compute displacement field $u_{FEM}(x,yz)$.
3.  **Export**: Interpolate the FEM solution onto a regular grid matching the PINN's plotting grid.
    - Format: CSV or `.npy` file.
    - Columns: `x, y, z, u_x, u_y, u_z`
    - Grid size: e.g., $100 \times 100 \times 100$ points.

## 3. Extract PINN Predictions
Run the PINN inference on the **same grid** used for the FEA export.
1.  Load trained model: `pinn_model.pth`.
2.  Generate grid points $(x_i, y_i, z_i)$.
3.  Predict $u_{PINN} = \text{Model}(x, y, z)$.
4.  Save as `pinn_results.npy`.

## 4. Quantitative Comparison
Calculate error metrics to quantify the deviation.

### Metrics
- **L2 Relative Error**: $$ \frac{|| u_{PINN} - u_{FEM} ||_2}{|| u_{FEM} ||_2} $$
    - *Target*: $< 1\%$ for high accuracy, $< 5\%$ for acceptable engineering approximation.
- **Max Absolute Error**: $$ \max | u_{PINN} - u_{FEM} | $$
    - Important for safety-critical peak displacement checks.

## 5. Visual Comparison
Generate side-by-side plots:
1.  **Displacement Contours**: Plot $u_z$ at the top surface for both PINN and FEM. Visual patterns should be identical.
2.  **Error Map**: Plot $| u_{PINN} - u_{FEM} |$ contour.
    - Look for "hotspots" of error (often at boundaries or load discontinuities).
3.  **Cross-Section Slice**: Compare deflection profiles at $y=L_y/2$.

## 6. Sensitivity Analysis (Optional)
If errors are high:
- **Refine FEM Mesh**: Ensure the benchmark is actually converged (ground truth).
- **Refine PINN Sampling**: Increase `N_INTERIOR` or weighting `w_pde`.
