# Finite Element Analysis (FEA) Workflow — 3D Thin Elastic Layer

This folder mirrors the thin-layer PINN project with a practical FEA plan: thin rectangular box, isotropic linear elasticity, clamped side faces, uniform downward traction on the central 1/9 top patch, rest traction-free.

## Layout

- `configs/` — shared problem inputs (geometry, material, loads, solver settings).
- `geometry/` — CAD or parametric geometry definitions and named sets.
- `mesh/` — meshing strategies, refinement notes, mesh artifacts.
- `materials/` — material property tables and unit conventions.
- `boundary_conditions/` — clamped sides, load patch, and free-top definitions.
- `solver/` — static linear solve settings (direct/iterative, tolerances).
- `postprocessing/` — field outputs, reaction summaries, convergence plots.
- `scripts/` — automation hooks (export/import, batch runs, checks).

## Edited Prompt (FEA workflow)

Below is a practical **Finite Element Analysis (FEA) workflow** for the *same 3D thin elastic layer problem* (thin rectangular box, isotropic linear elasticity, clamped side faces, uniform downward traction on the central 1/9 top patch, rest traction-free).

### 1) Define the problem (inputs + outputs)
- Geometry: `(0 < x < Lx, 0 < y < Ly, 0 < z < H)` with `H << Lx, Ly`.
- Material: linear elastic, isotropic with `(E, ν)` (or Lamé `(λ, μ)`).
- BCs: clamped on `x=0,Lx` and `y=0,Ly`; traction load `(0,0,-p0)` on central top patch; traction-free elsewhere on top.
- Quantities: max/min `uz` on top, stress/strain fields (von Mises, `σzz`), reaction forces on clamps, stress near patch edges.

### 2) Choose the modeling approach
- Use 3D solid linear elasticity (static equilibrium `-∇·σ(u)=0`).
- Decide on full 3D vs. symmetry reduction (e.g., quarter model if geometry/loading symmetric).

### 3) Build CAD / geometry + named boundaries
- Create rectangular box `(Lx, Ly, H)`.
- On top face `z=H`, define central square patch: `Lx/3 ≤ x ≤ 2Lx/3`, `Ly/3 ≤ y ≤ 2Ly/3` (area = `1/9 * Lx * Ly`).
- Name sets: `side_clamp`, `top_load_patch`, `top_free`, (optional) `bottom`.

### 4) Mesh strategy (make-or-break)
- 3D elements (tet/hex; hex/wedge preferred if structured).
- Through-thickness: at least 3–8 elements across `H`.
- Local refinement near load-patch edges and clamped edges/corners; maintain element quality.

### 5) Apply materials + units
- Assign `(E, ν)` (or `(λ, μ)`) with consistent units for `Lx, Ly, H, p0, E`.

### 6) Apply boundary conditions and loads
- Clamp sides: `ux=uy=uz=0` on `x=0,Lx` and `y=0,Ly`.
- Traction on patch: uniform pressure/traction `-p0` in `z` on `top_load_patch`.
- Traction-free elsewhere on top (`top_free`).
- Add reaction-force check: total reactions on clamps ≈ `p0 * (1/9) * Lx * Ly`.

### 7) Solve (static linear)
- Analysis: static, linear (small strain).
- Solver: direct sparse if available; tolerances tight enough for mesh studies.

### 8) Post-process results
- Visuals: displacement magnitude and `uz` on top; `σzz`, von Mises, principal stresses.
- Numbers: `max uz` location, peak stresses near patch boundary, total reaction forces, symmetry checks (if reduced model).

### 9) Verification and mesh convergence
- Patch-test sanity (no rigid modes with clamps).
- Load balance: reactions ≈ applied load.
- Mesh study (global + local refinements): track `max uz` convergence; stresses may converge slower (use averaged metrics).
- If stiffness issues through thickness, increase elements across `H`.

### 10) Optional extensions (mirror PINN remarks)
- Nondimensionalize inputs; symmetry models to cut runtime.
- Parameter sweeps over `H, p0, E, ν, patch size`; export response curves (`p0` vs `max uz`).
