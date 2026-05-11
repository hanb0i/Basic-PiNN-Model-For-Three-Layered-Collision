# PINN Generalization Study

## One-layer random samples, n=50

| Metric | Value |
|---|---:|
| Top uz MAE mean (%) | 2.987 |
| Top uz MAE worst (%) | 7.255 |
| Volume MAE mean (%) | 1.569 |
| Volume MAE worst (%) | 2.740 |
| Avg displacement relative error mean (%) | 12.902 |

| Representative case | Top uz MAE (%) | Volume MAE (%) | Peak FEM uz | Peak PINN uz | Parameters |
|---|---:|---:|---:|---:|---|
| one_layer_random_029 | 1.012 | 0.915 | -1.7436 | -1.73323 | E=1.873, t=0.09224 |
| one_layer_random_033 | 2.618 | 1.449 | -3.29451 | -3.38981 | E=2.252, t=0.06662 |
| one_layer_random_041 | 7.255 | 2.740 | -2.46867 | -3.05444 | E=5.613, t=0.05155 |
| one_layer_random_021 | 1.017 | 0.918 | -1.85051 | -1.84225 | E=2.208, t=0.08441 |
| one_layer_random_049 | 1.103 | 0.946 | -1.70229 | -1.70073 | E=2.479, t=0.08335 |

## Three-layer random samples, n=50

| Metric | Value |
|---|---:|
| Top uz MAE mean (%) | 12.107 |
| Top uz MAE worst (%) | 16.520 |
| Volume MAE mean (%) | 5.129 |
| Volume MAE worst (%) | 7.173 |
| Avg displacement relative error mean (%) | 59.301 |

| Representative case | Top uz MAE (%) | Volume MAE (%) | Peak FEM uz | Peak PINN uz | Parameters |
|---|---:|---:|---:|---:|---|
| random_interior_021 | 7.476 | 3.410 | -0.154755 | -0.124722 | E1=1.313, E2=7.527, E3=6.896, t1=0.02864, t2=0.03072, t3=0.0998 |
| random_interior_029 | 12.353 | 5.194 | -0.126704 | -0.0501265 | E1=7.395, E2=4.868, E3=2.265, t1=0.0695, t2=0.09291, t3=0.023 |
| random_interior_046 | 16.520 | 7.173 | -0.405858 | -0.0930444 | E1=1.201, E2=9.348, E3=7.196, t1=0.06322, t2=0.02582, t3=0.05095 |
| random_interior_026 | 8.521 | 3.832 | -0.0937298 | -0.0657773 | E1=3.271, E2=9.622, E3=5.739, t1=0.03086, t2=0.06719, t3=0.09573 |
| random_interior_047 | 9.720 | 4.095 | -0.0639594 | -0.0335393 | E1=9.803, E2=6.7, E3=4.809, t1=0.09902, t2=0.05769, t3=0.07711 |
