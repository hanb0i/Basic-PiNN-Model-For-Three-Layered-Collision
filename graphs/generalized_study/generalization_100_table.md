# PINN Generalization Study

## One-layer random samples, n=100

| Metric | Value |
|---|---:|
| Top uz MAE mean (%) | 2.939 |
| Top uz MAE worst (%) | 7.543 |
| Volume MAE mean (%) | 1.563 |
| Volume MAE worst (%) | 2.868 |
| Avg displacement relative error mean (%) | 12.626 |

| Representative case | Top uz MAE (%) | Volume MAE (%) | Peak FEM uz | Peak PINN uz | Parameters |
|---|---:|---:|---:|---:|---|
| one_layer_random_056 | 0.799 | 0.866 | -3.10635 | -3.04659 | E=1.364, t=0.08321 |
| one_layer_random_078 | 2.549 | 1.679 | -0.326779 | -0.328982 | E=3.271, t=0.1458 |
| one_layer_random_089 | 7.543 | 2.868 | -1.36302 | -1.66844 | E=9.203, t=0.05375 |
| one_layer_random_085 | 0.931 | 0.920 | -2.51134 | -2.46822 | E=1.21, t=0.09491 |
| one_layer_random_057 | 0.989 | 0.913 | -2.01488 | -2.00168 | E=2.074, t=0.08368 |

## Three-layer random samples, n=100

| Metric | Value |
|---|---:|
| Top uz MAE mean (%) | 12.151 |
| Top uz MAE worst (%) | 16.520 |
| Volume MAE mean (%) | 5.186 |
| Volume MAE worst (%) | 7.173 |
| Avg displacement relative error mean (%) | 59.174 |

| Representative case | Top uz MAE (%) | Volume MAE (%) | Peak FEM uz | Peak PINN uz | Parameters |
|---|---:|---:|---:|---:|---|
| random_interior_021 | 7.476 | 3.410 | -0.154755 | -0.124722 | E1=1.313, E2=7.527, E3=6.896, t1=0.02864, t2=0.03072, t3=0.0998 |
| random_interior_029 | 12.353 | 5.194 | -0.126704 | -0.0501265 | E1=7.395, E2=4.868, E3=2.265, t1=0.0695, t2=0.09291, t3=0.023 |
| random_interior_046 | 16.520 | 7.173 | -0.405858 | -0.0930444 | E1=1.201, E2=9.348, E3=7.196, t1=0.06322, t2=0.02582, t3=0.05095 |
| random_interior_026 | 8.521 | 3.832 | -0.0937298 | -0.0657773 | E1=3.271, E2=9.622, E3=5.739, t1=0.03086, t2=0.06719, t3=0.09573 |
| random_interior_097 | 8.874 | 4.045 | -0.197164 | -0.133047 | E1=6.333, E2=6.685, E3=2.026, t1=0.03119, t2=0.04981, t3=0.09817 |
