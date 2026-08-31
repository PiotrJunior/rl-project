### Acrobot-v1

| Variant | Final return (IQM, 95% CI) | AUC (IQM, 95% CI) | P(beats ε-greedy (ensemble)) | seeds |
|---|---|---|---|---|
| ε→Boltzmann (ensemble) | -82.9 [-88.7, -77.4] | -265.8 [-319.6, -232.3] | 0.67 [0.11, 1.00] | 3 |
| ε-greedy (ensemble) | -106.5 [-156.1, -80.5] | -282.0 [-294.2, -266.2] | (baseline) | 3 |
| ε-greedy (1 head) | -150.9 [-229.5, -86.5] | -256.1 [-323.0, -191.4] | 0.22 [0.00, 0.67] | 3 |
| uncertainty-gated (TD error) | -330.4 [-394.6, -222.0] | -404.3 [-458.4, -358.7] | 0.00 [0.00, 0.00] | 3 |
| uncertainty-gated (ensemble) | -346.6 [-500.0, -160.9] | -365.7 [-396.1, -315.6] | 0.00 [0.00, 0.00] | 3 |
