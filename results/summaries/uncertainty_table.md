### Acrobot-v1

| Variant | Final return (IQM, 95% CI) | AUC (IQM, 95% CI) | P(beats ε-greedy (ensemble)) | seeds |
|---|---|---|---|---|
| ε→Boltzmann (ensemble) | -84.1 [-92.1, -77.2] | -275.6 [-338.8, -231.4] | 0.56 [0.00, 1.00] | 3 |
| ε-greedy (ensemble) | -93.2 [-117.7, -80.3] | -267.8 [-339.8, -147.4] | (baseline) | 3 |
| ε-greedy (1 head) | -105.4 [-138.3, -81.1] | -253.7 [-316.8, -197.1] | 0.33 [0.00, 0.89] | 3 |
| uncertainty-gated (ensemble) | -385.8 [-500.0, -290.1] | -398.5 [-484.0, -330.5] | 0.00 [0.00, 0.00] | 3 |
| uncertainty-gated (TD error) | -398.5 [-500.0, -335.9] | -371.8 [-465.3, -294.3] | 0.00 [0.00, 0.00] | 3 |
