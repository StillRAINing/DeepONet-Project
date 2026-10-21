# E4 follow-up `ff6cons`: test and OOD (evaluated after the selection rule; not a re-selection)

Generated once by `scripts/final_eval.py`. Mean +- sample std over seeds 3001, 1, 2. Headline error: per-sample relative L2, channels scaled by the train std. `q2`: median own-scale relative L2 of q2. `consistency`: median rms(dq/dt - qdot) / rms(true qdot). `energy`: median excess energy-balance residual over the RK4 floor, as a fraction of the trajectory's energy range (test only).

## test (n = 1500)

| model | input | median | p95 | max | under 5 % | under 10 % | q2 | consistency | energy |
|---|---|---|---|---|---|---|---|---|---|
| mlp | samples | 0.0662 +- 0.0019 | 0.5865 +- 0.0044 | 1.3824 +- 0.2280 | 41.4 +- 0.8 % | 59.2 +- 0.3 % | 0.1342 +- 0.0015 | 0.1164 +- 0.0007 | 0.1518 +- 0.0020 |
| deeponet | samples | 0.1093 +- 0.0015 | 0.6136 +- 0.0017 | 1.8709 +- 0.0562 | 24.9 +- 0.4 % | 47.3 +- 0.5 % | 0.1847 +- 0.0051 | 0.0078 +- 0.0008 | 0.1948 +- 0.0052 |

## ood (n = 200)

| model | input | median | p95 | max | under 5 % | under 10 % | q2 | consistency |
|---|---|---|---|---|---|---|---|---|
| mlp | samples | 0.0497 +- 0.0017 | 0.1129 +- 0.0081 | 0.2119 +- 0.0268 | 50.3 +- 2.5 % | 92.3 +- 1.5 % | 0.1523 +- 0.0049 | 0.1623 +- 0.0080 |
| deeponet | samples | 0.0579 +- 0.0022 | 0.1528 +- 0.0034 | 0.3830 +- 0.0530 | 39.8 +- 2.9 % | 82.7 +- 2.4 % | 0.1721 +- 0.0068 | 0.0083 +- 0.0008 |

## ood_amp1 (n = 200)

| model | input | median | p95 | max | under 5 % | under 10 % | q2 | consistency |
|---|---|---|---|---|---|---|---|---|
| mlp | samples | 0.5652 +- 0.0109 | 0.9776 +- 0.0160 | 1.1900 +- 0.0090 | 0.0 +- 0.0 % | 1.0 +- 0.9 % | 0.5502 +- 0.0097 | 0.2866 +- 0.0122 |
| deeponet | samples | 0.5970 +- 0.0051 | 1.0266 +- 0.0422 | 1.4057 +- 0.2418 | 0.0 +- 0.0 % | 0.0 +- 0.0 % | 0.6165 +- 0.0238 | 0.0084 +- 0.0004 |

## ood_amp2 (n = 200)

| model | input | median | p95 | max | under 5 % | under 10 % | q2 | consistency |
|---|---|---|---|---|---|---|---|---|
| mlp | samples | 0.8019 +- 0.0092 | 1.0128 +- 0.0174 | 1.2717 +- 0.0594 | 0.0 +- 0.0 % | 0.0 +- 0.0 % | 0.7333 +- 0.0158 | 0.3596 +- 0.0063 |
| deeponet | samples | 0.8230 +- 0.0058 | 1.0953 +- 0.0529 | 1.4927 +- 0.0465 | 0.0 +- 0.0 % | 0.0 +- 0.0 % | 0.7213 +- 0.0077 | 0.0077 +- 0.0007 |

## ood_smooth (n = 200)

| model | input | median | p95 | max | under 5 % | under 10 % | q2 | consistency |
|---|---|---|---|---|---|---|---|---|
| mlp | samples | 0.1194 +- 0.0042 | 0.7072 +- 0.0295 | 0.9407 +- 0.0231 | 14.0 +- 0.5 % | 42.5 +- 0.5 % | 0.2746 +- 0.0119 | 0.1403 +- 0.0042 |
| deeponet | samples | 0.1372 +- 0.0020 | 0.7303 +- 0.0278 | 1.0923 +- 0.1155 | 9.0 +- 0.5 % | 33.8 +- 1.5 % | 0.2887 +- 0.0080 | 0.0081 +- 0.0009 |
