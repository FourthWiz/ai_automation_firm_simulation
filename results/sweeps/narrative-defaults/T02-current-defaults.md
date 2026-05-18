# T02 — Current-defaults narrative gap

Generated: 2026-05-18T09:49:35.566500


## T_review = inf

### Mean ± std cum_pi by strategy × hiring_mode

| strategy | enable_hiring | enable_replenish_hiring | off |
|---|---|---|---|
| all_A | 5860.1±271.1 | 5860.1±271.1 | 5860.1±271.1 |
| all_H | 4957.2±244.4 | 4957.2±244.4 | 4957.2±244.4 |
| all_T | 4521.9±125.2 | 4521.9±125.2 | 4521.9±125.2 |
| greedy_with_switching | 4205.5±552.4 | 4205.5±552.4 | 4205.5±552.4 |
| horizon_brute | 4521.9±125.2 | 4521.9±125.2 | 4521.9±125.2 |
| horizon_optimizer | 5860.1±271.1 | 5860.1±271.1 | 5860.1±271.1 |


### Narrative verdicts at T_review=inf
- N1: FAIL — FAIL by construction — T_review=inf disables firing path
- N2: FAIL — horizon_brute: hire=4509.3 <= off=4509.3 ✗; horizon_optimizer: hire=5766.8 <= off=5766.8 ✗
- N3: FAIL — replenish > hire at 0/5 seeds (threshold 6); median gap=0.0

## T_review = 10.0

### Mean ± std cum_pi by strategy × hiring_mode

| strategy | enable_hiring | enable_replenish_hiring | off |
|---|---|---|---|
| all_A | 6791.0±218.7 | 7165.9±227.6 | 6791.0±218.7 |
| all_H | 5376.0±117.4 | 5444.9±118.4 | 5376.0±117.4 |
| all_T | 2524.6±121.6 | 625.8±128.2 | 2524.6±121.6 |
| greedy_with_switching | 4450.7±345.2 | 2715.6±586.1 | 4450.7±345.2 |
| horizon_brute | 6791.0±218.7 | 7165.9±227.6 | 6791.0±218.7 |
| horizon_optimizer | 6791.0±218.7 | 5382.5±475.6 | 6791.0±218.7 |


### Narrative verdicts at T_review=10.0
- N1: FAIL — horizon_optimizer < greedy AND all_H at 0/5 seeds
- N2: FAIL — horizon_brute: hire=6765.6 <= off=6765.6 ✗; horizon_optimizer: hire=6765.6 <= off=6765.6 ✗
- N3: FAIL — replenish > hire at 0/5 seeds (threshold 6); median gap=-745.0