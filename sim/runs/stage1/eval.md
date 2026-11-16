| Depth | PPO success | Random success | PPO steps | A* steps | PPO / A* | Blocked per episode |
|---|---|---|---|---|---|---|
| 1 | 100% | 12% | 1.0 | 1.0 | 1.0 | 0.0 |
| 2 | 100% | 0% | 2.0 | 2.0 | 1.0 | 0.0 |
| 4 | 100% | 0% | 3.6 | 3.6 | 1.0 | 0.0 |
| 8 | 98% | 0% | 6.776 | 6.8 | 1.0 | 0.24 |
| 12 | 100% | 0% | 9.24 | 8.818 | 1.0 | 0.0 |
| 16 | 96% | 0% | 11.938 | 9.467 | 1.0 | 1.3 |
| 20 | 98% | 0% | 13.143 | 9.6 | 1.0 | 0.86 |
| 25 | 100% | 0% | 14.46 | 10.667 | 1.0 | 0.1 |
| 30 | 100% | 0% | 15.7 | 9.0 | 1.062 | 0.18 |

| Target from the stick | Success | Start distance | Closest distance | Steps | Blocked moves |
|---|---|---|---|---|---|
| generated flat (one layer, 6 half turns) | 0.28 | None | 1.24 | None | 12.6 |
| wolf | False | 20 | 5 | 200 | 0 |
| cat | False | 21 | 7 | 200 | 38 |
| snowflake | True | 23 | 0 | 23 | 0 |
| cobra | True | 18 | 0 | 18 | 0 |
| ball | True | 23 | 0 | 23 | 0 |

| Same targets, replayed with the exact sweep check (`executor="kinematic"`) | Success | Closest distance | Steps | Blocked moves |
|---|---|---|---|---|
| wolf | False | 7 | 200 | 63 |
| cat | False | 6 | 200 | 37 |
| snowflake | True | 0 | 24 | 1 |
| cobra | True | 0 | 18 | 0 |
| ball | False | 3 | 200 | 180 |
