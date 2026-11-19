| Targets | Mean distance | Success (deterministic) | Success (sampled) | Steps / distance | Blocked per episode |
|---|---|---|---|---|---|
| flat, depth 2 | 4.0 | 100% | 100% | 1.0 | 0.0 |
| flat, depth 4 | 6.7 | 100% | 100% | 1.0 | 0.04 |
| flat, depth 8 | 9.9 | 100% | 100% | 1.02 | 0.16 |
| flat, depth 12 | 12.1 | 100% | 100% | 1.03 | 0.06 |
| flat, depth 14 | 12.1 | 100% | 100% | 1.04 | 0.06 |
| near, depth 4 | 3.8 | 100% | 100% | 1.0 | 0.0 |
| near, depth 10 | 8.2 | 100% | 100% | 1.02 | 0.04 |
| near, depth 20 | 13.5 | 100% | 100% | 1.02 | 0.2 |
| near, depth 30 | 16.1 | 98% | 100% | 1.05 | 0.54 |

| Shape from the stick | Executor | Deterministic | Sampled |
|---|---|---|---|
| wolf | MuJoCo | not reached, closest 3 of 20 (78 blocked) | not reached, closest 3 of 20 (54 blocked) |
| wolf | exact sweep | not reached, closest 3 of 20 (78 blocked) | not reached, closest 3 of 20 (54 blocked) |
| cat | MuJoCo | not reached, closest 5 of 21 (88 blocked) | not reached, closest 5 of 21 (80 blocked) |
| cat | exact sweep | not reached, closest 5 of 21 (88 blocked) | not reached, closest 5 of 21 (80 blocked) |
| snowflake | MuJoCo | reached in 23 moves (0 blocked) | reached in 23 moves (0 blocked) |
| snowflake | exact sweep | reached in 23 moves (0 blocked) | reached in 23 moves (0 blocked) |
| cobra | MuJoCo | reached in 18 moves (0 blocked) | reached in 18 moves (0 blocked) |
| cobra | exact sweep | reached in 18 moves (0 blocked) | reached in 18 moves (0 blocked) |
| ball | MuJoCo | reached in 29 moves (4 blocked) | reached in 37 moves (6 blocked) |
| ball | exact sweep | reached in 29 moves (4 blocked) | reached in 37 moves (6 blocked) |
