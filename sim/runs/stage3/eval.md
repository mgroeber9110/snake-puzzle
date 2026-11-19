| Targets | Mean distance | Success (deterministic) | Success (sampled) | Steps / distance | Blocked per episode |
|---|---|---|---|---|---|
| flat, depth 2 | 4.0 | 100% | 100% | 1.0 | 0.0 |
| flat, depth 4 | 6.7 | 96% | 100% | 1.0 | 0.0 |
| flat, depth 8 | 9.9 | 96% | 100% | 1.02 | 0.1 |
| flat, depth 12 | 12.1 | 86% | 100% | 1.01 | 0.78 |
| flat, depth 14 | 12.1 | 92% | 100% | 1.0 | 0.92 |
| near, depth 4 | 3.8 | 100% | 100% | 1.0 | 0.0 |
| near, depth 10 | 8.2 | 100% | 100% | 1.0 | 0.02 |
| near, depth 20 | 13.5 | 88% | 98% | 1.01 | 4.78 |
| near, depth 30 | 16.1 | 88% | 90% | 1.02 | 8.52 |

| Shape from the stick | Executor | Deterministic | Sampled |
|---|---|---|---|
| wolf | MuJoCo | not reached, closest 12 of 20 (1 blocked) | not reached, closest 3 of 20 (2 blocked) |
| wolf | exact sweep | not reached, closest 13 of 20 (0 blocked) | not reached, closest 3 of 20 (2 blocked) |
| cat | MuJoCo | not reached, closest 13 of 21 (1 blocked) | not reached, closest 5 of 21 (20 blocked) |
| cat | exact sweep | not reached, closest 13 of 21 (0 blocked) | not reached, closest 5 of 21 (8 blocked) |
| snowflake | MuJoCo | reached in 23 moves (0 blocked) | reached in 23 moves (0 blocked) |
| snowflake | exact sweep | reached in 23 moves (0 blocked) | reached in 23 moves (0 blocked) |
| cobra | MuJoCo | reached in 18 moves (0 blocked) | reached in 18 moves (0 blocked) |
| cobra | exact sweep | reached in 18 moves (0 blocked) | reached in 18 moves (0 blocked) |
| ball | MuJoCo | reached in 25 moves (0 blocked) | reached in 25 moves (0 blocked) |
| ball | exact sweep | reached in 25 moves (0 blocked) | reached in 25 moves (0 blocked) |
