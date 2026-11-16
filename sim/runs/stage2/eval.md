| Targets | Mean distance | Success (deterministic) | Success (sampled) | Steps / distance | Blocked per episode |
|---|---|---|---|---|---|
| flat, depth 2 | 4.0 | 100% | 100% | 1.0 | 0.0 |
| flat, depth 4 | 6.7 | 100% | 100% | 1.0 | 0.0 |
| flat, depth 8 | 9.9 | 100% | 100% | 1.0 | 0.0 |
| flat, depth 12 | 12.1 | 100% | 100% | 1.0 | 0.0 |
| flat, depth 14 | 12.1 | 100% | 100% | 1.0 | 0.0 |
| near, depth 4 | 3.8 | 100% | 100% | 1.0 | 0.0 |
| near, depth 10 | 8.2 | 98% | 100% | 1.0 | 1.84 |
| near, depth 20 | 13.5 | 94% | 100% | 1.0 | 8.0 |
| near, depth 30 | 16.1 | 92% | 100% | 1.01 | 13.18 |

| Shape from the stick | Executor | Deterministic | Sampled |
|---|---|---|---|
| wolf | MuJoCo | not reached, closest 5 of 20 (112 blocked) | not reached, closest 3 of 20 (39 blocked) |
| wolf | exact sweep | not reached, closest 3 of 20 (61 blocked) | not reached, closest 3 of 20 (39 blocked) |
| cat | MuJoCo | not reached, closest 6 of 21 (155 blocked) | not reached, closest 5 of 21 (62 blocked) |
| cat | exact sweep | not reached, closest 6 of 21 (93 blocked) | not reached, closest 5 of 21 (90 blocked) |
| snowflake | MuJoCo | reached in 26 moves (3 blocked) | reached in 37 moves (8 blocked) |
| snowflake | exact sweep | reached in 26 moves (3 blocked) | reached in 37 moves (8 blocked) |
| cobra | MuJoCo | reached in 18 moves (0 blocked) | reached in 29 moves (7 blocked) |
| cobra | exact sweep | reached in 18 moves (0 blocked) | reached in 29 moves (7 blocked) |
| ball | MuJoCo | not reached, closest 2 of 23 (0 blocked) | not reached, closest 1 of 23 (46 blocked) |
| ball | exact sweep | not reached, closest 2 of 23 (0 blocked) | not reached, closest 1 of 23 (46 blocked) |
