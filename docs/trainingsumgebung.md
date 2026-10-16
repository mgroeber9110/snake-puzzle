# Entwurf: Trainingsumgebung für die Rubik Snake

Status: Entwurf zur Diskussion · Stand 2026-10-16

Ziel ist eine Gym-artige Umgebung (Gymnasium-API), in der ein neuronales Netz lernt, die Rubik Snake über Gelenkdrehungen zu steuern. Die bestehende Three.js-Simulation (`src/snake2.html`) bleibt die Visualisierung; das Training läuft headless gegen einen eigenen, exakten Kinematik-Kern.

## 1. Ausgangslage im Repo

`src/snake2.html` enthält das aktuelle Kinematikmodell, eng verwoben mit Three.js:

- 24 Prismen als Szenengraph-Kette (`Snake`-Klasse), jedes Gelenk ist eine `THREE.Group`, die um ihre lokale Y-Achse dreht. Gelenk `i` (1..23) verbindet Prisma `i-1` mit `i`; `angles[0]` ist unbenutzt.
- Gelenkwinkel sind Vielfache von 90°.
- Kollision: achsparallele Bounding-Boxen (`Box3.setFromObject`, um 0.2 geschrumpft), geprüft für Paare mit Abstand ≥ 3, nur im Endzustand der Drehung.

Für das Training reicht das aus drei Gründen nicht:

1. **Abhängigkeit vom Browser.** Vorwärtskinematik und Kollision brauchen Three.js und einen Szenengraphen. Für RL brauchen wir Millionen Schritte pro Stunde, headless und parallelisierbar.
2. **Kollision zu grob (aus dem Code gefolgert, nicht getestet).** Die Bounding-Box eines Halbwürfel-Prismas ist der volle Würfel. Zwei nicht benachbarte Prismen, die sich legal eine Würfelzelle teilen (wie bei der „Kugel“), würden als Kollision abgelehnt.
3. **Keine Prüfung während der Drehung.** Eine 90°-Drehung kann durch andere Prismen hindurchschwenken, obwohl Start- und Endzustand gültig sind. Für einen echten Roboter ist genau das entscheidend.

Kleinigkeit: `gameLoop` wählt absolute Zielwinkel aus `{0, π/2, π, -π/2}`, manuelle Drehungen summieren sich dagegen beliebig auf. Im neuen Kern werden Winkel immer modulo 4 gespeichert.

## 2. Entscheidung: Python für Training, JS nur zur Anzeige

**Empfehlung: Kinematik-Kern und Umgebung in Python**, die Browser-Simulation bekommt einen Replay-Modus.

| | Python | JavaScript |
|---|---|---|
| RL-Bibliotheken | Gymnasium, Stable-Baselines3, sb3-contrib (Action Masking), PyTorch, RLlib | TF.js, kaum fertige RL-Algorithmen |
| Parallelisierung | `VectorEnv`, Multiprocessing, später GPU (JAX/NumPy-vektorisiert) | Web Worker, mühsam |
| Physik für Phase 4 | MuJoCo, PyBullet | kaum brauchbar |
| Visualisierung | schwach | vorhanden (`snake2.html`) |

Damit beide Seiten dasselbe Modell rechnen, teilen sie **Testvektoren** (`tests/kinematics_vectors.json`: Gelenkwinkel → Prismenlagen, gültig/ungültig). Python ist die Referenz; JS prüft dagegen.

## 3. Kinematik-Kern (diskret, exakt)

### 3.1 Zustand

Der Zustand ist das Tupel der 23 Gelenkwinkel `q ∈ {0,1,2,3}^23` (Einheit 90°). `q = 0…0` ist die gestreckte Schlange. Die Lage im Raum wird relativ zu Prisma 0 berechnet, das fest bleibt (wie das Anker-Segment in `snake2.html`).

### 3.2 Vorwärtskinematik mit Ganzzahlen

Jedes Prisma ist ein halber Würfel (Würfel entlang einer Flächendiagonale geteilt). Alle erreichbaren Lagen liegen auf einem Gitter, deshalb rechnen wir ohne Gleitkomma:

- Pose eines Prismas: Rotationsmatrix `R_i` aus der Würfelgruppe (24 ganzzahlige 3×3-Matrizen) und Position `p_i` in halben Einheiten (Koordinaten ×2, dann ganzzahlig).
- Rekursion: `T_{i+1} = T_i · F · Rot_n(q_i · 90°)`, wobei `F` die feste Transformation von Prisma `i` zur gemeinsamen Quadratfläche ist und `Rot_n` die Drehung um die Flächennormale. Wegen der abwechselnden Lage der Prismen gibt es zwei Varianten von `F` (gerade/ungerade `i`).
- Belegung: jedes Prisma belegt eine Würfelzelle `c_i ∈ ℤ³` und eine von 12 Hälften (6 Diagonalebenen × 2 Seiten).

### 3.3 Kollision im Endzustand

Zwei Prismen kollidieren genau dann, wenn sie dieselbe Zelle belegen und **nicht** die zwei komplementären Hälften derselben Diagonalebene sind. Das ist exakt, kostet ein Hash-Lookup pro Prisma und löst Problem 2 aus Abschnitt 1.

### 3.4 Kollision während der Drehung (Sweep)

Dreht Gelenk `k`, schwenkt der kürzere Teil der Kette um die Gelenkachse. Prüfung:

- Winkel in Schritten (z. B. 5°) abtasten und die bewegten gegen die festen Prismen mit dem Separating-Axis-Test für konvexe Polyeder prüfen (Prismen sind konvex, je 5 Flächen).
- Vorfilter: nur Prismen, deren Abstand zur Drehachse überhaupt in Reichweite ist.
- Die Ergebnisse sind deterministisch und können pro `(lokale Nachbarschaft, Gelenk, Richtung)` gecacht werden.

Die Umgebung bietet beide Modi an (`collision="static"` schnell, `collision="sweep"` realistisch). Trainiert wird zuerst `static`, evaluiert und feinjustiert mit `sweep`.

### 3.5 API-Skizze

```python
class SnakeKinematics:
    def __init__(self, n_prisms: int = 24): ...
    def forward(self, q: np.ndarray) -> Poses:           # (n,3) Zellen, (n,) Hälften, (n,3,3) Rotationen
    def is_valid(self, q: np.ndarray) -> bool:           # Endzustand kollisionsfrei
    def can_move(self, q, joint: int, delta: int,
                 sweep: bool = True) -> bool:            # delta ∈ {+1, -1}
    def valid_move_mask(self, q, sweep=True) -> np.ndarray:  # (2*(n-1),) bool
```

Die Prismenzahl ist Parameter: kleine Schlangen (8, 12 Prismen) sind für Tests, erschöpfende Suche und Curriculum wertvoll.

## 4. Die Gym-Umgebung

```python
class RubikSnakeEnv(gymnasium.Env):
    def __init__(self, n_prisms=24, task="reach_shape", targets=None,
                 collision="static", max_steps=200, curriculum_depth=None): ...
```

### 4.1 Aktionen

`Discrete(2·23 = 46)`: Aktion `a` heißt Gelenk `a // 2`, Richtung `+90°` oder `-90°` (`a % 2`).

- 180° sind zwei Schritte. Das entspricht einem Servo, der in 90°-Schritten fährt, und hält den Aktionsraum klein.
- **Action Masking** statt Strafe: `action_masks()` liefert die gültigen Züge, trainiert wird mit `MaskablePPO` (sb3-contrib). Ungültige Aktionen ohne Maske: Zustand bleibt, Strafe `-1`, `info["invalid"] = True`.
- Zuordnung zu `snake2.html`: Gelenk `k` (0..22) entspricht dort Index `k+1`. Die Drehrichtung wird über die Testvektoren festgenagelt.

### 4.2 Beobachtung

`Dict`-Space, damit Policy-Netze wählen können, was sie nutzen:

| Schlüssel | Space | Inhalt |
|---|---|---|
| `angles` | `MultiDiscrete([4]*23)` | aktuelle Gelenkwinkel (für das Netz one-hot, 92 Werte) |
| `target` | `MultiDiscrete([4]*23)` | Zielkonfiguration (zielkonditioniertes Lernen) |
| `geometry` | `Box(24, 3+12)` | optional: Zelle relativ zum Schwerpunkt + Hälfte one-hot |
| `action_mask` | `MultiBinary(46)` | gültige Züge |

Die Winkel allein bestimmen den Zustand vollständig. `geometry` ist redundant, kann aber das Lernen räumlicher Kollisionen erleichtern (Kandidat für eine Ablation).

### 4.3 Aufgaben und Belohnung

**Aufgabe A, `reach_shape` (Start):** Von einer Startkonfiguration eine Zielform erreichen.

- `-0.01` pro Schritt, `+1` beim Erreichen, `terminated=True`.
- Potentialbasiertes Shaping (ändert die optimale Policy nicht): `r += γ·Φ(s') − Φ(s)` mit `Φ(s) = −d(s, Ziel)/23`, wobei `d` die Summe der zyklischen Winkelabstände `min(|Δ|, 4−|Δ|)` ist. Diese Distanz ignoriert Kollisionen und unterschätzt deshalb den echten Weg; als Shaping ist das in Ordnung.
- Optional formbasiertes Ziel: „gleiche Belegung bis auf Starrkörperbewegung“. Teurer, aber erlaubt Ziele, die nur als Form vorliegen.
- Symmetrie: Dieselbe Form kann von beiden Enden aus gelesen werden (umgekehrte Gelenkreihenfolge, eventuell mit gespiegelten Winkeln; die genaue Abbildung prüfen wir im Kern). Ein Ziel gilt auch in der gespiegelten Lesart als erreicht.

**Aufgabe B, `explore` (Hilfsaufgabe):** Belohnung für neue gültige Konfigurationen, gut zum Vortraining und als Test des Sweep-Checks.

**Aufgabe C, `locomotion` (Phase 4):** Schwerpunkt in eine Richtung bewegen. Braucht Schwerkraft, Bodenkontakt und Reibung, also eine Physik-Engine (siehe Abschnitt 6). Der rein kinematische Kern kann das nicht sinnvoll abbilden.

### 4.4 Episoden und Curriculum

- `reset()` setzt standardmäßig auf die gestreckte Schlange und zieht ein Ziel aus `targets`.
- **Reverse Curriculum:** Mit `curriculum_depth=N` startet die Episode N zufällige gültige Züge vom Ziel entfernt. N wächst, sobald die Erfolgsrate über z. B. 80 % liegt. Das löst das Problem dünner Belohnung bei 4^23 ≈ 7·10¹³ Winkelkombinationen (vor Abzug der Kollisionen).
- `truncated=True` nach `max_steps` (Default 200, bei Curriculum `4·N`).
- `info`: `distance`, `invalid`, `collision_kind` (`static`/`sweep`), `success`.

### 4.5 Baselines

Bevor ein Netz trainiert wird, brauchen wir Vergleichswerte:

1. Zufallsagent mit Maske.
2. A*/BFS mit der Winkeldistanz als Heuristik: liefert für kleine Abstände optimale Lösungen. Diese dienen als Qualitätsmaß (Schritte des Agenten / optimale Schritte) und als Daten für Imitation Learning.

## 5. Anbindung an die Browser-Simulation

Gemeinsames Austauschformat für Trajektorien:

```json
{
  "n_prisms": 24,
  "start": [0, 0, 0, "... 23 Werte ..."],
  "moves": [[5, 1], [12, -1]],
  "meta": {"policy": "ppo-v1", "task": "reach_shape", "target": "ball"}
}
```

Änderungen an `src/`:

- `src/snake-core.js`: Kinematik und Kollision aus Abschnitt 3 ohne Three.js, als ES-Modul. `snake2.html` nutzt es für die Gültigkeitsprüfung, die Three.js-Klasse nur noch für die Darstellung.
- Replay-Modus in `snake2.html`: Trajektorie per Datei oder `?replay=<url>` laden und abspielen.
- Optional später: Policy als ONNX exportieren und mit onnxruntime-web live im Browser steuern.

## 6. Ausblick: vom Puzzle zum Roboter (Phase 4)

Für die echte Bewegungssteuerung kommt eine Physikebene dazu:

- Der Kern erzeugt aus `n_prisms` und Prismengeometrie ein MuJoCo-Modell (MJCF): 23 Drehgelenke, Massen, Reibung, Boden.
- Aktionen werden kontinuierlich (Zielwinkel oder Geschwindigkeit pro Servo, `Box(-1, 1, (23,))`) oder bleiben diskrete 90°-Befehle, die ein Regler ausführt.
- Belohnung für `locomotion`: Fortschritt des Schwerpunkts minus Energie- und Kollisionskosten.
- Sim-to-Real: Domain Randomization (Masse, Reibung, Servo-Spiel), sobald Hardwaredaten vorliegen.

Die diskrete Umgebung bleibt dabei nützlich: als Planer auf hoher Ebene, der Gelenkfolgen vorschlägt, die die Physikebene ausführt.

## 7. Vorgeschlagene Struktur und Reihenfolge

```
sim/
  pyproject.toml          # rubiksnake, Abhängigkeiten: numpy, gymnasium
  rubiksnake/
    kinematics.py         # Abschnitt 3
    collision.py          # statisch + Sweep
    env.py                # RubikSnakeEnv
    shapes.py             # Zielformen-Bibliothek
    search.py             # BFS/A*-Baseline
  tests/
    kinematics_vectors.json   # geteilt mit JS
    test_kinematics.py
  train/
    ppo_masked.py         # sb3-contrib MaskablePPO
src/
  snake-core.js           # JS-Port, gegen dieselben Vektoren getestet
  snake2.html             # + Replay-Modus
```

| Schritt | Ergebnis | Abnahme |
|---|---|---|
| 1 | Kinematik-Kern + statische Kollision | Testvektoren, bekannte Formen gültig, Kugel nicht mehr falsch abgelehnt |
| 2 | Sweep-Kollision | Fälle, in denen Endzustand gültig, Drehung aber blockiert ist |
| 3 | `RubikSnakeEnv` + Baselines | `gymnasium.utils.env_checker` grün, Zufallsagent und A* laufen |
| 4 | Replay im Browser | Trajektorie aus Python sieht im Browser identisch aus |
| 5 | MaskablePPO mit Curriculum | Erfolgsrate über Curriculum-Tiefe, Vergleich zu A* |
| 6 | MuJoCo-Modell, `locomotion` | später, abhängig von der Hardware |

## 8. Offene Fragen

1. **Was soll das Netz zuerst können:** Formen erreichen (Aufgabe A, vorgeschlagener Start) oder Fortbewegung (Aufgabe C)?
2. **Hardware:** Gibt es schon einen Plan für die Aktorik (ein Servo pro Gelenk, nur 90°-Raster)? Das entscheidet, ob Phase 4 diskret oder kontinuierlich wird.
3. **Zielformen:** Welche Formen sollen in `shapes.py`? Bekannte Figuren aus der Rubik-Snake-Notation lassen sich übernehmen, sobald die Winkelkonvention über die Testvektoren festgelegt ist.
