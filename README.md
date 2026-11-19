# Rubik's Snake Simulation

This project is a simulation of the Rubik's Snake puzzle using Three.js. It provides an interactive 3D representation of the puzzle, allowing users to manipulate and animate the snake.

To play, serve the `src` folder from a local webserver (e.g. using `ruby -run -e httpd -- src -p 8000`).

## Iteration 1

This was one of my first attempts at "vibe coding", starting from a prototype in Perplexity and then refining it
iteratively through Copilot in VSCode and some manual coding.

The snake's segments can be rotated by clicking on them. Dragging the screen outside of the snake can be used to rotate the view.

So far, there is no checking for valid configurations.

## Iteration 2

Started from scatch, this time using only Gemini 3 (mostly Thinking) and edited exclusively through prompting.

This adds hit detection against invalid configurations and also allows starting/stopping automatic movements.

## Iteration 3: Kinematic model

`src/snake3.html` builds on iteration 2, but separates the kinematics from the rendering.
`src/kinematics.js` holds the pure kinematic model of the snake, independent of Three.js, so it can be reused
for training a controller:

- 24 wedges, 23 joints; joint angles in quarter turns (0..3), all zeros is the straight stick.
- `forwardKinematics(angles)` returns the lattice cell and rotation of every wedge (fractional angles are allowed for animation).
- `findCollisions(angles)` / `isValid(angles)` check for overlapping wedges exactly: every wedge is half of a lattice cube,
  and two wedges only fit into the same cube when they are complementary halves.
- `validMoves(angles)` lists all quarter-turn twists that lead to a valid shape (only the end state is checked, not the sweep).

`src/sweep.js` checks the path of a twist, not only its end state: it samples the rotation finely and tests the exact
wedge geometry (separating axis test on the triangular prisms). `isTwistFree(angles, joint, direction)` combines both checks.

`src/notation.js` reads and writes shapes in the notation by Thomas Wolter
(http://thomas-wolter.de/rubik_notation_en.html), e.g. `2R2-3L2-4L2-5R2-6R2-7L2-8L2-10L2-10R2-12R2` (Wolf):
`parseNotation(text)` returns the joint angles, `toNotation(angles)` the notation string. Dark wedge `n` is model wedge
`2(n-1)`, `nR` is joint `2n-2`, `nL` is joint `2n-3`, and position `p` is joint angle `p` on a left joint and `4-p` on a right joint.
This is checked against the shapes on the site: the Ball and Snowflake examples come out as ball and flat snowflake,
and the Cobra matches its picture (not its mirror image).
In `src/snake3.html` the current shape is shown in this notation, and a shape can be typed in and set with "Set Shape".

`src/snake3.html` uses these modules for placing the segments and for collision checks, so it must be served together with
`kinematics.js`, `sweep.js` and `notation.js`.

Run the tests with `npm test` (Node 22 or later, no dependencies).

## Iteration 4: Replay of the trained controller

`src/snake4.html` builds on iteration 3 and adds a replay of trajectories from the trained network (`sim/`, see
`sim/README.md`). A trajectory is a JSON file in the exchange format of `docs/training-environment.md`:
`{"num_wedges": 24, "start": [...23 angles], "moves": [[joint, direction], ...], "meta": {"target": "<Wolter notation>", "target_name": "ball"}}`.

- Pick one of the examples in `src/replays/` (Ball, Snowflake, Cobra, a near shape, a Snowflake with blocked tries,
  and a failed attempt at the Wolf), load your own file, or open `snake4.html?replay=<url>` (`&speed=1..10`,
  `&autoplay=0`).
- Play, Step and Reset control the replay. Every move is checked with `isTwistFree`; moves the network tried that are
  blocked on the way are shown as a short wiggle in red and leave the shape unchanged, as in the training environment.
- If `meta.target` is given (Wolter notation, `""` for the straight stick), the target is shown as a translucent
  ghost beside the snake (`meta.target_name` is an optional label for the panel), and the panel shows the distance to it and whether it was reached (also when
  read from the other end of the snake).
- The trajectories written by `sim/train/evaluate*.py` (`sim/runs/*/trajectories/`) can be loaded directly.

## VR and AR on Meta Quest

`src/snake-xr.html` shows the snake in VR or in AR (passthrough) with WebXR, so it runs in the Meta Quest browser
without a native app. It uses the same modules as `snake4.html` (`kinematics.js`, `sweep.js`, `notation.js`) and the
same replays and trajectory format (`?replay=<url>`, `&speed=1..10`, `&autoplay=0`, and `&cube=<cm>` for the size of
one lattice cube, default 4 cm).

- On the desktop the page works like `snake4.html`. In the headset, "Enter VR" puts the snake on a small table in a dark
  room, "Enter AR" puts it in your real room with a shadow under it.
- Inside VR/AR a panel next to the snake has the replay controls (◀ ▶ to choose an example, Play/Pause, Step, Reset,
  target ghost on/off, speed), the size of the snake, random auto moves, Straight, Recenter, the current shape in
  Wolter notation and Exit. In AR, "Place on surface" puts the snake on a table or the floor (WebXR hit test).
- Controllers: trigger on a segment twists it +90° (the panel can switch this to −90°), grip on a segment twists it
  the other way, grip elsewhere grabs the snake (or the panel) and carries it. Thumbstick left/right turns the snake,
  up/down scales it. With hand tracking, pinching works like the trigger.
- A shape can be typed into the notation field before entering VR/AR.

WebXR needs a secure origin, so the Quest browser only offers VR/AR when the page comes from `https://` or from
`localhost`. Either put `src` on an HTTPS server, or connect the Quest by USB and forward the local server to it:
`adb reverse tcp:8000 tcp:8000`, then open `http://localhost:8000/snake-xr.html` on the Quest.
For testing on the desktop, Meta's Immersive Web Emulator (browser extension) emulates a Quest.
