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

`src/snake3.html` uses these modules for placing the segments and for collision checks, so it must be served together with
`kinematics.js` and `sweep.js`.

Run the tests with `npm test` (Node 22 or later, no dependencies).
