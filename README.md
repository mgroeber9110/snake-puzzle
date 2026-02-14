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
