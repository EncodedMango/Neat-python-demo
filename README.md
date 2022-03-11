# Neat-python-demo
The source code for a neat-python demo.

This is a simulation I've written in python using pygame and neat-python.
There are 250 players, each of them starts at the center, there is a terminator (large magenta dot) that eliminates the players one by one. Each player is given its current direction, distance from center, distance from center to the
world border (red circle), the direction the terminator is in, and the direction to the nearest goal (magenta dots).

Each player has a definite life-span of 200 loops, when it reaches a goal this is reset to 0.
There are 50 goals on the screen that randomly change position when a player reaches it.
