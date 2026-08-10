import numpy as np
from src.render import render_for_vlm

T_passive_active = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -0.1],
    [0, 0, 0, 1]
])

render_for_vlm("models/meshes/blue_cup/blue_cup.obj", "models/meshes/green_cup/green_cup.obj",
               "./render.png", T_passive_world=T_passive_active, axis="world")

