import numpy as np
from src.camera import Camera

cam = Camera()
cam.start()
cam.capture(render=True, save=False, calculate_extrinsics=True, extrinsics_path="artifacts/calibration/extrinsics.npy")
cam.stop()

print(np.load("artifacts/calibration/extrinsics.npy"))