import numpy as np


def get_angle(x1, y1, x2, y2, x3, y3):
    a = np.array([x1, y1])  # стегно
    b = np.array([x2, y2])  # коліно (центральна точка)
    c = np.array([x3, y3])  # стопа (кісточка)

    ab = a - b
    cb = c - b

    dot = ab @ cb
    norm_ab = (ab @ ab) ** 0.5
    norm_cb = (cb @ cb) ** 0.5
    angle = np.arccos(dot / norm_ab / norm_cb)
    angle = angle / np.pi * 180

    return angle