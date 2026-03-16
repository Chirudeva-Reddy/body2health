import cv2
import numpy as np
import yaml


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_mask(path: str, mask_uint8: np.ndarray) -> None:
    cv2.imwrite(path, mask_uint8)


def save_png(path: str, img_uint8: np.ndarray) -> None:
    cv2.imwrite(path, img_uint8)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
