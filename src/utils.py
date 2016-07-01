import numpy as np
import cv2


def process_image(image, gray, height, width):
    image = image.astype(np.float32).mean(axis=2) if gray else image
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image if gray else image.transpose(2, 0, 1)


def horizontal_flip(img):
    """
    Flips the image along the horizontal axis.
    """
    return img[:, ::-1, :]


def vertical_flip(img):
    """
    Flips the image along the vertical axis.
    """
    return img[:, :, ::-1]


def rotation_180(img):
    """
    Rotate the image of 180 degrees.
    """
    return img[:, ::-1, ::-1]


def transpose(img):
    """
    Transpose the image.
    """
    return img.transpose((0, 2, 1))
