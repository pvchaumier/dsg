import numpy as np
import cv2


def process_image(image, gray, height, width):
    image = image.astype(np.float32).mean(axis=2) if gray else image
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image[np.newaxis, :, :] if gray else image.transpose(2, 0, 1)


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


def rotation_90(img):
    """
    Rotate the image of 90 degrees.
    """
    return np.rot90(img.transpose((1, 2, 0)), 1).transpose((2, 0, 1))


def rotation_180(img):
    """
    Rotate the image of 180 degrees.
    """
    return img[:, ::-1, ::-1]


def rotation_270(img):
    """
    Rotate the image of 270 degrees.
    """
    return np.rot90(img.transpose((1, 2, 0)), 3).transpose((2, 0, 1))


def transposition(img):
    """
    Transpose the image.
    """
    return img.transpose((0, 2, 1))


def random_transformation(x, y):
    """
    Create a random transformation for a given sample.
    """
    assert y in xrange(4)
    # y == 0: North-South orientation
    # y == 1: East-West orientation
    # y == 2: Flat roof
    # y == 3: Other

    possible_transformations = ["identity", "horizontal_flip", "vertical_flip", "rotation_90", "rotation_180", "rotation_270", "transposition"]
    transformation = np.random.choice(possible_transformations)

    if transformation == "identity":
        return x, y
    elif transformation == "horizontal_flip":
        return horizontal_flip(x), y
    elif transformation == "vertical_flip":
        return vertical_flip(x), y
    elif transformation == "rotation_90":
        if y <= 1:
            return rotation_90(x), 1 - y
        else:
            return rotation_90(x), y
    elif transformation == "rotation_180":
        return rotation_180(x), y
    elif transformation == "rotation_270":
        if y <= 1:
            return rotation_270(x), 1 - y
        else:
            return rotation_270(x), y
    elif transformation == "transposition":
        if y <= 1:
            return transposition(x), 1 - y
        else:
            return transposition(x), y

    assert(False)
