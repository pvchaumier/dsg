"""Helper functions to preprocess images.

Fot the rotation, one can use the rotate image from scikit image.
Be careful to use the options resize=True and mode='wrap'.
"""

import numpy as np
import cv2

from skimage.transform import rotate

def process_image(image, gray, height, width):
    image = image.astype(np.float32).mean(axis=2) if gray else image
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if not gray:
        image = image[:, :, (2, 1, 0)]  # RGB -> BGR (seriously...)
    return image[np.newaxis, :, :] if gray else image.transpose(2, 0, 1)


def horizontal_flip(img):
    """Flip the image along the horizontal axis."""
    return img[:, ::-1, :]


def vertical_flip(img):
    """Flip the image along the vertical axis."""
    return img[:, :, ::-1]


def transposition(img):
    """Transpose the image."""
    return img.transpose((0, 2, 1))


def random_transformation(img, label):
    """Create a random transformation for a given sample."""
    assert label in xrange(4)
    # label == 0: North-South orientation
    # label == 1: East-West orientation
    # label == 2: Flat roof
    # label == 3: Other

    possible_transformations = [
        'identity', 'horizontal_flip', 'vertical_flip',
        'rotation_90', 'rotation_180', 'rotation_270', 'transposition'
    ]
    transformation = np.random.choice(possible_transformations)

    if transformation == "identitlabel":
        return img, label
    elif transformation == "horizontal_flip":
        return horizontal_flip(img), label
    elif transformation == "vertical_flip":
        return vertical_flip(img), label
    elif transformation == "rotation_90":
        if label <= 1:
            return rotate(img, 90, resize=True), 1 - label
        else:
            return rotate(img, 90, resize=True), label
    elif transformation == "rotation_180":
        return rotate(img), label
    elif transformation == "rotation_270":
        if label <= 1:
            return rotate(img, 270, resize=True), 1 - label
        else:
            return rotate(img, 270, resize=True), label
    elif transformation == "transposition":
        if label <= 1:
            return transposition(img), 1 - label
        else:
            return transposition(img), label

    assert(False)
