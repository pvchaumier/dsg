"""Helper functions to preprocess images.

Fot the rotation, one can use the rotate image from scikit image.
Be careful to use the options resize=True and mode='wrap'.
"""

import random

import numpy as np
import cv2

from skimage.color import rgb2hsv, hsv2rgb
from skimage.color.adapt_rgb import adapt_rgb, hsv_value
from skimage.exposure import adjust_sigmoid
from skimage.filters.rank import mean_bilateral
from skimage.morphology import disk
from skimage.transform import rotate


def process_image(image, gray, height, width):
    """Resize the image to height and width."""
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


def saturate(img, rate):
    """Change saturation."""
    assert -1.0 <= rate <= 1.0
    new_img = rgb2hsv(img)
    if rate < 0:
        new_img[:, :, 1] += rate * new_img[:, :, 1]
    else:
        new_img[:, :, 1] += rate * (1 - new_img[:, :, 1])
    return hsv2rgb(new_img)


@adapt_rgb(hsv_value)
def mean_b(img):
    """Return image with preserved borders but smoothed inners."""
    return mean_bilateral(img, disk(10), s0=10, s1=50)


def random_transformation(img, label):
    """Create a random transformation for a given sample.

    Labels:
    - 0: North-South orientation
    - 1: East-West orientation
    - 2: Flat roof
    - 3: Other
    """
    assert label in xrange(4), 'Label error, value should be between 0 and 3'

    # for the transformation involving a rotation, the angle is modified
    # randomly to change a bit the angle.
    possible_transformations = [
        'identity',
        'rotation_90',
        'rotation_180',
        'rotation_270',
        'horizontal_flip',
        'vertical_flip',
        'transposition',
        'saturate',
        'adjust_contrast',
        'mean_b'
    ]
    transformation = random.choice(possible_transformations)

    #
    # Transformation label invariant
    #
    if transformation == 'identity':
        angle = random.choice([-10, -5, 0, 5, 10])
        return rotate(img, angle, resize=True, mode='wrap'), label
    if transformation == 'horizontal_flip':
        return horizontal_flip(img), label
    if transformation == 'vertical_flip':
        return vertical_flip(img), label
    if transformation == 'rotation_180':
        angle = 180 + random.choice([-10, -5, 0, 5, 10])
        return rotate(img, angle, resize=True, mode='wrap'), label
    if transformation == 'saturate':
        rate = random.choice([0.1, 0.2, -0.25, -0.5])
        return saturate(img, rate), label
    if transformation == 'adjust_contrast':
        return adjust_sigmoid(img), label
    if transformation == 'mean_b':
        return mean_b(img), label

    #
    # Transformation label variant
    #
    new_label = 1 - label if label < 2 else label

    if transformation == 'rotation_90':
        angle = 90 + random.choice([-10, -5, 0, 5, 10])
        return rotate(img, angle, resize=True, mode='wrap'), new_label
    if transformation == 'rotation_270':
        angle = 270 + random.choice([-10, -5, 0, 5, 10])
        return rotate(img, angle, resize=True, mode='wrap'), new_label
    if transformation == 'transposition':
        return transposition(img), new_label

assert(False)
