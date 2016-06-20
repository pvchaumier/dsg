"""Preprocess image."""


def rotation(img, angle):
    pass


def transpose(img):
    """Transpose the image."""
    return img[::-1,::-1,:]


def symmetry_horiz(img):
    """Flips the image along an horizontal axis."""
    return img[::-1,:,:]


def symmetry_vertical(img):
    """Flips the image along an vertical axis."""
    return img[:,::-1,:]


def resize(img, new_size=80):
    pass


def kmeans(img, k=20):
    pass
