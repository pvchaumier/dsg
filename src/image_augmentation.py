"""Augment the dataset by modifying images."""

import os

from collections import defaultdict

from scipy import misc
from skimage.transform import rotate
from helpers import horizontal_flip, resize_cv, transposition, vertical_flip

NEW_SIZE = 80


def data_augmentation(old_label_id, old_img_folder, new_img_folder):
    """Grow the size of the original dataset using data augmentation.

    Given a dictionary which keys are labels and values are list of ids of
    images of the corresponding label, create a new image folder with the
    original images plus the transformed ones and return a new dictionary of
    ids and corresponding labels.

    The different labels are:
    - 0: North-South orientation
    - 1: East-West orientation
    - 2: Flat roof
    - 3: Other
    """
    new_label_id = defaultdict(list)
    for label, img_id in old_label_id.items():
        # north-south orientation
        old_img_path = os.path.join(old_img_folder, str(img_id) + '.jpg')
        img = misc.imread(old_img_path)

        # Save original image
        img_resized = resize_cv(img, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id))


        #
        # non label specific transformations
        #

        # horizontal flip, image saved with a _h at the end of the name
        img_horiz = horizontal_flip(img)
        img_resized = resize_cv(img_horiz, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_h.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_h')

        # vertical flip, image saved with a _v at the end of the name
        img_vert = vertical_flip(img)
        img_resized = resize_cv(img_vert, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_v.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_v')

        # 180° rotation
        img_rot180 = rotate(img, 180)
        img_resized = resize_cv(img_rot180, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r180.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r180')


        # small angle rotation (5 and 10°)
        img_rot5 = rotate(img, 5, resize=True, mode='wrap')
        img_resized = resize_cv(img_rot5, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r5.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r5')
        img_rot10 = rotate(img, 10, resize=True, mode='wrap')
        img_resized = resize_cv(img_rot10, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r10.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r10')


        #
        # label specific transformation
        #

        # transposition
        img_tr = transposition(img)
        img_resized = resize_cv(img_rot180, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_tr.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_tr')

        # rotation 90°
        img_rot90 = rotate(img, 90)
        img_resized = resize_cv(img_rot90, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r90.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r90')
        img_rot80 = rotate(img, 80, resize=True, mode='wrap'))
        img_resized = resize_cv(img_rot80, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r80.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r80')
        img_rot85 = rotate(img, 85, resize=True, mode='wrap'))
        img_resized = resize_cv(img_rot85, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r85.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r85')
        img_rot95 = rotate(img, 95, resize=True, mode='wrap'))
        img_resized = resize_cv(img_rot95, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r95.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r95')
        img_rot100 = rotate(img, 100, resize=True, mode='wrap'))
        img_resized = resize_cv(img_rot100, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r100.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r100')

        # rotation 270°
        img_rot270 = rotate(img, 270)
        img_resized = resize_cv(img_rot270, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r270.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r270')
        img_rot260 = rotate(img, 260, resize=True, mode='wrap'))
        img_resized = resize_cv(img_rot260, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r260.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r260')
        img_rot265 = rotate(img, 265, resize=True, mode='wrap'))
        img_resized = resize_cv(img_rot265, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r265.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r265')
        img_rot275 = rotate(img, 275, resize=True, mode='wrap'))
        img_resized = resize_cv(img_rot275, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r275.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r275')
        img_rot280 = rotate(img, 280, resize=True, mode='wrap'))
        img_resized = resize_cv(img_rot280, False, NEW_SIZE, NEW_SIZE)
        new_img_path = os.path.join(new_img_folder, str(img_id)) + '_r280.jpg'
        misc.imsave(new_img_path, img_resized)
        new_label_id[label].append(str(img_id)) + '_r280')


    return new_label_id
