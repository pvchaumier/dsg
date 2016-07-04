import os
import sys
import time
import logging
import argparse
import numpy as np
import theano
import scipy
import theano.sandbox.cuda

import pandas as pd

# Parse parameters
parser = argparse.ArgumentParser(description='first-test')
parser.add_argument("--network", type=str, default="simple", help="Network type")
parser.add_argument("--height", type=int, default=150, help="Image height")
parser.add_argument("--width", type=int, default=200, help="Image width")
parser.add_argument("--gray", type=int, default=0, help="Use grayscale")
parser.add_argument("--n_filters", type=int, default=96, help="Number of filters on each layer")
parser.add_argument("--n_conv", type=int, default=2, help="Number of convolutional layers")
parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr_method", type=str, default="rmsprop", help="Learning method (SGD, RMSProp, Adadelta, Adam..)")
parser.add_argument("--dropout", type=float, default=0., help="Dropout")
parser.add_argument("--data_augm", type=int, default=1, help="Data augmentation")
parser.add_argument("--dev_size", type=int, default=1500, help="Development set size")
parser.add_argument("--comment", type=str, default="", help="Comment")
parser.add_argument("--evaluate", type=int, default=0, help="Fast evaluation of the model")
parser.add_argument("--reload", type=int, default=0, help="Reload previous model")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--server", type=str, default="", help="Server")
parser.add_argument("--seed", type=int, default=0, help="Seed")
opts = parser.parse_args()


# Set path and libraries locations
if opts.server == 'lor':
    sys.path.append('/usr0/home/glample/Research/perso/UltraDeep/')
    data_path = '/usr0/home/glample/Research/kaggle/DataScienceGame/data/'
    dump_path = '/usr0/home/glample/dumped/kaggle/'
elif opts.server == 'local':
    sys.path.append('/home/guillaume/Documents/Research/perso/UltraDeep/')
    data_path = '/home/guillaume/Documents/Research/kaggle/DataScienceGame/data/'
    dump_path = '/home/guillaume/dumped/kaggle/'
else:
    assert False


# Libraries / Helpers
from utils import get_experiment_name
from experiment import Experiment
from nn import build_network, build_vgg_network
from helpers import process_image, random_transformation


# Initialize seed and GPU
np.random.seed(opts.seed)
if opts.gpu_id >= 0:
    theano.sandbox.cuda.use("gpu%i" % opts.gpu_id)


# Parse and save parameters
parameters = {}
parameters['network'] = opts.network
parameters['height'] = opts.height
parameters['width'] = opts.width
parameters['gray'] = opts.gray == 1
parameters['n_filters'] = opts.n_filters
parameters['n_conv'] = opts.n_conv
parameters['hidden_dim'] = opts.hidden_dim
parameters['batch_size'] = opts.batch_size
parameters['lr_method'] = opts.lr_method
parameters['dropout'] = opts.dropout
parameters['data_augm'] = opts.data_augm == 1
parameters['dev_size'] = opts.dev_size
parameters['comment'] = opts.comment
parameters['seed'] = opts.seed


# Parameters check
assert parameters['n_conv'] in xrange(2, 4) and 0 <= parameters['dropout'] < 1
assert parameters['network'] in ['simple', 'vgg']
assert parameters['network'] == 'simple' or (not parameters['gray'] and parameters['height'] == parameters['width'] == 224)

# Define experiment
experiment_name = 'first,' + get_experiment_name(parameters)
experiment = Experiment(experiment_name, dump_path)


# Experiment initialization
logger = logging.getLogger()
logger.info('Starting experiment: %s' % experiment_name)
logger.info(parameters)


# Build the network
logger.info("Building network...")
f_eval, f_train = (build_network if parameters['network'] == 'simple' else build_vgg_network)(parameters, experiment, opts.evaluate)


# Reload the previous best model
if opts.reload == 1 or opts.evaluate:
    logger.info('Reloading previous model...')
    experiment.load()
else:
    assert False, ('This file is to evaluate for semi supervised learning, '
                   'not to train a new model.')


# Load dataset
images_path = os.path.join(data_path, 'roof_images')
path_train_csv=os.path.join(data_path, 'id_train.csv')
path_test_csv=os.path.join(data_path, 'id_train.csv')
logger.info('Found %i images' % len(ids_ssl))


def load_images_in_memory():
    """Load the images for semi-supervised learning in memory."""
    # Get the ids of the images to use for semi-supervised learning
    ids_train = list(pd.read_csv(path_train_csv)['Id'])
    ids_test = list(pd.read_csv(path_test_csv)['Id'])
    ids_not_to_label = ids_train + ids_test
    ids = [int(file.split('.')[0])
           for file in os.listdir(path_images)
           if file.endswith('.jpg')]
    ids_to_label = [i for i in ids if i not in ids_not_to_label]

    # Build the dataset in memory
    id_to_img = {}
    for i, img_id in enumerate(ids_to_label):
        if i == 0:
            continue
        img_path = os.path.join(images_path, str(img_id) + '.jpg')
        id_to_img[i - 1] = {
            'img_id': img_id,
            'label': 4,
            'image': scipy.misc.imread(img_path)
        }

    return id_to_img


def write_proba_predictions(ids, y_proba_pred,
                            predpath=os.path.join(data_path, 'id_ssl.csv')):
    """Write the probability of belonging to each class in a csv file."""
    assert len(ids) == y_proba_pred.shape[0]
    with open(predpath, 'w') as f:
        f.write('Id,Class1,Class2,Class3,Class4\n')
        for i in len(ids):
            f.write(
                str(ids[i]) + ',' +
                str(y_proba_pred[i,0]) + ',' +
                str(y_proba_pred[i,1]) + ',' +
                str(y_proba_pred[i,2]) + ',' +
                str(y_proba_pred[i,3])
            )
    logger.info('Wrote %i predictions into %s' % (len(predictions), predpath))


def predict_proba(x, batch_size=100):
    """Return the probability of belonging to each class."""
    x = np.array(x)

    # Voting classification
    sum_votes = np.ones((len(x), 4)).astype(np.float32)

    # Identity
    predictions1 = np.zeros((len(x), 4)).astype(np.float32)
    for i in xrange(0, len(x), batch_size):
        predictions1[i:i + batch_size] = f_eval(x[i:i + batch_size])
    # sum_votes[np.arange(len(x)), predictions1.argmax(axis=1)] += 1
    sum_votes *= predictions1
    logger.info('1/7')

    # Horizontal flip
    predictions2 = np.zeros((len(x), 4)).astype(np.float32)
    for i in xrange(0, len(x), batch_size):
        predictions2[i:i + batch_size] = f_eval(x[i:i + batch_size][:, :, ::-1, :])
    # sum_votes[np.arange(len(x)), predictions2.argmax(axis=1)] += 1
    sum_votes *= predictions2
    logger.info('2/7')

    # Vertical flip
    predictions3 = np.zeros((len(x), 4)).astype(np.float32)
    for i in xrange(0, len(x), batch_size):
        predictions3[i:i + batch_size] = f_eval(x[i:i + batch_size][:, :, :, ::-1])
    # sum_votes[np.arange(len(x)), predictions3.argmax(axis=1)] += 1
    sum_votes *= predictions3
    logger.info('3/7')

    # Rotation 90
    predictions4 = np.zeros((len(x), 4)).astype(np.float32)
    for i in xrange(0, len(x), batch_size):
        predictions4[i:i + batch_size] = f_eval(np.rot90(np.transpose(x[i:i + batch_size], (2, 3, 0, 1)), 1).transpose((2, 3, 0, 1)))
    predictions4 = predictions4[:, (1, 0, 2, 3)]
    # sum_votes[np.arange(len(x)), predictions4.argmax(axis=1)] += 1
    sum_votes *= predictions4
    logger.info('4/7')

    # Rotation 180
    predictions5 = np.zeros((len(x), 4)).astype(np.float32)
    for i in xrange(0, len(x), batch_size):
        predictions5[i:i + batch_size] = f_eval(x[i:i + batch_size][:, :, ::-1, ::-1])
    # sum_votes[np.arange(len(x)), predictions5.argmax(axis=1)] += 1
    sum_votes *= predictions5
    logger.info('5/7')

    # Rotation 270
    predictions6 = np.zeros((len(x), 4)).astype(np.float32)
    for i in xrange(0, len(x), batch_size):
        predictions6[i:i + batch_size] = f_eval(np.rot90(np.transpose(x[i:i + batch_size], (2, 3, 0, 1)), 3).transpose((2, 3, 0, 1)))
    predictions6 = predictions6[:, (1, 0, 2, 3)]
    # sum_votes[np.arange(len(x)), predictions6.argmax(axis=1)] += 1
    sum_votes *= predictions6
    logger.info('6/7')

    # Transposition
    predictions7 = np.zeros((len(x), 4)).astype(np.float32)
    for i in xrange(0, len(x), batch_size):
        predictions7[i:i + batch_size] = f_eval(x[i:i + batch_size].transpose((0, 1, 3, 2)))
    predictions7 = predictions7[:, (1, 0, 2, 3)]
    # sum_votes[np.arange(len(x)), predictions7.argmax(axis=1)] += 1
    sum_votes *= predictions7
    logger.info('7/7')

    # Final prediction
    return sum_votes


# Evaluate the model and write the predictions into a file
id_to_img_ssl = load_images_in_memory()
logger.info('Found %i images to classify.' % len(id_to_img_ssl))
id_data_test, x_data_test = zip(
    *[(v['img_id'], process_image(v['image'], parameters['gray'], opts.height, opts.width).astype(np.float32) / 255.)
    for _, v in id_to_img_test.items()]`
)
y_pred_trans = predict_proba(x_data_test)
write_proba_predictions(id_data_test, y_pred_trans)
