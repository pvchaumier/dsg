import os
import sys
import time
import logging
import argparse
import numpy as np
import theano
import scipy
import theano.sandbox.cuda


# Parse parameters
parser = argparse.ArgumentParser(description='first-test')
parser.add_argument("--height", type=int, default=150, help="Image height")
parser.add_argument("--width", type=int, default=200, help="Image width")
parser.add_argument("--gray", type=int, default=0, help="Use grayscale")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr_method", type=str, default="rmsprop", help="Learning method (SGD, RMSProp, Adadelta, Adam..)")
parser.add_argument("--dropout", type=float, default=0., help="Dropout")
parser.add_argument("--dev_size", type=int, default=1500, help="Development set size")
parser.add_argument("--comment", type=str, default="", help="Comment")
parser.add_argument("--evaluate", type=int, default=0, help="Fast evaluation of the model")
parser.add_argument("--reload", type=int, default=0, help="Reload previous model")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--server", type=str, default="", help="Server")
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
from nn import build_network
from helpers import process_image


# Initialize GPU
if opts.gpu_id >= 0:
    theano.sandbox.cuda.use("gpu0")


# Parse and save parameters
parameters = {}
parameters['height'] = opts.height
parameters['width'] = opts.width
parameters['gray'] = opts.gray == 1
parameters['n_conv'] = opts.n_conv
parameters['hidden_dim'] = opts.hidden_dim
parameters['batch_size'] = opts.batch_size
parameters['lr_method'] = opts.lr_method
parameters['dropout'] = opts.dropout
parameters['dev_size'] = opts.dev_size
parameters['comment'] = opts.comment


# Define experiment
experiment_name = 'first,' + get_experiment_name(parameters)
experiment = Experiment(experiment_name, dump_path)


# Experiment initialization
logger = logging.getLogger()
logger.info('Starting experiment: %s' % experiment_name)
if int(opts.reload) == 1 or opts.evaluate:
    logger.info('Reloading previous model...')
    experiment.load()
logger.info(parameters)


# Build the network
logger.info("Building network...")
f_eval, f_train = build_network(parameters)


# Load dataset
images_path = os.path.join(data_path, 'roof_images')
images_filenames = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
logger.info('Found %i images' % len(images_filenames))


# Parse CSV file
id_to_img = {}
img_to_id = {}
for i, line in enumerate(open(os.path.join(data_path, 'id_train.csv'))):
    if i == 0:
        continue
    line = line.rstrip().split(',')
    assert len(line) == 2 and line[1].isdigit() and (line[0].isdigit() or line[0][0] == '-' and line[0][1:].isdigit())
    assert int(line[0]) not in img_to_id
    img_to_id[int(line[0])] = i - 1
    if int(line[0]) < 0:  # TODO: why are there negative IDs?????
        logger.warning('Found negative ID: "%s"' % line)
        continue
    image_path = os.path.join(images_path, '%i.jpg' % abs(int(line[0])))
    assert os.path.isfile(image_path), image_path
    id_to_img[i - 1] = {
        'img_id': int(line[0]),
        'label': int(line[1]),
        'image': scipy.misc.imread(image_path)
    }
logger.info('Found %i elements' % len(id_to_img))


# Process images, split the train and dev sets
x_data = [(process_image(v['image'], parameters['gray'], opts.height, opts.width).astype(np.float32) / 255., v['label'] - 1) for k, v in id_to_img.items()]
x_data, y_data = zip(*x_data)
x_train = x_data[:-opts.dev_size]
y_train = y_data[:-opts.dev_size]
x_valid = x_data[-opts.dev_size:]
y_valid = y_data[-opts.dev_size:]


# Evaluate the model
def evaluate(x, y, batch_size=100):
    count_correct = 0
    for i in xrange(0, len(x), batch_size):
        count_correct += np.sum(f_eval(x[i:i + batch_size]).argmax(axis=1) == y[i:i + batch_size])
    return count_correct * 1.0 / len(x)


# Training initialization
batch_size = 1
n_epochs = 1000000
best_accuracy = -1
count = 0
last_costs = []
start = time.time()


# Training
for n_epoch in xrange(n_epochs):
    logger.info('Starting epoch %i...' % n_epoch)
    perm = np.random.permutation(len(x_train))
    x_train = [x_train[i] for i in perm]
    y_train = [y_train[i] for i in perm]
    for j in xrange(0, len(x_train), batch_size):
        count += 1
        new_cost = f_train(x_train[j:j + batch_size], y_train[j:j + batch_size])
        last_costs.append(new_cost)
        if count % 100 == 0:
            logger.info('{0:>6} - {1}'.format(count, np.mean(last_costs)))
            last_costs = []
    new_accuracy = evaluate(x_valid, y_valid)
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
    logger.info('Epoch %i done.' % n_epoch)
    logger.info('Time: %.5f - New accuracy: %.5f - Best: %.5f' % (time.time() - start, new_accuracy, best_accuracy))
