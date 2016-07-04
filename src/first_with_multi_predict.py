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


# Load dataset
images_path = os.path.join(data_path, 'roof_images')
images_filenames = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
logger.info('Found %i images' % len(images_filenames))


# Parse CSV file
def parse_CSV(train):
    filename = 'id_train.csv' if train else 'sample_submission4.csv'
    id_to_img = {}
    img_to_id = {}
    for i, line in enumerate(open(os.path.join(data_path, filename))):
        if i == 0:
            continue
        line = line.rstrip().split(',')
        assert len(line) == 2 and line[1].isdigit() and (line[0].isdigit() or line[0][0] == '-' and line[0][1:].isdigit())
        assert int(line[0]) not in img_to_id
        img_to_id[int(line[0])] = i - 1
        # if int(line[0]) < 0:  # TODO: why are there negative IDs?????
        #     logger.warning('Found negative ID: "%s"' % line)
        #     continue
        image_path = os.path.join(images_path, '%i.jpg' % int(line[0]))
        assert os.path.isfile(image_path), image_path
        id_to_img[i - 1] = {
            'img_id': int(line[0]),
            'label': int(line[1]),
            'image': scipy.misc.imread(image_path)
        }
    return id_to_img, img_to_id


# Load training set images
id_to_img, img_to_id = parse_CSV(True)
logger.info('Found %i images for training.' % len(id_to_img))


# Process images
x_data = [(v['img_id'], process_image(v['image'], parameters['gray'], opts.height, opts.width).astype(np.float32) / 255., v['label'] - 1) for _, v in id_to_img.items()]
id_data, x_data, y_data = zip(*x_data)


# Split the train and dev sets
np.random.seed(opts.seed)
permutation = np.random.permutation(len(x_data))
x_data = [x_data[i] for i in permutation]
y_data = [y_data[i] for i in permutation]
x_train = x_data[:-opts.dev_size]
y_train = y_data[:-opts.dev_size]
x_valid = x_data[-opts.dev_size:]
y_valid = y_data[-opts.dev_size:]


# Make predictions
def get_predictions(x, batch_size=100):
    predictions = np.zeros(len(x)).astype(np.int32)
    for i in xrange(0, len(x), batch_size):
        predictions[i:i + batch_size] = f_eval(x[i:i + batch_size]).argmax(axis=1)
    return predictions


# Make predictions using transformations
def get_predictions_transformations(x, batch_size=100):
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
    return sum_votes.argmax(axis=1)


# Evaluate the model
def evaluate(x, y):
    y_pred = get_predictions(x)
    return np.sum(y == y_pred) * 1.0 / len(x)


# Evaluate the model using transformations
def evaluate_transformations(x, y):
    y_pred = get_predictions_transformations(x)
    return np.sum(y == y_pred) * 1.0 / len(x)


# Write predictions into a file
def write_predictions(idx, predictions):
    assert len(idx) == len(predictions)
    predictions_path = os.path.join(experiment.dump_path, 'predictions.csv')
    with open(predictions_path, 'w') as f:
        f.write('Id,label\n' + '\n'.join("%i,%i" % (i, y + 1) for (i, y) in zip(idx, predictions)) + '\n')
    logger.info('Wrote %i predictions into %s' % (len(predictions), predictions_path))


# Write train + dev predictions with confidence and label into a file
def write_predictions_with_conf(idx, predictions, predictions_gold):
    assert len(idx) == len(predictions) == len(predictions_gold)
    predictions_path = os.path.join(experiment.dump_path, 'train_predictions.csv')
    with open(predictions_path, 'w') as f:
        f.write('Id,label\n' + '\n'.join("%i,%s,%i,%i" % (i, str(y), y.argmax() + 1, y_gold + 1) for (i, y, y_gold) in zip(idx, predictions, predictions_gold)) + '\n')
    logger.info('Wrote %i predictions into %s' % (len(predictions), predictions_path))


# Evaluate the model and write the predictions into a file
if opts.evaluate == 1:
    # y_pred = get_predictions(x_valid)
    # logger.info('Score on dev: %f' % (np.sum(y_pred == y_valid) * 1.0 / len(y_valid)))
    y_pred_trans = get_predictions_transformations(x_valid)
    logger.info('Score on dev using transformations: %f' % (np.sum(y_pred_trans == y_valid) * 1.0 / len(y_valid)))
    id_to_img_test, img_to_id_test = parse_CSV(False)
    logger.info('Found %i images to classify.' % len(id_to_img_test))
    id_data_test, x_data_test = zip(*[(v['img_id'], process_image(v['image'], parameters['gray'], opts.height, opts.width).astype(np.float32) / 255.) for _, v in id_to_img_test.items()])
    y_pred_trans = get_predictions_transformations(x_data_test)
    write_predictions(id_data_test, y_pred_trans)
    exit()


# Evaluate the model and write the train + dev predictions into a file, with the probabilities and the correct label
if opts.evaluate == 2:
    y_pred = get_predictions(x_valid)
    logger.info('Score on dev: %f' % (np.sum(y_pred == y_valid) * 1.0 / len(y_valid)))
    write_predictions_with_conf(id_data, y_pred, y_valid)
    exit()


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
    permutation = np.random.permutation(len(x_train))
    x_train = [x_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]
    for j in xrange(0, len(x_train), batch_size):
        count += 1
        x_input, y_input = x_train[j:j + batch_size], y_train[j:j + batch_size]
        if parameters['data_augm']:
            x_input, y_input = zip(*[random_transformation(x, y) for x, y in zip(x_input, y_input)])
        new_cost = f_train(x_input, y_input)
        last_costs.append(new_cost)
        if count % 200 == 0:
            logger.info('{0:>6} - {1}'.format(count, np.mean(last_costs)))
            last_costs = []
    new_accuracy = evaluate_transformations(x_valid, y_valid)
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
        experiment.dump('New best accuracy: %f' % best_accuracy)
    logger.info('Epoch %i done.' % n_epoch)
    logger.info('Time: %.5f - New accuracy: %.5f - Best: %.5f' % (time.time() - start, new_accuracy, best_accuracy))
