import os
import sys
import time
import logging
import argparse
import numpy as np
import theano
import theano.tensor as T
import scipy
import cv2
import theano.sandbox.cuda


# Parse parameters
parser = argparse.ArgumentParser(description='first-test')
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr_method", type=str, default="rmsprop", help="Learning method (SGD, RMSProp, Adadelta, Adam..)")
parser.add_argument("--dropout", type=float, default=0., help="Dropout")
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
else:
    assert(opts.server == 'local')
    sys.path.append('/home/guillaume/Documents/Research/perso/UltraDeep/')
    data_path = '/home/guillaume/Documents/Research/kaggle/DataScienceGame/data/'
else:
    assert False

# Initialize GPU
if opts.gpu_id >= 0:
    theano.sandbox.cuda.use("gpu0")

# Import UltraDeep
from experiment import Experiment
from learning_method import LearningMethod
from layer import HiddenLayer, EmbeddingLayer, DropoutLayer
from convolution import Conv2DLayer

# Load dataset
images_path = os.path.join(data_path, 'roof_images')
images_filenames = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
print('Found %i images' % len(images_filenames))

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
        print('Found negative ID: "%s"' % line)
        continue
    image_path = os.path.join(images_path, '%i.jpg' % abs(int(line[0])))
    assert os.path.isfile(image_path), image_path
    id_to_img[i - 1] = {
        'img_id': int(line[0]),
        'label': int(line[1]),
        'image': scipy.misc.imread(image_path)
    }
print('Found %i elements' % len(id_to_img))





x_data = [(process_image(v['image'], False, 80, 80).astype(np.float32) / 255., v['label'] - 1) for k, v in id_to_img.items()]
x_data, y_data = zip(*x_data)


dropout_rate = 0.5
hidden_layer_size = 256

is_train = T.iscalar()
x_indexes = T.ivector()
x_image = T.ftensor4()
y_image = T.ivector()

# 2 convolutions
conv_layer1 = Conv2DLayer(20, 3, 8, 8, 'valid', (4, 4), 'conv_layer1')
conv_layer2 = Conv2DLayer(64, 20, 8, 8, 'valid', (2, 2), 'conv_layer2')
conv_output = conv_layer2.link(conv_layer1.link(x_image)).reshape((x_image.shape[0], 64 * 6 * 6))
# print('conv_output', conv_output.eval({x_image: np.random.rand(2, 3, 80, 80).astype(np.float32)}).shape)

# dropout layer
dropout_layer1 = DropoutLayer(p=dropout_rate)
conv_output = T.switch(T.neq(is_train, 0), dropout_layer1.link(conv_output), (1 - dropout_rate) * conv_output)

# hidden layer
hidden_layer = HiddenLayer(64 * 6 * 6, hidden_layer_size, activation='relu')
hidden_output = hidden_layer.link(conv_output)
# print('hidden_output', hidden_output.eval({is_train: 0, x_image: np.random.rand(2, 3, 80, 80).astype(np.float32)}).shape)

# dropout layer
dropout_layer2 = DropoutLayer(p=dropout_rate)
hidden_output = T.switch(T.neq(is_train, 0), dropout_layer2.link(hidden_output), (1 - dropout_rate) * hidden_output)

# final layer
final_layer = HiddenLayer(hidden_layer_size, 4, activation='softmax')
final_output = final_layer.link(hidden_output)
# print('final_output', final_output.eval({is_train: 0, x_image: np.random.rand(2, 3, 80, 80).astype(np.float32)}).shape)

# cost
cost = T.nnet.categorical_crossentropy(final_output, y_image).mean()
# print('cost', cost.eval({is_train: 1, x_image: np.random.rand(2, 3, 80, 80).astype(np.float32), y_image: np.random.randint(0, 4, (2,)).astype(np.int32)}))


params = conv_layer1.params + conv_layer2.params + hidden_layer.params + final_layer.params
lr_method_parameters = {'lr': 0.001}

f_eval = theano.function(
    inputs=[x_image],
    outputs=final_output,
    givens={is_train: np.cast['int32'](0)}
)

f_train = theano.function(
    inputs=[x_image, y_image],
    outputs=cost,
    updates=LearningMethod(clip=5.0).get_updates('sgd', cost, params, **lr_method_parameters),
    givens={is_train: np.cast['int32'](1)}
)

def evaluate(x, y, batch_size=100):
    count_correct = 0
    for i in xrange(0, len(x), batch_size):
        count_correct += np.sum(f_eval(x[i:i + batch_size]).argmax(axis=1) == y[i:i + batch_size])
    return count_correct * 1.0 / len(x)


x_train = x_data[:7000]
y_train = y_data[:7000]
x_valid = x_data[7000:]
y_valid = y_data[7000:]

batch_size = 1
n_epochs = 1000000
# best_accuracy = -1
count = 0
last_costs = []
start = time.time()

for n_epoch in xrange(n_epochs):
    print('Starting epoch %i...' % n_epoch)
    perm = np.random.permutation(len(x_train))
    x_train = [x_train[i] for i in perm]
    y_train = [y_train[i] for i in perm]
    for j in xrange(0, len(x_train), batch_size):
        count += 1
        new_cost = f_train(x_train[j:j + batch_size], y_train[j:j + batch_size])
        last_costs.append(new_cost)
        if count % 100 == 0:
            print('{0:>6} - {1}'.format(count, np.mean(last_costs)))
            last_costs = []
    new_accuracy = evaluate(x_valid, y_valid)
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
    print('Epoch %i done.' % n_epoch)
    print('Time: %.5f - New accuracy: %.5f - Best: %.5f' % (time.time() - start, new_accuracy, best_accuracy))
