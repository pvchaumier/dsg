import pickle
import operator
import numpy as np
import theano
import theano.tensor as T

# UltraDeep
from learning_method import LearningMethod
from layer import HiddenLayer, DropoutLayer
from convolution import Conv2DLayer
from pooling import PoolLayer2D


def parse_optimization_method(lr_method):
    """
    Parse optimization method parameters.
    """
    if "-" in lr_method:
        lr_method_name = lr_method[:lr_method.find('-')]
        lr_method_parameters = {}
        for x in lr_method[lr_method.find('-') + 1:].split('-'):
            split = x.split('_')
            assert len(split) == 2
            lr_method_parameters[split[0]] = float(split[1])
    else:
        lr_method_name = lr_method
        lr_method_parameters = {}

    return lr_method_name, lr_method_parameters


def build_network(parameters, experiment):
    """
    Build the network.
    """
    is_train = T.iscalar()
    x_image = T.ftensor4()
    y_image = T.ivector()

    # 2 or 3 convolutions
    conv_layer1 = Conv2DLayer(parameters['n_filters'], 1 if parameters['gray'] else 3, 3, 3, 'valid', (1, 1), 'conv_layer1')
    conv_output = conv_layer1.link(x_image)
    if parameters['n_conv'] == 3:
        conv_layer2 = Conv2DLayer(parameters['n_filters'], parameters['n_filters'], 3, 3, 'valid', (1, 1), 'conv_layer2')
        conv_output = conv_layer2.link(conv_output)
    conv_layer3 = Conv2DLayer(parameters['n_filters'], parameters['n_filters'], 3, 3, 'valid', (2, 2), 'conv_layer3')
    conv_output = conv_layer3.link(conv_output)

    # dropout layer 1
    dropout_layer1 = DropoutLayer(p=parameters['dropout'])
    conv_output = T.switch(T.neq(is_train, 0), dropout_layer1.link(conv_output), (1 - parameters['dropout']) * conv_output)

    # hidden layer
    conv_output_shape = conv_output.eval({is_train: 0, x_image: np.random.rand(1, 1 if parameters['gray'] else 3, parameters['height'], parameters['width']).astype(np.float32)}).shape
    hidden_input_dim = reduce(operator.mul, conv_output_shape, 1)
    hidden_layer = HiddenLayer(hidden_input_dim, parameters['hidden_dim'], activation='relu')
    hidden_output = hidden_layer.link(conv_output.reshape((x_image.shape[0], hidden_input_dim)))

    # dropout layer 2
    dropout_layer2 = DropoutLayer(p=parameters['dropout'])
    hidden_output = T.switch(T.neq(is_train, 0), dropout_layer2.link(hidden_output), (1 - parameters['dropout']) * hidden_output)

    # final layer
    final_layer = HiddenLayer(parameters['hidden_dim'], 4, activation='softmax')
    final_output = final_layer.link(hidden_output)

    # cost
    cost = T.nnet.categorical_crossentropy(final_output, y_image).mean()

    # parameters
    params = conv_layer1.params + conv_layer3.params
    if parameters['n_conv'] == 3:
        params += conv_layer2.params
    params += hidden_layer.params + final_layer.params

    # experiment components
    experiment.add_component(conv_layer1)
    if parameters['n_conv'] == 3:
        experiment.add_component(conv_layer2)
    experiment.add_component(conv_layer3)
    experiment.add_component(hidden_layer)
    experiment.add_component(final_layer)

    # learning method
    lr_method_name, lr_method_parameters = parse_optimization_method(parameters['lr_method'])

    # build functions
    f_eval = theano.function(
        inputs=[x_image],
        outputs=final_output,
        givens={is_train: np.cast['int32'](0)}
    )
    f_train = theano.function(
        inputs=[x_image, y_image],
        outputs=cost,
        updates=LearningMethod(5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters),
        givens={is_train: np.cast['int32'](1)}
    )

    return f_eval, f_train


def build_vgg_network(parameters, experiment):
    """
    Build the network (pretrained from VGG net).
    """
    pretrained_model = pickle.load(open('/usr0/home/glample/vgg19.pkl'))

    is_train = T.iscalar()
    x_image = T.ftensor4()
    y_image = T.ivector()

    # convolution layer 1
    conv1_1 = Conv2DLayer(64, 3, 3, 3, 1, (1, 1), 'conv1_1')
    conv1_2 = Conv2DLayer(64, 64, 3, 3, 1, (1, 1), 'conv1_2')
    pool1 = PoolLayer2D(2, 2)
    output_conv1_1 = conv1_1.link(x_image)
    output_conv1_2 = conv1_2.link(output_conv1_1)
    output_pool1 = pool1.link(output_conv1_2)

    # convolution layer 2
    conv2_1 = Conv2DLayer(128, 64, 3, 3, 1, (1, 1), 'conv2_1')
    conv2_2 = Conv2DLayer(128, 128, 3, 3, 1, (1, 1), 'conv2_2')
    pool2 = PoolLayer2D(2, 2)
    output_conv2_1 = conv2_1.link(output_pool1)
    output_conv2_2 = conv2_2.link(output_conv2_1)
    output_pool2 = pool2.link(output_conv2_2)

    # convolution layer 3
    conv3_1 = Conv2DLayer(256, 128, 3, 3, 1, (1, 1), 'conv3_1')
    conv3_2 = Conv2DLayer(256, 256, 3, 3, 1, (1, 1), 'conv3_2')
    conv3_3 = Conv2DLayer(256, 256, 3, 3, 1, (1, 1), 'conv3_3')
    conv3_4 = Conv2DLayer(256, 256, 3, 3, 1, (1, 1), 'conv3_4')
    pool3 = PoolLayer2D(2, 2)
    output_conv3_1 = conv3_1.link(output_pool2)
    output_conv3_2 = conv3_2.link(output_conv3_1)
    output_conv3_3 = conv3_3.link(output_conv3_2)
    output_conv3_4 = conv3_4.link(output_conv3_3)
    output_pool3 = pool3.link(output_conv3_4)

    # convolution layer 4
    conv4_1 = Conv2DLayer(512, 256, 3, 3, 1, (1, 1), 'conv4_1')
    conv4_2 = Conv2DLayer(512, 512, 3, 3, 1, (1, 1), 'conv4_2')
    conv4_3 = Conv2DLayer(512, 512, 3, 3, 1, (1, 1), 'conv4_3')
    conv4_4 = Conv2DLayer(512, 512, 3, 3, 1, (1, 1), 'conv4_4')
    pool4 = PoolLayer2D(2, 2)
    output_conv4_1 = conv4_1.link(output_pool3)
    output_conv4_2 = conv4_2.link(output_conv4_1)
    output_conv4_3 = conv4_3.link(output_conv4_2)
    output_conv4_4 = conv4_4.link(output_conv4_3)
    output_pool4 = pool4.link(output_conv4_4)

    # convolution layer 5
    conv5_1 = Conv2DLayer(512, 512, 3, 3, 1, (1, 1), 'conv5_1')
    conv5_2 = Conv2DLayer(512, 512, 3, 3, 1, (1, 1), 'conv5_2')
    conv5_3 = Conv2DLayer(512, 512, 3, 3, 1, (1, 1), 'conv5_3')
    conv5_4 = Conv2DLayer(512, 512, 3, 3, 1, (1, 1), 'conv5_4')
    pool5 = PoolLayer2D(2, 2)
    output_conv5_1 = conv5_1.link(output_pool4)
    output_conv5_2 = conv5_2.link(output_conv5_1)
    output_conv5_3 = conv5_3.link(output_conv5_2)
    output_conv5_4 = conv5_4.link(output_conv5_3)
    output_pool5 = pool5.link(output_conv5_4)

    # fully connected layers
    full1 = HiddenLayer(input_dim=512 * 7 * 7, output_dim=parameters['hidden_dim'], activation='relu', name='full1')
    full2 = HiddenLayer(input_dim=parameters['hidden_dim'], output_dim=4, activation='softmax', name='full2')

    # dropout layers
    dropout_layer1 = DropoutLayer(p=parameters['dropout'])
    dropout_layer2 = DropoutLayer(p=parameters['dropout'])

    # output
    output_pool5 = T.switch(T.neq(is_train, 0), dropout_layer1.link(output_pool5), (1 - parameters['dropout']) * output_pool5)
    output_full1 = full1.link(output_pool5.reshape((x_image.shape[0], 512 * 7 * 7)))
    output_full1 = T.switch(T.neq(is_train, 0), dropout_layer2.link(output_full1), (1 - parameters['dropout']) * output_full1)
    output_full2 = full2.link(output_full1)

    # cost
    cost = T.nnet.categorical_crossentropy(output_full2, y_image).mean()

    # parameters
    params = []
    params += conv1_1.params + conv1_2.params
    params += conv2_1.params + conv2_2.params
    params += conv3_1.params + conv3_2.params + conv3_3.params + conv3_4.params
    params += conv4_1.params + conv4_2.params + conv4_3.params + conv4_4.params
    params += conv5_1.params + conv5_2.params + conv5_3.params + conv5_4.params
    params += full1.params + full2.params

    # experiment components
    experiment.add_component(conv1_1)
    experiment.add_component(conv1_2)
    experiment.add_component(conv2_1)
    experiment.add_component(conv2_2)
    experiment.add_component(conv3_1)
    experiment.add_component(conv3_2)
    experiment.add_component(conv3_3)
    experiment.add_component(conv3_4)
    experiment.add_component(conv4_1)
    experiment.add_component(conv4_2)
    experiment.add_component(conv4_3)
    experiment.add_component(conv4_4)
    experiment.add_component(conv5_1)
    experiment.add_component(conv5_2)
    experiment.add_component(conv5_3)
    experiment.add_component(conv5_4)
    experiment.add_component(full1)
    experiment.add_component(full2)

    # load pretrained model
    for i, (pretrained, param) in enumerate(zip(pretrained_model['param values'], params)):
        if i == 16 * 2:
            break
        assert pretrained.shape == param.get_value().shape
        param.set_value(pretrained)

    # learning method
    lr_method_name, lr_method_parameters = parse_optimization_method(parameters['lr_method'])

    # build functions
    f_eval = theano.function(
        inputs=[x_image],
        outputs=output_full2,
        givens={is_train: np.cast['int32'](0)}
    )
    f_train = theano.function(
        inputs=[x_image, y_image],
        outputs=cost,
        updates=LearningMethod(5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters),
        givens={is_train: np.cast['int32'](1)}
    )

    return f_eval, f_train
