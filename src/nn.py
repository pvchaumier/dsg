import numpy as np
import theano
import theano.tensor as T

# UltraDeep
from learning_method import LearningMethod
from layer import HiddenLayer, DropoutLayer
from convolution import Conv2DLayer


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


def build_network(parameters):
    """
    Build the network.
    """
    dropout_rate = 0.5
    hidden_layer_size = 256

    is_train = T.iscalar()
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
    lr_method_name, lr_method_parameters = parse_optimization_method(parameters['lr_method'])

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
