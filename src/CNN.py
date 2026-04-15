"""
Convolutional Neural Network Wrapper.

Author: Thomas W. Rauber mailto:trauber@gmail.com
"""


from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, f1_score


from keras.models import Sequential
# from keras.layers import Layer
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
# from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
# from keras.utils import to_categorical

from matplotlib import pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np



# http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
class CNN(BaseEstimator, RegressorMixin, ClassifierMixin):
    """
    Convolutional Neural Network Wrapper.

    https://keras.io/layers/convolutional/
    Author: Thomas W. Rauber mailto:trauber@gmail.com
    """

    def __init__(self, activation='relu', activation_hidden='sigmoid',
                 activation_out='softmax', batch_size=1, chan=1, dim=1,
                 epochs=10, filters=1, kernel_size=1,
                 loss='mean_squared_error',
                 dim_output=2, num_hidden=10,
                 optimizer=SGD(), verbose=1):

        self.activation = activation
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        self.batch_size = batch_size
        self.chan = chan				 # inumber of input channels
        self.dim = dim					 # dimension of input feature vector
        self.epochs = epochs
        self.filters = filters			 # Integer, the dimensionality of the
                                         # output space (i.e. the number of
                                         # output filters in the convolution)
        self.kernel_size = kernel_size	 # An integer or tuple/list of a
                                         # single integer, specifying the
                                         # length of the 1D convolution window
        self.loss = loss
        self.dim_output = dim_output	 # number classes
        self.num_hidden = num_hidden
        self.optimizer = optimizer
        self.verbose = verbose
        # Check, if all parameters list are stored as local class parameter
        # Otherwise a warning will be issued

        self.seq = Sequential(name='ConvNet1')
        self.name = 'Simple CNN'

        # create 1-D convolution layer
        # https://keras.io/api/layers/convolution_layers
        #
        # input: dx1-dimensional signals with chan channels -> (dim, chan) tensors.
        input_shape = (dim, chan)

        # Output Shape: (None, dim-kernel_size+1, filters) = (d1,d2,d3)
        # Param # = (chan x kernel_size + 1) x filters  ; (+0, if no bias)
        convlay = Conv1D(name='convlay', input_shape=input_shape,
                         kernel_size=kernel_size, filters=filters,
                         use_bias=True)
        self.seq.add(convlay)
        # Output Shape: (None, dim-kernel_size+1, filters) = (d1,d2,d3)
        actlay = Activation(activation)
        self.seq.add(actlay)   # Function of ReLU activation is detector, [1], p. 71, fig. 5.9

        # create 1-D pooling layer
        # https://keras.io/api/layers/pooling_layers
        # Pooling operation: [1] p. 68
        # Output Shape: (None, trunc(d2/pool_size), d3) = (d1,d4,d3)
        # poollay = MaxPooling1D(pool_size=4, strides=None, padding='valid')
        # self.add(poollay)

        # now arrived at output of Convolution-Detector-Pooling Building Block, [1], p.70

        # Output Shape: (None, d4 x d3) = (d1,d5)
        self.seq.add(Flatten())

        #  create dense layer
        #  https://keras.io/api/layers/core_layers/dense/
        # Output Shape: (None, numhidden1) = (d1,d6)
        # Param # = (d5 + 1) x d6  ; (+0, if no bias)
        self.seq.add(Dense(num_hidden, activation=activation_hidden))
        # self.seq.add(Dropout(0.5))

        # Output Shape: (None, dim_output) = (d1,d7)
        # Param # = (d6 + 1) x d7  ; (+0, if no bias)
        self.seq.add(Dense(dim_output, activation=activation_out))

        # https://keras.io/optimizers/#sgd
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)
        self.seq.compile(loss=loss, optimizer=optimizer)
        # print('Compiled.')

    def summary(self):
        """Print summary."""
        print('seq.summary():\n', self.seq.summary())
        config = self.seq.get_config()
        print('seq.get_config():\n', config)

    def fit(self, X_train, Y_train):
        """Wrap fit method."""
        # print('DEBUG: fit> dim=', self.dim, 'chan=', self.chan, 'batch_size=', self.batch_size, 'epochs=', self.epochs)
        self.seq.build(input_shape=(self.dim, self.chan))
        self.seq.fit(X_train, Y_train, batch_size=self.batch_size,
                     epochs=self.epochs, verbose=self.verbose)

    def predict(self, X):
        """Wrap fit predict."""
        return self.seq.predict(X)


'''
    def get_params(self, deep=True):
        """Wrap fit get_parm."""
        params = {'kernel_size', 'filters'}
        return params
'''



def test_fukunaga():

    from Fukunaga_Data import Fukunaga_Data
    from LabelBinarizer2 import LabelBinarizer2
    seed = None
    seed = 66649
    np.random.seed(seed=seed)
    num_samples_per_class = 50
    cv_method = 'KFOLD'
    K = 3
    cv_method = 'LOO'

    fd = Fukunaga_Data()
    dataset = 'I_I'
    dataset = 'I_LAMBDA'
    dataset = 'I_4I'
    fd.set_dataset(dataset)

    fd.gen_data(seed=seed, n=num_samples_per_class)
    print('Fukunaga data set=', fd.datasetname)
    # test_QDA(fd, fd.X, fd.y) # Resubstitution

    X = fd.X
    y = fd.y
    classname = ('Pos', 'Neg')

    print('Using Iris flower dataset ...')
    import sklearn.datasets as datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    classname = iris.target_names
    dim_output = len(classname)

    n_samples, dim = X.shape
    y = np.resize(y, (n_samples,))
    chan = 1
    X = np.resize(X, (X.shape[0], X.shape[1], 1))
    dim_output = len(np.unique(y))
    lb = LabelBinarizer2()
    Y = lb.fit_transform(y)
    # print('y=\n', y, '\nY=\n', Y); quit()

    kernel_size = 3    # size of the convolution filter (image would be e.g. tupel (3,3) )
    filters = 10  # number of convolution filters

    epochs = 100
    batch_size = 1
    verbose = 1

    model = CNN(dim=dim, chan=chan, dim_output=dim_output,
                filters=filters, kernel_size=kernel_size,
                batch_size=batch_size, epochs=epochs, verbose=verbose)

    print('model.summary():\n')
    model.summary()
    # quit(); # myexit()
    params = model.get_params()
    print('\n\nmodel=', model)
    print('params=', params)

    skf = StratifiedKFold(n_splits=K, shuffle=True)

    y_pred_overall = []
    y_test_overall = []
    k = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print('X_train.shape=', X_train.shape, 'X_test.shape=', X_test.shape)
        print('K-Fold: Fold #', k+1, 'of', K)

        # model.seq.build(input_shape=(model.dim, model.chan))
        # model.seq.fit(X_train, Y_train, batch_size=batch_size,
        #               epochs=epochs, verbose=verbose)http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

        model.fit(X_train, Y_train)

        y_pred = model.predict(X_test)
        y_pred_class = lb.inverse_transform(y_pred)
        # y_pred_class = model.seq.predict_classes(X_test)
        y_pred_overall = np.concatenate([y_pred_overall, y_pred_class])
        y_test_overall = np.concatenate([y_test_overall, y_test])
        k += 1

    print('CNN Classification Report: ')
    # print('y_test_overall=', y_test_overall,
    # 'y_pred_overall=', y_pred_overall, 'classname=', classname)
    print(classification_report(y_test_overall, y_pred_overall,
                                target_names=classname, digits=3))
    print('Accuracy=', '%.2f %%' %
          (100*accuracy_score(y_test_overall, y_pred_overall)))
    print('Macro-averaged F1=', '%.3f' %
          (f1_score(y_test_overall, y_pred_overall, average='macro')))
    print('Micro-averaged F1=', '%.3f' %
          (f1_score(y_test_overall, y_pred_overall, average='micro')))
    print('CNN Confusion Matrix: ')
    print(confusion_matrix(y_test_overall, y_pred_overall))
    return

    print('Executing main() ....')
    model = CNN(20, 4, 5, 6, 7)  # ; return
    params = model.get_params()
    print('params=', params)

    key_value_pair = {'activation': 'sigmoid'}
    model.set_params(**key_value_pair)

    params = model.get_params()
    print('params=', params)
    keys = model.get_params().keys()
    print('keys=', keys)

    model.build(input_shape=(model.dim, model.chan))



def has2have_GPU():
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


from scipy.special import expit, softmax
_actfunc = {'logistic': expit, 'tanh': np.tanh, 'softmax': softmax,
            'relu': lambda x: np.maximum(x, 0),
            'identity': lambda x: x, 'sign': np.sign}

def autoencode():
    seed = None
    seed = 66649
    np.random.seed(seed=seed)

    kernel_size = 3    # size of the convolution filter (image would be e.g. tupel (3,3) )
    filters = 10  # number of convolution filters

    epochs = 3
    batch_size = 1
    verbose = 1

    signalsrc = '/home/thomas/ninfabox/IMS/2nd_test/2004.02.12.12.12.39'
    signalsrc = '/export/thomas/IMS/2nd_test/2004.02.12.12.12.39'
    X = np.loadtxt(signalsrc)
    print('Training input shape: ', X.shape)

    n_samples, dim = X.shape
    chan = 1
    X = np.resize(X, (X.shape[0], X.shape[1], 1))
    dim_output = dim

    model = CNN(dim=dim, chan=chan, dim_output=dim_output,
                filters=filters, kernel_size=kernel_size,
                batch_size=batch_size, epochs=epochs, verbose=verbose)

    print('model.summary():\n')
    model.summary()
    # quit(); # myexit()
    params = model.get_params()
    print('\n\nmodel=', model)
    print('params=', params)
    model.fit(X, X)


def autoencoder1():
    datadir = '/home/thomas/ninfabox/IMS/2nd_test/'
    datadir = '/export/thomas/IMS/2nd_test/'
    signalfiles = ('2004.02.12.12.12.39', '2004.02.12.12.22.39',
                   '2004.02.12.12.32.39', '2004.02.12.12.42.39', )
    n = len(signalfiles)

    output = []
    for i in range(n):
        signalsrc = datadir + signalfiles[i]
        Xi = np.loadtxt(signalsrc)
        output.append(Xi)
    X = np.stack(output)
    print('Training input shape: ', X.shape)

    n_samples, dim_input, chan = X.shape
    # X = np.resize(X, (X.shape[0], X.shape[1], 1))
    # dim_output = dim_input

    model = keras.Sequential(
        [
            layers.Input(shape=(dim_input, chan)),
            layers.Conv1D(
                filters=32, kernel_size=7, padding='same', strides=2,
                activation='relu'
            ),
            # https://stackoverflow.com/questions/51542442/what-is-the-default-stride-length-in-keras-conv1d
            #  output_dim = 1 + (input_dim - kernel_size)/stride
            # If instead you want to preserve the input dimensionality at each
            # convolutional layer, setting padding='same' results in padding
            # the input such that the output has the same length as
            # the original input.
            #  output_dim = (input_dim)/stride
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding='same', strides=2,
                activation='relu'
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding='same', strides=2,
                activation='relu', name='bottleneck'
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding='same', strides=2,
                activation='relu'
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding='same'),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    optimizer = keras.optimizers.SGD(learning_rate=0.01)
    loss = 'mse'
    loss = 'kl_divergence'
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    history = model.fit(
        X,
        X,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=5, mode='min')
        ],
    )

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()


    #  https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction

    layer_name = 'bottleneck'
    extractor = keras.Model(inputs=model.input,
                            outputs=model.get_layer(layer_name).output)
    extracted = extractor(X)
    print('Autoencode extraction:', extracted)

"""
    kernel_size = 3    # size of the 1-D convolution filter
    filters = 10  # number of convolution filters

    epochs = 3
    batch_size = 1
    verbose = 1

    seq = Sequential(name='AutoEncoder1')
    input_shape = (dim, chan)

    convlay1 = Conv1D(name='convlay1', input_shape=input_shape,
                     kernel_size=kernel_size, filters=filters,
                     use_bias=True)
    seq.add(convlay1)
    #seq.add(Flatten())

    dense1 = Dense(units=8, activation=tf.keras.activations.sigmoid)
    seq.add(dense1)

    dense2 = Dense(units=8, activation=tf.keras.activations.sigmoid)
    seq.add(dense2)


    print('model.summary():\n')
    seq.summary()
    config = seq.get_config()
    print('seq.get_config():\n', config)

    # https://keras.io/api/optimizers/
    optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False,
                    name="SGD")
    # https://keras.io/losses
    loss = 'mean_squared_error'
    # https://en.wikipedia.org/wiki/Cross_entropy
    loss = 'categorical_crossentropy'

    seq.compile(loss=loss, optimizer=optimizer)
    print('Compiled.')
"""


def main():
    test_fukunaga(); raise Exception('...')
    autoencoder1(); raise Exception('...')
    autoencode(); raise Exception('...')


if __name__ == "__main__":
    # has2have_GPU()
    main()
