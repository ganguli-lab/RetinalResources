

from __future__ import print_function
from __future__ import print_function
import os
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Layer, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, GaussianNoise, UpSampling2D, Input, LocallyConnected2D, ZeroPadding2D, Lambda
from keras import backend as K
from keras import metrics
from keras.models import Model
import numpy as np
import sys
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
import keras
from keras.layers import Layer, Activation
from keras import metrics
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import argparse


NX = 32
NY = 32
NC = 1
img_rows, img_cols, img_chns = NX, NY, NC

load_dir = os.path.join(os.getcwd(), 'saved_models')



img_width = 32
img_height = 32

K.set_learning_phase(1)

def deprocess_image(x):
    x -= x.mean()
    if (x.std() > 1e-5):
        x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


batch_size = 32
num_classes = 10
epochs = 0
data_augmentation = True
num_predictions = 20
batch_norm = 0

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--trial_label', default='Trial1',
                    help='For labeling different runs of the same model')
parser.add_argument('--noise_start', type=float, default=0.0,
                    help='Input noise')
parser.add_argument('--noise_end', type=float, default=0.0,
                    help='Retinal output noise')
parser.add_argument('--retina_out_weight_reg', type=float, default=0.0,
                    help='L1 regularization on retinal output weights')
parser.add_argument('--reg', type=float, default=0.0,
                    help='L1 weight regularization for layers besides the retinal output layer')
parser.add_argument('--retina_hidden_channels', type=int, default=32,
                    help='Channels in hidden layers of retina')
parser.add_argument('--retina_out_stride', type=int, default=1,
                    help='Stride at output layer of retina')
parser.add_argument('--task', default='classification',
                    help='e.g. classification or reconstruction')
parser.add_argument('--filter_size', type=int, default=9,
                    help='Convolutional filter size')
parser.add_argument('--retina_layers', type=int, default=2,
                    help='Number of layers in retina')
parser.add_argument('--vvs_layers', type=int, default=2,
                    help='Number of convolutional layers in VVS')
parser.add_argument('--use_b', type=int, default=1,
                    help='Whether or not to use bias terms in retinal output layer')
parser.add_argument('--actreg', type=float, default=0.0,
                    help='L1 regularization on retinal output')
parser.add_argument('--retina_out_width', type=int, default=1,
                    help='Number of output channels in Retina')
parser.add_argument('--vvs_width', type=int, default=32,
                    help='Number of output channels in VVS layers')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train model')
parser.add_argument('--layer_name', default=None,
                    help='Keras model name of layer being visualized')




args = parser.parse_args()


trial_label = args.trial_label
noise_start = args.noise_start
noise_end = args.noise_end
retina_out_weight_reg = args.retina_out_weight_reg
retina_hidden_channels = args.retina_hidden_channels
retina_out_stride = args.retina_out_stride
task = args.task
filter_size = args.filter_size
retina_layers = args.retina_layers
vvs_layers = args.vvs_layers
use_b = args.use_b
actreg = args.actreg
retina_out_width = args.retina_out_width
vvs_width = args.vvs_width
epochs = args.epochs
reg = args.reg
layer_name = args.layer_name


model_name = 'cifar10_type_'+trial_label+'_noise_start_'+str(noise_start)+'_noise_end_'+str(noise_end)+'_reg_'+str(reg)+'_retina_reg_'+str(retina_out_weight_reg)+'_retina_hidden_channels_'+str(retina_hidden_channels)+'_SS_'+str(retina_out_stride)+'_task_'+task+'_filter_size_'+str(filter_size)+'_retina_layers_'+str(retina_layers)+'_vvs_layers'+str(vvs_layers)+'_bias_'+str(use_b)+'_actreg_'+str(actreg)+'_retina_out_channels_'+str(retina_out_width)+'_vvs_width_'+str(vvs_width)+'_epochs_'+str(epochs)
model_name = 'SAVED'+'_'+model_name

model_path = os.path.join(load_dir, model_name)

if use_b == 1:
    use_b = True
else:
    use_b = False




(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.mean(x_train, 3, keepdims=True)
x_test = np.mean(x_test, 3, keepdims=True) 
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




filters = 64
NX = 32
NY = 32
NC = 1
img_rows, img_cols, img_chns = NX, NY, NC
num_conv = 3
intermediate_dim = 1024

x = Input(shape=x_train[0].shape)
gn = GaussianNoise(noise_start)(x)
if retina_layers > 2:
    conv1_nonlin = Conv2D(retina_hidden_channels, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg),  padding='same', name='retina_1', activation='relu', input_shape=x_train.shape[1:])(gn)
    retina_out = Conv2D(retina_hidden_channels, (filter_size, filter_size),  kernel_regularizer=keras.regularizers.l1(reg), padding='same',  activation='relu',  name='retina_2', trainable=True)(conv1_nonlin)
    for iterationX in range(retina_layers - 2):
        if iterationX == retina_layers - 3:
            retina_out = Conv2D(retina_out_width, (filter_size, filter_size), strides=(retina_out_stride,retina_out_stride), kernel_regularizer=keras.regularizers.l1(retina_out_weight_reg), activity_regularizer=keras.regularizers.l1(actreg), padding='same', name='retina_'+str(iterationX+3), activation='relu', use_bias=use_b)(retina_out)
        else:
            retina_out = Conv2D(retina_hidden_channels, (filter_size, filter_size),  kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='retina_'+str(iterationX+3), activation='relu')(retina_out)



if retina_layers == 2:
    conv1_nonlin = Conv2D(retina_hidden_channels, (filter_size, filter_size),  kernel_regularizer=keras.regularizers.l1(reg), padding='same', input_shape=x_train.shape[1:], name='retina_1', activation='relu', trainable=True)(gn)

    retina_out = Conv2D(retina_out_width, (filter_size, filter_size), strides=(retina_out_stride,retina_out_stride), kernel_regularizer=keras.regularizers.l1(retina_out_weight_reg), padding='same',  activation='relu', activity_regularizer=keras.regularizers.l1(actreg), use_bias=use_b, name='retina_2', trainable=True)(conv1_nonlin)


elif retina_layers == 1:
    retina_out = Conv2D(retina_out_width, (filter_size, filter_size), strides=(retina_out_stride,retina_out_stride), kernel_regularizer=keras.regularizers.l1(specalreg), activity_regularizer=keras.regularizers.l1(actreg), padding='same', input_shape=x_train.shape[1:], use_bias = use_b, name='retina_1', activation='relu', trainable=True)(gn)

elif retina_layers == 0:
    retina_out = gn


if noise_end > 0:
    retina_out = GaussianNoise(noise_end)(retina_out)


if vvs_layers > 2:
    vvs_1_layer = Conv2D(vvs_width, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='vvs_1', activation='relu')
    vvs_1 = vvs_1_layer(retina_out)
    vvs_2 = Conv2D(vvs_width, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='vvs_2', activation='relu')(vvs_1)
    for iterationX in range(vvs_layers - 2):
        vvs_2 = Conv2D(vvs_width, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='vvs_'+str(iterationX+3), activation='relu')(vvs_2)
    flattened = Flatten()(vvs_2)

if vvs_layers == 2:
    vvs_1_layer = Conv2D(vvs_width, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='vvs_1', activation='relu', trainable=True)
    vvs_1 = vvs_1_layer(retina_out)

    vvs_2 = Conv2D(vvs_width, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='vvs_2', activation='relu', trainable=True)(vvs_1)
    
    flattened = Flatten()(vvs_2)

elif vvs_layers == 1:
    vvs_1_layer = Conv2D(vvs_width, (filter_size, filter_size), kernel_regularizer=keras.regularizers.l1(reg), padding='same', name='vvs_1', activation='relu')
    vvs_1 = vvs_1_layer(retina_out)
    flattened = Flatten()(vvs_1)

elif vvs_layers == 0:
    flattened = Flatten()(retina_out)


hidden = Dense(intermediate_dim, kernel_regularizer=keras.regularizers.l1(reg), name='dense1', activation='relu', trainable=True)(flattened)

output = Dense(num_classes, name='dense2', activation='softmax', trainable=True)(hidden)



# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

if task == 'classification':
    model = Model(x, output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                   metrics=['accuracy'])

else:
    sys.exit("No other task types besides classification configured yet")

from keras.models import load_model
model = load_model(model_path)

model.summary()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



input_img = model.input

layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
print('Layer Options', layer_dict.keys())
input_img = layer_dict['input_1'].output
print(layer_dict)

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

v1_weights = vvs_1_layer.get_weights()
np.save('saved_weights/V1W_'+model_name+'.npy', v1_weights)

kept_filters = []
RFs = []
for filter_index in range(layer_dict[layer_name].output.shape[3]):
    print(layer_dict[layer_name].output.shape)

    print('Processing filter %d' % filter_index)
    start_time = time.time()

    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        image_rep_size_x = layer_output.shape[2]
        image_rep_size_y = layer_output.shape[3]
        loss = K.mean(layer_output[:, filter_index, image_rep_size_x//2, image_rep_size_y//2])
    else:
        image_rep_size_x = layer_output.shape[1]
        image_rep_size_y = layer_output.shape[2]
        loss = K.mean(layer_output[:, image_rep_size_x//2, image_rep_size_y//2, filter_index])

    grads = K.gradients(loss, input_img)[0]
    grads = normalize(grads)
    iterate = K.function([input_img], [loss, grads])

    layer_out_func = K.function([input_img], [layer_output])

    step = 1.

    # start from blank image
    if K.image_data_format() == 'channels_first':
        input_img_data = 0.0*np.ones((1, 1, img_width, img_height))
    else:
        input_img_data = 0.0*np.ones((1, img_width, img_height, 1))


    # we run gradient ascent for 1 step so it's just a computation of the gradient
    loss_value, grads_value = iterate([input_img_data])
    layerout = layer_out_func([input_img_data])[0]

    input_img_data += grads_value * step
    RFs.append(input_img_data[0])


    img = deprocess_image(input_img_data[0])

    kept_filters.append((img, loss_value))
    end_time = time.time()


n = 5# visualization grid size

# pick filters with highest loss
kept_filters.sort(key=lambda x: x[1], reverse=True)

np.save('saved_filters/FIL_'+model_name+'_'+str(layer_name)+'.npy', RFs)


kept_filters = kept_filters[:n * n]

margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))
for i in range(n):
    for j in range(n):
        try:
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
        except:
            pass #not enough RFs to fill the grid, this is fine

imsave('saved_visualizations/VIS_'+model_name+'_'+str(layer_name)+'.png', stitched_filters)






