import numpy as np
import time
from keras.preprocessing.image import save_img
from keras import backend as K
from JsonHelper import loadModel

modelName = 'model'
model = loadModel(modelName)

img_width = 128
img_height = 128

model.summary()

layer_name = 'conv2d_transpose_2' #'conv2d_1', 'conv2d_2', 'conv2d_transpose_1', 'conv2d_transpose_2', 'conv2d_3'

#%%
def deprocessImage(x):
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#%%
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

#%%
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
kept_filters = []

for filter_index in range(64):
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    gradients = K.gradients(loss, input_img)[0]
    gradients = normalize(gradients)

    iterate = K.function([input_img], [loss, gradients])

    step = 1.

    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 1, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    for i in range(20):
        loss_value, gradients_value = iterate([input_img_data])
        input_img_data += gradients_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            break
            
    if loss_value > 0.:
        img = deprocessImage(input_img_data[0])
        kept_filters.append((img, loss_value))
        img_transposed = img.transpose(2, 1, 0)
        save_img('%s_filter_%d.png' % (layer_name, filter_index), img_transposed)
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))