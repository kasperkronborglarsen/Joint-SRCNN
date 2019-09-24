from keras.models import Model
from keras.layers import Add, Input, Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras import backend as K
import keras.optimizers as optimizers

K.set_image_dim_ordering('th')

#%%
def PSNR(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

#%%
def SRCNN(img_rows, img_cols):
    input_shape = Input(shape=(1, img_rows, img_cols))    
    
    c1 = Conv2D(64, (5, 5), activation='relu', padding='same')(input_shape)
    c2 = Conv2D(64, (5, 5), activation='relu', padding='same')(c1)
    
    c2_2 = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(c2)
    x1 = Add()([c2, c2_2])
    
    c1_2 = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(x1)
    x2 = Add()([c1, c1_2])
    
    predicted_images = Conv2D(1, (5, 5), activation='linear', padding='same')(x2)
    
    model = Model(input_shape, predicted_images)
    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[PSNR])
    
    return model

#%%
def SRCNN_Untrained(img_rows, img_cols, neurons_1, neurons_2, filter_size1, filter_size2, filter_size3, learning_rate, momentum, optimizer_algorithm, epochs, batch_size):
    input_shape = Input(shape=(1, img_rows, img_cols))
    
    c1 = Conv2D(neurons_1, (filter_size1, filter_size1), activation='relu', padding='same')(input_shape)
    c2 = Conv2D(neurons_2, (filter_size2, filter_size2), activation='relu', padding='same')(x1)
    
    c2_2 = Conv2DTranspose(neurons_2, (filter_size2, filter_size2), activation='relu', padding='same')(x2)
    x1 = Add()([c2, c2_2])
    
    c1_2 = Conv2DTranspose(neurons_1, (filter_size1, filter_size1), activation='relu', padding='same')(x1)
    x2 = Add()([c1, c1_2])
    
    predicted_images = Conv2D(1, (filter_size3, filter_size3), activation='linear', padding='same')(x2)
    
    model = Model(input_shape, predicted_images)    
    model.compile(optimizer=optimizer_algorithm, loss='mse', metrics=[PSNR])
    
    return model

#%%
def SRCNNMaxpool(img_rows, img_cols):
    input_shape = Input(shape=(1, img_rows, img_cols))
    
    c1 = Conv2D(64, (5, 5), activation='relu', padding='same')(input_shape)
    c1 = Conv2D(64, (5, 5), activation='relu', padding='same')(c1)

    x = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    c2 = Conv2D(64, (5, 5), activation='relu', padding='same')(c2)

    x = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (5, 5), activation='relu', padding='same')(x)

    x = UpSampling2D()(c3)

    c2_2 = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    c2_2 = Conv2D(64, (5, 5), activation='relu', padding='same')(c2_2)

    m1 = Add()([c2, c2_2])
    m1 = UpSampling2D()(m1)

    c1_2 = Conv2D(64, (5, 5), activation='relu', padding='same')(m1)
    c1_2 = Conv2D(64, (5, 5), activation='relu', padding='same')(c1_2)

    m2 = Add()([c1, c1_2])

    predicted_images = Conv2D(1, (5, 5), activation='linear', border_mode='same')(m2)
    
    model = Model(input_shape, predicted_images)
    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[PSNR])
    
    return model

#%%    
def JSRCNN(img_rows, img_cols):
    input_shape_proton = Input(shape=(1, img_rows, img_cols))
    
    c1_1_proton = Conv2D(64, (5, 5), activation='relu', padding='same')(input_shape_proton)
    c2_1_proton = Conv2D(64, (5, 5), activation='relu', padding='same')(c1_1_proton)

    c2_2_proton = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(c2_1_proton)
    c2_proton = Add()([c2_1_proton, c2_2_proton])
    
    c1_2_proton = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(c2_proton)
    decoded_proton = Add()([c1_1_proton, c1_2_proton])
    
    
    input_shape_hp = Input(shape=(1, img_rows, img_cols))
    
    c1_1_hp = Conv2D(64, (5, 5), activation='relu', padding='same')(input_shape_hp)
    c2_1_hp = Conv2D(64, (5, 5), activation='relu', padding='same')(c1_1_hp)

    c2_2_hp = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(c2_1_hp)
    c2_hp = Add()([c2_1_hp, c2_2_hp])
    
    c1_2_hp = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(c2_hp)
    decoded_hp = Add()([c1_1_hp, c1_2_hp])
    
    
    merged_vector = Add()([decoded_hp, decoded_proton])
    
    
    c1_1_joint = Conv2D(64, (5, 5), activation='relu', padding='same')(merged_vector)
    c2_1_joint = Conv2D(64, (5, 5), activation='relu', padding='same')(c1_1_joint)

    c2_2_joint = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(c2_1_joint)
    c2_joint = Add()([c2_1_joint, c2_2_joint])
    
    c1_2_joint = Conv2DTranspose(64, (5, 5), activation='relu', padding='same')(c2_joint)
    c1_joint = Add()([c1_1_joint, c1_2_joint])
    
    predicted_sr_hp = Conv2D(1, (5, 5), activation='linear', padding='same')(c1_joint)
    
        
    model = Model([input_shape_proton, input_shape_hp], predicted_sr_hp)
    adam = optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='mse', metrics=[PSNR])
    
    return model