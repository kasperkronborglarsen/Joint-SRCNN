from ImageProcessing import getTrainingImages
from Models import SRCNN
from ModelHelper import saveModel
import matplotlib.pyplot as plt

#%%
path_GT = r'%systemdrive%:/users/%username%/%GT_image_folder%'
path_LR = r'%systemdrive%:/users/%username%/%LR_image_folder%'

X_train, X_test, Y_train, Y_test = getTrainingImages(path_GT, path_LR)

img_rows = X_train.shape[2]
img_cols = X_train.shape[3] 

#%%
modelName = 'models/SRCNN_HP_275_128x128'
model = SRCNN(img_rows, img_cols)

history = model.fit(X_train, Y_train,
                    batch_size=64,
                    epochs=100,
                    validation_data=(X_test,Y_test),
                    verbose=1)
 
score = model.evaluate(X_test, Y_test)

print('Loss: %.3f, PSNR: %.3f' % (score[0], score[1]))

#%%
plt.plot(history.history['PSNR'])
plt.plot(history.history['val_PSNR'])
plt.title('model PSNR')
plt.ylabel('PSNR')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
saveModel(model, modelName)