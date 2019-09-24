from ImageProcessing import getJointTrainingImages
import Models
from ModelHelper import saveModel
import matplotlib.pyplot as plt

#%%
path_HP = r'%systemdrive%:/users/%username%/%HP_image_folder%'
path_proton = r'%systemdrive%:/users/%username%/%proton_image_folder%'
path_HP_GT = r'%systemdrive%:/users/%username%/%HP_GT_image_folder%'
size = 128

X_hp_train, X_hp_test, X_proton_train, X_proton_test, Y_train, Y_test = getJointTrainingImages(path_HP, path_proton, path_HP_GT, size)
img_rows = X_hp_train.shape[2]
img_cols = X_hp_train.shape[3]

#%%
modelName = 'models/x'
model = Models.JSRCNN(img_rows, img_cols)

history = model.fit([X_proton_train, X_hp_train], Y_train,
                    batch_size=64,
                    epochs=100,
                    validation_data=([X_proton_test, X_hp_test], Y_test),
                    verbose=1)
 
score = model.evaluate([X_proton_test, X_hp_test], Y_test)

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