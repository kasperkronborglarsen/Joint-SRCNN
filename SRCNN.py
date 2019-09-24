from ModelHelper import loadModel
from ImageProcessing import saveImages, getImageMatrix

#%%
path_input = r'%systemdrive%:/users/%username%/%input_image_folder%'
path_output = r'%systemdrive%:/users/%username%/%SR_image_folder%'

images = getImageMatrix(path_input)
img_rows = images.shape[2]
img_cols = images.shape[3]

modelName = 'models/SRCNN_HP_275_128x128'
model = loadModel(modelName)

predicted_images = model.predict(images)

#%%
saveImages(path_output, predicted_images)