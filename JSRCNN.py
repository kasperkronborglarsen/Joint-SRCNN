from ModelHelper import loadModel
from ImageProcessing import saveImages, getImageMatrix, createTrainingImages, getResizedImageMatrix

#%%
path_HP_GT_input = r'%systemdrive%:/users/%username%/%HP_GT_image_folder%'
path_HP_LR_input = r'%systemdrive%:/users/%username%/%HP_LR_image_folder%'
path_proton_input = r'%systemdrive%:/users/%username%/%proton_image_folder%'
path_output = r'%systemdrive%:/users/%username%/%output_image_folder%'

hp_images_LR = createTrainingImages(path_HP_GT_input, path_HP_LR_input)
hp_images = getImageMatrix(path_HP_LR_input)
proton_images = getImageMatrix(path_proton_input)
img_rows = hp_images.shape[2]
img_cols = hp_images.shape[3]
    
modelName = 'models/JSRCNN_v2'
model = loadModel(modelName)

predicted_images = model.predict([proton_images, hp_images])

#%%
saveImages(path_output, predicted_images)