from keras.models import model_from_json

#%%
def saveModel(model, modelName):
    # serialize model to JSON
    model_json = model.to_json()
    with open(modelName + '.json', 'w') as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(modelName + '.h5')

#%%      
def loadModel(modelName):
    # load json and create model
    json_file = open(modelName + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(modelName + '.h5')
    
    return model