import ImageProcessing
import Models
import time
from keras.optimizers import SGD, Adam

#%%
X_train, X_test, Y_train, Y_test = ImageProcessing.main()

img_rows = X_train.shape[2]
img_cols = X_train.shape[3] 

file = open('hyperparameters.txt','w')
run_time = time.time()

for neurons_1 in [64]:
    for neurons_2 in [64]:
        for filter_size_1 in [5]:
            for filter_size_2 in [5]:
                for filter_size_3 in [5]:
                    for activation in ['relu']:
                        for epochs in [100]:
                            for learning_rate in [0.1, 0.01, 0.001]:
                                for momentum in [0.1, 0.5, 0.9]:                                    
                                    adam = Adam(lr=learning_rate)
                                    sgd = SGD(lr=learning_rate, momentum=momentum)
                                            
                                    for batch_size in [64]:
                                        for optimizer_algorithm_str in ['sgd', 'adam']:
                                            if optimizer_algorithm_str == 'sgd':
                                                optimizer_algorithm = sgd
                                            else:
                                                optimizer_algorithm = adam                                        
                                           
                                                start_time = time.time()
                                                model = Models.SRCNN_Untrained(img_rows, img_cols, neurons_1, neurons_2, filter_size_1, 
                                                                           filter_size_2, filter_size_3, learning_rate, momentum, 
                                                                           optimizer_algorithm, epochs, batch_size)
                                            
                                            
                                                history = model.fit(X_train, Y_train,
                                                                batch_size=batch_size,
                                                                epochs=epochs,
                                                                validation_data=(X_test,Y_test),
                                                                verbose=1)
                                                score = model.evaluate(X_test, Y_test)
                                                total_time = time.time() - start_time
                                                file.write('Loss: %.3f, PSNR: %.3f with: # neurons_1: %i, # neurons_2: %i,'
                                                        ' filter size_1: %s, filter size_2: %s, filter size_3: %s,'
                                                        ' activation: %s, # epochs: %i, learning_rate: %.4f, momentum: %.4f,'
                                                        ' batch size: %i, optimizer: %s, elapsed time: %.3f \n'
                                                        % (score[0], score[1], neurons_1, neurons_2, 
                                                           str(filter_size_1) + 'x' + str(filter_size_1), 
                                                           str(filter_size_2) + 'x' + str(filter_size_2),
                                                           str(filter_size_3) + 'x' + str(filter_size_3),
                                                           activation, epochs, learning_rate, momentum,
                                                           batch_size, optimizer_algorithm_str, total_time))
                                                print('Loss: %.3f, PSNR: %.3f with: # neurons_1: %i, # neurons_2: %i,'
                                                   ' filter size_1: %s, filter size_2: %s, filter size_3: %s,'
                                                   ' activation: %s, # epochs: %i, learning_rate: %.4f,'
                                                   ' momentum: %.4f, batch size: %i, optimizer: %s, elapsed time: %.3f \n'
                                                   % (score[0], score[1], neurons_1, neurons_2,
                                                       str(filter_size_1) + 'x' + str(filter_size_1),
                                                       str(filter_size_2) + 'x' + str(filter_size_2),
                                                       str(filter_size_3) + 'x' + str(filter_size_3),
                                                       activation, epochs, learning_rate, momentum, batch_size,
                                                       optimizer_algorithm_str, total_time))
total_run_time = time.time() - run_time
file.write('Total time: %.3f' % total_run_time)
file.close()