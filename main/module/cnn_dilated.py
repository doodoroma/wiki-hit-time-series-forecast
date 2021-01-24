from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten, Input

#def cnn_dilated(X_train, y_train, X_valid, y_valid, LATENT_DIM, KERNEL_SIZE, BATCH_SIZE, EPOCHS, LAG, HORIZON, 
#                loss, optimizer, earlystop, best_val):

def cnn_dilated(train_inputs, valid_inputs, LATENT_DIM, KERNEL_SIZE, BATCH_SIZE, EPOCHS, LAG, HORIZON, 
                loss, optimizer, earlystop, best_val, verbose = 0):

    #X_train = X_train.values.reshape( (X_train.shape[0], X_train.shape[1], 1) )
    #X_valid = X_valid.values.reshape( (X_valid.shape[0], X_valid.shape[1], 1) )

    model = Sequential()
    model.add(Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=1, input_shape=(LAG, 1)))
    model.add(Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=2))
    model.add(Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=4))
    model.add(Flatten())
    model.add(Dense(HORIZON, activation='linear'))

    model.compile(optimizer=optimizer, loss=loss)

    # history = model.fit(X_train,
    #                     y_train,
    #                     batch_size=BATCH_SIZE,
    #                     epochs=EPOCHS,
    #                     validation_data=(X_valid, y_valid),
    #                     callbacks=[earlystop, best_val],
    #                     verbose=1)
    
    history = model.fit(train_inputs['encoder_input'],
                train_inputs['target'],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(valid_inputs['encoder_input'], valid_inputs['target']),
                    callbacks=[earlystop, best_val],
                    verbose=verbose)
    return model, history
