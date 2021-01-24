from keras.models import Model, Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten, Input

#def rnn_vector_output(X_train, y_train, X_valid, y_valid, RECURRENT_MODEL, LATENT_DIM, BATCH_SIZE, EPOCHS, LAG, HORIZON, loss, optimizer, file_header):
    #X_train = X_train.values.reshape( (X_train.shape[0], X_train.shape[1], 1) )
    #X_valid = X_valid.values.reshape( (X_valid.shape[0], X_valid.shape[1], 1) )
def rnn_vector_output(train_inputs, valid_inputs, RECURRENT_MODEL, LATENT_DIM, BATCH_SIZE, EPOCHS, LAG, HORIZON, 
                    loss, optimizer, earlystop, best_val, verbose):

    model = Sequential()
    model.add(RECURRENT_MODEL(LATENT_DIM, input_shape=(LAG, 1)))
    model.add(Dense(HORIZON))

    model.compile(optimizer=optimizer, loss= loss)


    # history = model.fit(X_train,
    #                     y_train,
    #                     batch_size=BATCH_SIZE,
    #                     epochs=EPOCHS,
    #                     validation_data=(X_valid, y_valid),
    #                     callbacks=[earlystop, best_val],
    #                     verbose=1)
    history = model.fit(train_inputs['encoder_input'], train_inputs['target'],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(valid_inputs['encoder_input'], valid_inputs['target']),
        callbacks=[earlystop, best_val],
        verbose=verbose)
    return model, history
