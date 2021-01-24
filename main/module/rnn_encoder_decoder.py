from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten, Input

def rnn_encoder_decoder(train_inputs, valid_inputs, RECURRENT_MODEL, LATENT_DIM, BATCH_SIZE, EPOCHS, LAG, HORIZON, 
                        loss, optimizer, earlystop, best_val, verbose = 0):
        
        model = Sequential()
        model.add(RECURRENT_MODEL(LATENT_DIM, input_shape=(LAG, 1)))
        model.add(RepeatVector(HORIZON))
        model.add(RECURRENT_MODEL(LATENT_DIM, return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.add(Flatten())

        model.compile(optimizer=optimizer, loss=loss)

        history = model.fit(train_inputs['encoder_input'],
                train_inputs['target'],
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(valid_inputs['encoder_input'], valid_inputs['target']),
                callbacks=[earlystop, best_val],
                verbose=verbose)
        return model, history
