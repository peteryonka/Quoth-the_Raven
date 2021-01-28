import numpy as np
import os
from pickle import dump
from contextlib import redirect_stdout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session

clear_session()
# setting a random seed for reproducibility
np.random.seed(2021)

def load_sequences(path_and_filename):
    sequence_data = open(path_and_filename).read()
    sequences = sequence_data.split('\n')

    words_in_seq = len(sequences[0].split()) - 1

    print(f'{len(sequences)} sequences have been loaded.')
    print(f'Each sequence has {words_in_seq} word token(s) plus an output token.')
    return sequences, words_in_seq

sequence_list, seq_length = load_sequences('./cleaned_poe_tot_seq_len_26.txt')

# map words to integers for each sequence
def tokenize_words(sequence_list, filter_string='', lower_case=True):

    tokenizer = Tokenizer(filters=filter_string, lower=lower_case)

    tokenizer.fit_on_texts(sequence_list)

    sequences = tokenizer.texts_to_sequences(sequence_list)

    vocabulary_size = len(tokenizer.word_index)

    print(f'Sequences have been tokenized using Keras API Tokenizer.')
    print(f'Vocabulary size is {vocabulary_size}')

    return tokenizer, sequences, vocabulary_size


tokenizer, sequences, vocab_size = tokenize_words(sequence_list, filter_string='', lower_case=False)

# function to create our independent (X) and dependent variables (y)
def input_and_output_sequences(sequences, vocab_size):
    sequences = np.array(sequences)
    X, y = sequences[:,:-1], sequences[:, -1]
    y = utils.to_categorical(y, num_classes = vocab_size+1) # plus one required due to 0-offset of array
    return X, y


X, y = input_and_output_sequences(sequences, vocab_size)

# build our LSTM model
def build_LSTM_model(vocab_size, seq_length, layer_size=256, embedding=True, embedding_vector_space=128, dropout=True, dropout_rate=0.2):

    model = Sequential()

    if embedding:
        model.add(Embedding(input_dim=vocab_size+1, output_dim=embedding_vector_space, input_length=seq_length))
        model.add(LSTM(layer_size, return_sequences=True))
    else:
        model.add(LSTM(layer_size, input_shape = (seq_length, vocab_size+1), return_sequences=True))

    if dropout:
        model.add(Dropout(dropout_rate))

    model.add(LSTM(layer_size))

    if dropout:
        model.add(Dropout(dropout_rate))

    model.add(Dense(layer_size, activation='relu'))

    model.add(Dense(vocab_size+1, activation='softmax'))

    print(f"Model has been created.\n\nHere's a summary:")
    print(f'----------------------')
    print(model.summary())

    model_name = f'{seq_length}_seqlen_LSTM_model'


    return model, model_name

model, model_name = build_LSTM_model(vocab_size, seq_length)


# create checkpoints to save model weights (if an improvement) at each epoch
# FILES ARE LARGE! 110MB FOR CURRENT MODEL WITH 9M PARAMS -- Adjust for your needs
if not os.path.exists(f'./Model_weights_{model_name}'):
    os.mkdir(f'./Model_weights_{model_name}')

checkpoint_path = f'./Model_weights_{model_name}/{model_name}_weights' + '-improvement-{epoch:02d}-{loss:.4f}-acc{accuracy:.4f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model -- don't forget to adjust if you don't want to include the callback to save after each epoch if model improves
model.fit(X,y, batch_size=64, epochs=100, callbacks=callback_list)

# save model, model summary, tokenizer
if not os.path.exists(f'./Models_{model_name}'):
    os.mkdir(f'./Models_{model_name}')

with open(f'./Models_{model_name}/{model_name}_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

model.save(f'./Models_{model_name}/{model_name}_word_model.h5')

dump(tokenizer, open(f'./Models_{model_name}/{model_name}_tokenizer.pkl', 'wb'))
