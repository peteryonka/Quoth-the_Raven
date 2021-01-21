
# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function invoked for specified epochs. Prints generated text.
def on_epoch_end(epoch, logs):
    # Using epoch+1 to be consistent with the training epochs printed by Keras
    if epoch+1 == 1 or epoch+1 == 15:
        print()
        print('----- Generating text after Epoch: %d' % epoch)        start_index = random.randint(0, len(user) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)
            generated = ''
            sentence = user[start_index: start_index + maxlen]            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')            sys.stdout.write(generated)
            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.
                    preds = model.predict(x_pred, verbose=0)[0]                next_index = sample(preds, diversity)                next_char = indices_char[next_index]                generated += next_char
                    sentence = sentence[1:] + next_char                sys.stdout.write(next_char)                sys.stdout.flush()
        print()
    else:
        print()
        print('----- Not generating text after Epoch: %d' % epoch)
        generate_text = LambdaCallback(on_epoch_end=on_epoch_end)
