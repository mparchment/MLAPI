from flask import Flask, jsonify
from keras.models import load_model
import numpy as np
import pickle

encoder_model = load_model('model/en_model.h5', compile=False)
decoder_model = load_model('model/de_model.h5', compile=False)
input_token_index = pickle.load(open('model/input_token_index.pkl', 'rb'))
target_token_index = pickle.load(open('model/target_token_index.pkl', 'rb'))
max_decoder_seq_length = pickle.load(open('model/max_decoder_seq_length.pkl', 'rb'))
max_encoder_seq_length = pickle.load(open('model/max_encoder_seq_length.pkl', 'rb'))
num_decoder_tokens = pickle.load(open('model/num_decoder_tokens.pkl', 'rb'))
num_encoder_tokens = pickle.load(open('model/num_encoder_tokens.pkl', 'rb'))


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    reverse_target_char_index = {v: k for k, v in target_token_index.items()}

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def transliterate(word):
    encoded_word = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')

    for t, char in enumerate(word):
        encoded_word[0, t, input_token_index[char]] = 1.
    encoded_word[0, t + 1:, input_token_index[' ']] = 1.

    return decode_sequence(encoded_word)


app = Flask(__name__)


@app.route('/toarabic/<word>', methods=['GET'])
def return_word(word):
    return jsonify({'word': transliterate(word)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000)