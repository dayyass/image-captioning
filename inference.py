import io
import json
import os

import numpy as np
import tensorflow as tf
from googletrans import Translator
from PIL import Image
from telebot import TeleBot
from tensorflow.contrib import keras

import utils
from keras_utils import reset_tf_session

# HYPER-PARAMETERS

VOCAB_PATH = "./model/vocab.json"
PATH_TO_MODEL = "./model/weights"

IMG_SIZE = 299
IMG_EMBED_SIZE = 2048
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120

PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"


# VOCAB

with open(VOCAB_PATH, mode="r") as f:
    vocab = json.load(f)

idx2word = {idx: w for w, idx in vocab.items()}
pad_idx = vocab[PAD]


# TENSORFLOW

L = keras.layers
K = keras.backend

# remember to reset your graph if you want to start building it from scratch!
s = reset_tf_session()
tf.set_random_seed(42)


# MODEL

# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(
        model.inputs, keras.layers.GlobalAveragePooling2D()(model.output)
    )
    return model, preprocess_for_model


class decoder:
    # [batch_size, IMG_EMBED_SIZE] of CNN image features
    img_embeds = tf.placeholder("float32", [None, IMG_EMBED_SIZE])
    # [batch_size, time steps] of word ids
    sentences = tf.placeholder("int32", [None, None])

    # we use bottleneck here to reduce the number of parameters
    # image embedding -> bottleneck
    img_embed_to_bottleneck = L.Dense(
        IMG_EMBED_BOTTLENECK,
        input_shape=(None, IMG_EMBED_SIZE),
        activation="elu",
    )
    # image embedding bottleneck -> lstm initial state
    img_embed_bottleneck_to_h0 = L.Dense(
        LSTM_UNITS,
        input_shape=(None, IMG_EMBED_BOTTLENECK),
        activation="elu",
    )
    # word -> embedding
    word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
    # lstm cell (from tensorflow)
    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)

    # we use bottleneck here to reduce model complexity
    # lstm output -> logits bottleneck
    token_logits_bottleneck = L.Dense(
        LOGIT_BOTTLENECK,
        input_shape=(None, LSTM_UNITS),
        activation="elu",
    )
    # logits bottleneck -> logits for next token prediction
    token_logits = L.Dense(
        len(vocab),
        input_shape=(None, LOGIT_BOTTLENECK),
    )

    # initial lstm cell state of shape (None, LSTM_UNITS),
    # we need to condition it on `img_embeds` placeholder.
    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))

    # embed all tokens but the last for lstm input,
    # remember that L.Embedding is callable,
    # use `sentences` placeholder as input.
    word_embeds = word_embed(sentences[:, :-1])

    # during training we use ground truth tokens `word_embeds` as context for next token prediction.
    # that means that we know all the inputs for our lstm and can get
    # all the hidden states with one tensorflow operation (tf.nn.dynamic_rnn).
    # `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
    hidden_states, _ = tf.nn.dynamic_rnn(
        lstm,
        word_embeds,
        initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0),
    )

    # now we need to calculate token logits for all the hidden states

    # first, we reshape `hidden_states` to [-1, LSTM_UNITS]
    flat_hidden_states = tf.reshape(hidden_states, (-1, LSTM_UNITS))

    # then, we calculate logits for next tokens using `token_logits_bottleneck` and `token_logits` layers
    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))

    # then, we flatten the ground truth token ids.
    # remember, that we predict next tokens for each time step,
    # use `sentences` placeholder.
    flat_ground_truth = tf.reshape(sentences[:, 1:], (-1,))

    # we need to know where we have real tokens (not padding) in `flat_ground_truth`,
    # we don't want to propagate the loss for padded output tokens,
    # fill `flat_loss_mask` with 1.0 for real tokens (not pad_idx) and 0.0 otherwise.
    flat_loss_mask = tf.not_equal(flat_ground_truth, pad_idx)

    # compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by lstm
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth, logits=flat_token_logits
    )

    # compute average `xent` over tokens with nonzero `flat_loss_mask`.
    # we don't want to account misclassification of PAD tokens, because that doesn't make sense,
    # we have PAD tokens for batching purposes only!
    loss = tf.reduce_mean(tf.boolean_mask(xent, flat_loss_mask))


# will be used to save/load network weights.
# you need to reset your default graph and define it in the same way to be able to load the saved weights!
saver = tf.train.Saver()

# intialize all variables
s.run(tf.global_variables_initializer())


class final_model:
    # CNN encoder
    encoder, preprocess_for_model = get_cnn_encoder()

    # keras applications corrupt our graph, so we restore trained weights
    saver.restore(s, os.path.abspath(PATH_TO_MODEL))

    # containers for current lstm state
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")
    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

    # input images
    input_images = tf.placeholder("float32", [1, IMG_SIZE, IMG_SIZE, 3], name="images")

    # get image embeddings
    img_embeds = encoder(input_images)

    # initialize lstm state conditioned on image
    init_c = init_h = decoder.img_embed_bottleneck_to_h0(
        decoder.img_embed_to_bottleneck(img_embeds)
    )
    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)

    # current word index
    current_word = tf.placeholder("int32", [1], name="current_input")

    # embedding for current word
    word_embed = decoder.word_embed(current_word)

    # apply lstm cell, get new lstm states
    new_c, new_h = decoder.lstm(
        word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h)
    )[1]

    # compute logits for next token
    new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))
    # compute probabilities for next token
    new_probs = tf.nn.softmax(new_logits)

    # `one_step` outputs probabilities of next token and updates lstm hidden state
    one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)


# this is an actual prediction loop
def generate_caption(image, t=1, sample=False, max_len=20):
    """
    Generate caption for given image.
    if `sample` is True, we will sample next token from predicted probability distribution.
    `t` is a temperature during that sampling,
        higher `t` causes more uniform-like distribution = more chaos.
    """
    # condition lstm on the image
    s.run(
        final_model.init_lstm,
        {final_model.input_images: [image]},
    )

    # current caption
    # start with only START token
    caption = [vocab[START]]

    for _ in range(max_len):
        next_word_probs = s.run(
            final_model.one_step,
            {final_model.current_word: [caption[-1]]},
        )[0]
        next_word_probs = next_word_probs.ravel()

        # apply temperature
        next_word_probs = next_word_probs ** (1 / t) / np.sum(
            next_word_probs ** (1 / t)
        )

        if sample:
            next_word = np.random.choice(range(len(vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break

    return list(map(idx2word.get, caption))


def apply_model_to_image(img):
    img = utils.crop_and_preprocess(
        img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model
    )
    return " ".join(generate_caption(img)[1:-1])


if __name__ == "__main__":
    translator = Translator()
    bot = TeleBot("TOKEN")

    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.send_message(
            message.chat.id,
            "Hello, I'm an image captioning chat bot :)\nПривет, я чат-бот, который описыват изображения :)",
        )

    @bot.message_handler(content_types=["text"])
    def send_text(message):
        print(message.text, message.chat.id)
        bot.send_message(message.chat.id, "Send me an image\nОтправь мне изображение")

    @bot.message_handler(content_types=["photo"])
    def send_image(message):
        fileID = message.photo[-1].file_id
        file = bot.get_file(fileID)
        print("file.file_path =", file.file_path)
        downloaded_file = bot.download_file(file.file_path)
        arr = Image.open(io.BytesIO(downloaded_file))
        arr = np.array(arr)
        caption = apply_model_to_image(arr)
        translations = translator.translate(caption, dest="ru", src="en")
        ru_caption = translations.text
        bot.send_message(message.chat.id, "{}\n{}".format(caption, ru_caption))

    bot.polling()
