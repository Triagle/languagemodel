import itertools
import math
import os
import pickle
import re
from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tqdm
from nltk.corpus import brown


class BengioModel(keras.Model):
    ''' Model that replicates the architecture of Bengio et al.  '''

    def __init__(self, window_size: int, vocabulary_size: int, embedding_size: int = 60, hidden_units: int = 50, regulariser_l=1e-4, use_linear=True, dropout_rate=0.2):
        ''' Initialise model.

        Args:
        - window_size :: Number of words used for context.
        - vocabulary_size :: Size of the vocabulary in the corpus.
        - embedding_size :: Size of the embedding layer.
        - hidden_units :: Number of hidden units in the hidden layer.
        - regulariser_l :: How strong regularisation is (As l -> inf, regularisation gets arbitrarily strong and smooths parameters).
          NOTE: The default value of 0.1 is *just* a placeholder, as the paper didn't specify strength.

        '''
        super().__init__()
        self.window_size = window_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.use_linear = use_linear
        # Improves generalisation of the model by adding dropout to the input (thus reducing the model's dependence on words)
        self.dropout = layers.Dropout(
            dropout_rate, noise_shape=(None, window_size, 1))
        # Takes the place of tanh(d + Hx)
        # You could easily chuck a few more layers here if you wanted to experiment with depth.
        # Not sure why the original paper uses the tanh function (legacy????). I would recommend substituting this with a relu.
        self.non_linear = layers.Dense(hidden_units, activation=tf.nn.relu)
        self.embedding = layers.Embedding(
            vocabulary_size, embedding_size, embeddings_regularizer=keras.regularizers.l2(l2=regulariser_l))
        self.W = layers.Dense(vocabulary_size, use_bias=False,
                              kernel_regularizer=keras.regularizers.l2(l2=regulariser_l))
        self.U = layers.Dense(vocabulary_size, use_bias=False,
                              kernel_regularizer=keras.regularizers.l2(l2=regulariser_l))
        self.b = tf.Variable(tf.random.uniform(
            (vocabulary_size,), minval=-1, maxval=1))

    def call(self, inputs, apply_dropout=True):
        if apply_dropout:
            drop_inputs = self.dropout(inputs)
            embed = self.embedding(drop_inputs)
        else:
            embed = self.embedding(inputs)
        # The embedding output will be a tensor of shape (batch_size, self.window_size, self.embedding_size), i.e one embedding per word in the window
        # This reshape call concatenates all of the embeddings together.
        embed = tf.reshape(embed, (-1, self.embedding_size * self.window_size))
        act = self.non_linear(embed)
        non_linear = self.U(act)
        if self.use_linear:
            linear = self.W(embed)
            logit = linear + non_linear + self.b
        else:
            logit = non_linear + self.b
        return logit


class SubwordModel(keras.Model):
    def __init__(self, window_size: int, vocabulary_size: int, subword_size: int, embedding_size: int = 60, hidden_units: int = 60, regulariser_l=1e-4, use_linear=True):
        super().__init__()
        self.window_size = window_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.use_linear = use_linear
        self.non_linear = layers.Dense(hidden_units, activation=tf.nn.relu)
        self.embedding = layers.Embedding(
            subword_size, embedding_size, embeddings_regularizer=keras.regularizers.l2(l2=regulariser_l))
        self.W = layers.Dense(vocabulary_size, use_bias=False,
                              kernel_regularizer=keras.regularizers.l2(l2=regulariser_l))
        self.U = layers.Dense(vocabulary_size, use_bias=False,
                              kernel_regularizer=keras.regularizers.l2(l2=regulariser_l))
        self.b = tf.Variable(tf.random.uniform(
            (vocabulary_size,), minval=-1, maxval=1))

    def call(self, inputs):
        embed = self.embedding(inputs)
        # embed is of size (batch_sz, window_size, variable_subword_size, embedding_size)
        embed = tf.reduce_sum(embed, 2).to_tensor()
        # The embedding output will be a tensor of shape (batch_size, self.window_size, self.embedding_size), i.e one embedding per word in the window
        # This reshape call concatenates all of the embeddings together.
        embed = tf.reshape(embed, (-1, self.embedding_size * self.window_size))
        act = self.non_linear(embed)
        non_linear = self.U(act)
        if self.use_linear:
            linear = self.W(embed)
            logit = linear + non_linear + self.b
        else:
            logit = non_linear + self.b
        return logit


def window(list, n: int):
    ''' Produce a rolling window over a list of length n (using pad when if we run out of elements). '''
    for i in range(len(list) - n + 1):
        yield list[i: i + n]


def subword_chunks(word, n_low=3, n_high=6):
    word_sequence = '<' + word + '>'
    subwords = []
    for i in range(n_low, min(len(word_sequence), n_high + 1)):
        subwords.extend(window(word_sequence, i))
    subwords.append(word_sequence)
    return subwords


def load_data(window_size: int, subword_size_min: int = 3, subword_size_max: int = 6):
    ''' This code is almost identical to what Ben had '''
    words = [w.lower() for w in brown.words() if re.match(r"[A-z']+", w)]
    counter = Counter(words)
    counter['<UNK>'] = float('inf')
    print('Stripping low frequency words...')
    words = [w if counter[w] >= 4 else '<UNK>' for w in words]
    print('Extracting subword chunks...')
    subwords = [['<UNK>'] if w == '<UNK>' else subword_chunks(w, subword_size_min, subword_size_max)
                for w in words]
    print('Creating word maps...')
    subword_un = set(itertools.chain(*subwords))
    vocab_map = dict(map(reversed, enumerate(set(words))))
    subword_map = dict(map(reversed, enumerate(subword_un)))
    print('Converting words to vectorised hashes')
    subwords = [[subword_map[s] for s in w] for w in subwords]
    words = [vocab_map[word] for word in words]
    print('Creating training/validation/testing windows')
    windows = list(window(subwords, window_size))
    windows.pop()
    labels = words[window_size:]
    return subword_map, vocab_map, tf.ragged.constant(windows), tf.constant(labels)


def perplexity(y_true, y_pred):
    ''' Compute the perplexity of the model. '''
    ce = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True))
    return tf.exp(ce)


def cross_entropy(y_true, y_pred):
    return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))


WINDOW_SIZE = 6
NUM_EPOCHS = 50
EMBED_DIM = 60
HIDDEN_DIM = 100
BATCH_SIZE = 512
SEED = 31415

if not all(os.path.exists(p) for p in ['windows', 'labels', 'vocab_map', 'subword_map']):
    subword_map, vocab_map, windows, labels = load_data(WINDOW_SIZE)

    with open('windows', 'wb') as f:
        pickle.dump(windows.numpy(), f)
    with open('labels', 'wb') as f:
        pickle.dump(labels.numpy(), f)
    with open('subword_map', 'wb') as f:
        pickle.dump(subword_map, f)
    with open('vocab_map', 'wb') as f:
        pickle.dump(vocab_map, f)
else:
    with open('windows', 'rb') as f:
        windows = tf.ragged.constant(pickle.load(f))
    with open('labels', 'rb') as f:
        labels = tf.constant(pickle.load(f))
    with open('subword_map', 'rb') as f:
        subword_map = pickle.load(f)
    with open('vocab_map', 'rb') as f:
        vocab_map = pickle.load(f)


# Shuffle the window and label tensors.
# The initial shuffling will determine the train/val/test split.
# The variable SEED controls what shuffle is produced.
tf.random.set_seed(SEED)
indices = tf.range(0, windows.shape[0], dtype=tf.int32)
shuffle = tf.random.shuffle(indices)
windows = tf.gather(windows, shuffle)
labels = tf.gather(labels, shuffle)

# This code splits the dataset into train/validation/test.
# The way it's split is as follows:
#            train (64%)       val (16%)   test (20%)
#  <-------------------------><-------><----------->
# [.................................................] (dataset)
#
# Tweak TRAIN_VAL_SPLIT and VAL_SPLIT to change the proportion.
n = windows.shape[0]
split = int(0.8 * n)
val_split = int(0.8 * split)
train_windows = windows[:val_split]
train_labels = labels[:val_split]
val_windows = windows[val_split:split]
val_labels = labels[val_split:split]
test_windows = windows[split:]
test_labels = labels[split:]

vocab_size = len(vocab_map) + 1
subword_size = len(subword_map) + 1

# Because BengioModel subclasses the keras Model class you can do all sorts of interesting things with it.
# Check out https://keras.io/api/models/model/ for a list of supported methods and properties.
# model = BengioModel(WINDOW_SIZE, vocab_size, inv_map,
#                     embedding_size=EMBED_DIM, hidden_units=HIDDEN_DIM, use_linear=False)
model = SubwordModel(WINDOW_SIZE, vocab_size, subword_size,
                     embedding_size=EMBED_DIM, hidden_units=HIDDEN_DIM, use_linear=True)
optimiser = tf.optimizers.Adam(learning_rate=1e-3)
# Checkpointing is super useful for making sure your progress isn't lost over a few hours.
# Basically it'll save your weights to disk and then can load them in case one epoch looks interesting or your computer dies.
checkpoint = tf.train.Checkpoint(optimiser=optimiser, model=model)
for epoch in range(NUM_EPOCHS):

    indices = tf.range(0, train_windows.shape[0], dtype=tf.int32)
    shuffle = tf.random.shuffle(indices)
    train_windows = tf.gather(train_windows, shuffle)
    train_labels = tf.gather(train_labels, shuffle)

    p = tqdm.trange(0, train_windows.shape[0], BATCH_SIZE)
    training_loss = 0
    for i in p:
        batch_windows = train_windows[i:i + BATCH_SIZE]
        batch_labels = train_labels[i:i + BATCH_SIZE]
        with tf.GradientTape() as tape:
            out = model.call(batch_windows)
            loss = cross_entropy(batch_labels, out)
            loss += sum(model.losses)
        v = model.trainable_variables
        gradients = tape.gradient(loss, v)
        optimiser.apply_gradients(zip(gradients, v))
        loss_val = loss.numpy().item()
        training_loss += loss_val
        if (i // BATCH_SIZE) % 50 == 0:
            loss_val = loss.numpy().item()
            p.set_description(
                f'{epoch + 1}/{NUM_EPOCHS} loss: {loss_val:02.3f}, perplexity: {round(np.exp(loss_val)):04}')
    training_loss /= i + 1
    training_perplexity = round(np.exp(training_loss))
    print(f'{epoch + 1}/{NUM_EPOCHS}, training loss: {training_loss:02.3f}, training perplexity: {training_perplexity:04}')
    val_losses = np.zeros(
        (val_windows.shape[0] // BATCH_SIZE + 1,), dtype=float)
    ce = 0
    for i in tqdm.trange(0, val_windows.shape[0], BATCH_SIZE):
        val_batch = val_windows[i: i + BATCH_SIZE]
        val_batch_labels = val_labels[i: i + BATCH_SIZE]
        val_out = model.call(val_batch)
        val_loss = tf.reduce_sum(tf.losses.sparse_categorical_crossentropy(
            val_batch_labels, val_out, from_logits=True))
        ce += val_loss.numpy()
    mean_val_loss = ce / val_windows.shape[0]
    mean_val_perplexity = round(np.exp(mean_val_loss))
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, validation loss: {mean_val_loss:02.3f}, validation perplexity: {mean_val_perplexity:04}')
    checkpoint.save(os.path.join('chkpt', 'chkpt'))
