import math
import os
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
        self.non_linear2 = layers.Dense(hidden_units, activation=tf.nn.relu)
        self.non_linear3 = layers.Dense(hidden_units, activation=tf.nn.relu)
        # NOTE: Paper didn't specify if the embedding is regularised????
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
        act = self.non_linear2(act)
        act = self.non_linear3(act)
        non_linear = self.U(act)
        if self.use_linear:
            linear = self.W(embed)
            logit = linear + non_linear + self.b
        else:
            logit = non_linear + self.b
        return logit


def window(list, n: int):
    ''' Produce a rolling window over a list of length n (using pad when if we run out of elements). '''
    for i in range(len(list) - n - 1):
        yield list[i: i + n]


def load_data(filename: str, window_size: int):
    ''' This code is almost identical to what Ben had '''
    words = [w.lower() for w in brown.words() if re.match(r"[A-z']+", w)]
    counts = Counter(words)
    counts['<UNK>'] = float('inf')
    vocab = filter(lambda w: counts[w] >= 4, counts)
    vocab_map = dict(map(reversed, enumerate(vocab)))
    words = list(map(lambda w: vocab_map.get(w, vocab_map['<UNK>']), words))
    windows = list(window(words, window_size))
    labels = words[window_size:len(words) - 1]
    return vocab_map, tf.constant(windows), tf.constant(labels)


def perplexity(y_true, y_pred):
    ''' Compute the perplexity of the model. '''
    ce = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True))
    return tf.exp(ce)


def cross_entropy(y_true, y_pred):
    return tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))


print(tf.config.list_physical_devices())
WINDOW_SIZE = 6
NUM_EPOCHS = 50
EMBED_DIM = 60
HIDDEN_DIM = 100
BATCH_SIZE = 128
SEED = 31415

vocab_map, windows, labels = load_data('brown.txt', WINDOW_SIZE)

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

# Checkpointing is super useful for making sure your progress isn't lost over a few hours.
# Basically it'll save your weights to disk and then can load them in case one epoch looks interesting or your computer dies.
# Check out https://keras.io/api/callbacks/model_checkpoint/ for details on the flags you can configure for this.

checkpoint = keras.callbacks.ModelCheckpoint('model.{epoch:02d}-{val_loss:.2f}.h5',
                                             # This will only save the model if it beats the best validation accuracy.
                                             # Disable this to save more but use more disk space.
                                             save_best_only=True)

vocab_size = len(vocab_map) + 1

model = BengioModel(WINDOW_SIZE, vocab_size,
                    embedding_size=EMBED_DIM, hidden_units=HIDDEN_DIM, use_linear=False)
# Because BengioModel subclasses the keras Model class you can do all sorts of interesting things with it.
# Check out https://keras.io/api/models/model/ for a list of supported methods and properties.
optimiser = tf.optimizers.Adam(learning_rate=1e-3)
checkpoint = tf.train.Checkpoint(optimiser=optimiser, model=model)
for epoch in range(NUM_EPOCHS):

    indices = tf.range(0, train_windows.shape[0], dtype=tf.int32)
    shuffle = tf.random.shuffle(indices)
    train_windows = tf.gather(train_windows, shuffle)
    train_labels = tf.gather(train_labels, shuffle)

    p = tqdm.trange(0, train_windows.shape[0], BATCH_SIZE)
    for i in p:
        batch_windows = train_windows[i:i + BATCH_SIZE]
        batch_labels = train_labels[i:i + BATCH_SIZE]
        with tf.GradientTape() as tape:
            out = model.call(batch_windows, apply_dropout=False)
            loss = cross_entropy(batch_labels, out)
            loss += sum(model.losses)
        v = model.trainable_variables
        gradients = tape.gradient(loss, v)
        optimiser.apply_gradients(zip(gradients, v))
        if (i // BATCH_SIZE) % 50 == 0:
            loss_val = loss.numpy().item()
            p.set_description(
                f'{epoch + 1}/{NUM_EPOCHS} loss: {loss_val:02.3f}, perplexity: {round(math.exp(loss_val)):04}')

    val_losses = np.zeros(
        (val_windows.shape[0] // BATCH_SIZE + 1,), dtype=float)
    ce = 0
    for i in tqdm.trange(0, val_windows.shape[0], BATCH_SIZE):
        val_batch = val_windows[i: i + BATCH_SIZE]
        val_batch_labels = val_labels[i: i + BATCH_SIZE]
        val_out = model.call(val_batch, apply_dropout=False)
        val_loss = tf.reduce_sum(tf.losses.sparse_categorical_crossentropy(
            val_batch_labels, val_out, from_logits=True))
        ce += val_loss.numpy()
    mean_val_loss = ce / val_windows.shape[0]
    mean_val_perplexity = np.round(np.exp(mean_val_loss))
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, validation loss: {mean_val_loss:02.3f}, validation perplexity: {mean_val_perplexity:04}')
    checkpoint.save(os.path.join('chkpt', 'chkpt'))

# model.compile(optimizer='adam',
#               loss=cross_entropy,
#               metrics=[perplexity])
# model.fit(train_windows, train_labels,
#           # Comment this line to disable checkpointing
#           callbacks=[checkpoint],
#           batch_size=BATCH_SIZE,
#           validation_data=(val_windows, val_labels),
#           epochs=NUM_EPOCHS)
# model.evaluate(test_windows, test_labels)
