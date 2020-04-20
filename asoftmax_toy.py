import tensorflow as tf
import numpy as np
import time

np.random.seed(5)
num_label = 50000
head_num = 10000
model_dim = 128
epoch = 1000


class Adaptive_softmax(tf.keras.layers.Layer):
    def __init__(self, model_dim, dim_factor):
        super().__init__()
        self.tail_dim = model_dim / dim_factor
        self.dense_head = tf.keras.layers.Dense(head_num + 1)
        self.dense_tail = tf.keras.layers.Dense(self.tail_dim)
        self.logits_tail = tf.keras.layers.Dense(num_label - head_num)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, embeds):
        head_logits = self.dense_head(embeds)
        tail_logits = self.logits_tail(self.dense_tail(embeds))
        head_probs = self.softmax(head_logits)
        tail_in_head = head_probs[:, head_num]
        tail_probs = self.softmax(tail_logits) * tf.expand_dims(tail_in_head, -1)
        full_probs = tf.concat([head_probs[:, :head_num], tail_probs], axis=-1)
        return full_probs


class mylayer(tf.keras.Model):
    def __init__(self, model_dim, factor_dim):
        super().__init__()
        self.embed_ = tf.keras.layers.Embedding(num_label, 128)
        self.Asoftmax = Adaptive_softmax(model_dim, factor_dim)

    def call(self, input):
        embed = self.embed_(input)
        full_probs = self.Asoftmax(embed)
        return full_probs


model = mylayer(model_dim, 4)
opti = tf.keras.optimizers.Adam(learning_rate=1e-3)
start = time.time()
while (epoch):
    num = np.random.randint(0, num_label, 100)
    label = [[i] for i in num]
    with tf.GradientTape() as tape:
        full_prob = model(num)
        loss = -tf.math.log(tf.compat.v1.batch_gather(full_prob, label))
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.variables)
    opti.apply_gradients(grads_and_vars=zip(grads, model.variables))
    epoch -= 1
    print(loss)
