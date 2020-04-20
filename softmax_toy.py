import tensorflow as tf
import numpy as np

np.random.seed(5)
num_label = 50000
dim = 128
batch_size = 100
epoch = 1000


class mylayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embed_ = tf.keras.layers.Embedding(num_label, dim)
        self.dense_ = tf.keras.layers.Dense(num_label)
        self.softmax_ = tf.keras.layers.Softmax()

    def call(self, input):
        embed = self.embed_(input)
        dense_ = self.dense_(embed)
        softmax = self.softmax_(dense_)
        return softmax


model = mylayer()
opti = tf.keras.optimizers.Adam(learning_rate=1e-3)
while (epoch):
    num = np.random.randint(0, num_label, batch_size)
    label = [[i] for i in num]
    with tf.GradientTape() as tape:
        prob = model(num)
        loss = -tf.math.log(tf.compat.v1.batch_gather(prob, label))
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.variables)
    opti.apply_gradients(grads_and_vars=zip(grads, model.variables))
    epoch-=1
    print(loss)
