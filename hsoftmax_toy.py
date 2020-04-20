import tensorflow as tf
import numpy as np
import huffman_tree

np.random.seed(5)
num_label = 500
dim = 128
epoch = 100
node_count = np.sort(np.random.randint(0, 1000, num_label))[::-1]
tree, paths, codes, paths_length = huffman_tree.build_huffman_tree(node_count)


class hierarchical_Softmax(tf.keras.layers.Layer):
    def __init__(self, label_num):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(label_num, dim)

    def call(self, hidden_, target, target_path, target_path_len, target_code):
        embeds = self.embed(target_path)
        flag = tf.cast(target_code, tf.float32)
        score = tf.squeeze(tf.nn.sigmoid(tf.matmul(embeds, tf.expand_dims(hidden_, -1))))
        score_ = tf.negative(tf.math.log(tf.multiply(flag, score) + tf.multiply(1 - flag, 1 - score)))
        path_score = tf.compat.v1.batch_gather(score_, tf.ragged.range(target_path_len))
        return tf.reduce_mean(path_score)


class mylayer(tf.keras.Model):
    def __init__(self, num_label):
        super().__init__()
        self.embed_ = tf.keras.layers.Embedding(num_label, dim)
        self.dense_ = tf.keras.layers.Dense(num_label)
        self.prob = hierarchical_Softmax(label_num=num_label)

    def call(self, inputs, target, target_path, target_path_len, target_code):
        embed = self.embed_(inputs)
        prob = self.prob(embed, target, target_path, target_path_len, target_code)
        return prob


def get_node_info(label_list):
    target_path = []
    target_path_len = []
    target_code = []
    for i in label_list:
        target_path.append(paths[i])
        target_path_len.append(paths_length[i])
        target_code.append(codes[i])
    return target_path, target_path_len, target_code


model = mylayer(len(paths))
opti = tf.keras.optimizers.Adam(learning_rate=1e-3)
while True:
    num = np.random.randint(0, num_label, 100)
    target_path, target_path_len, target_code = get_node_info(num)
    label = [[i] for i in num]
    with tf.GradientTape() as tape:
        prob = model(num, num, np.array(target_path), np.array(target_path_len), np.array(target_code))
        loss = prob

    grads = tape.gradient(loss, model.variables)
    opti.apply_gradients(grads_and_vars=zip(grads, model.variables))
    epoch-=1
    print(loss)