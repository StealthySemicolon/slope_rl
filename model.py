import tensorflow as tf

"""
NVIDIA model used
Image normalization to avoid saturation and make gradients work better.
Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
Drop out (0.5)
Fully connected: neurons: 100, activation: ELU
Fully connected: neurons: 50, activation: ELU
Fully connected: neurons: 10, activation: ELU
Fully connected: neurons: 1 (output)
# the convolution layers are meant to handle feature engineering
the fully connected layer for predicting the steering angle.
dropout avoids overfitting
ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
"""

class Model:
    def __init__(self, sess, num_categories=1):
        self.sess = sess
        self.input = tf.placeholder(tf.float32, [None, 200, 150, 3], name="input")
        self.is_train = tf.placeholder_with_default(False, shape=None, name="is_train")

        self.model = tf.layers.conv2d(self.input, filters=24, kernel_size=(5, 5), strides=(2, 2), activation=tf.nn.elu)
        self.model = tf.layers.conv2d(self.model, filters=36, kernel_size=(5, 5), strides=(2, 2), activation=tf.nn.elu)
        self.model = tf.layers.conv2d(self.model, filters=48, kernel_size=(5, 5), strides=(2, 2), activation=tf.nn.elu)
        self.model = tf.layers.conv2d(self.model, filters=64, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.elu)
        self.model = tf.layers.conv2d(self.model, filters=64, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.elu)
        
        self.model = tf.layers.flatten(self.model)
        self.model = tf.layers.dropout(self.model, rate=0.5, training=self.is_train)
        self.model = tf.layers.dense(self.model, 100, activation=tf.nn.elu)
        self.model = tf.layers.dense(self.model, 50, activation=tf.nn.elu)
        self.model = tf.layers.dense(self.model, 10, activation=tf.nn.elu)

        self.logits = tf.layers.dense(self.model, num_categories)
        self.logits = tf.multiply(self.logits, 1, name="logits")
        self.softmax = tf.nn.softmax(self.logits)

        self.output = tf.multiply(self.softmax, 1, name="output")
        self.y_true = tf.placeholder(tf.float32, [None, num_categories], name="y_true")

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_true)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=1)

        self.sess.run(tf.global_variables_initializer())
    def fit(self, X, y, epochs=10):
        for _ in range(epochs):
            self.sess.run(self.optimizer, feed_dict={self.input:X, self.y_true:y, self.is_train:True})
    def predict(self, X):
        return self.sess.run(self.output, feed_dict={self.input:X})
    def save(self, folder_name):
        self.saver.save(self.sess, folder_name + "/model")
    def load(self, folder_name):
        self.saver = tf.train.import_meta_graph(folder_name + "/model")
        self.saver.restore(self.sess, tf.train.latest_checkpoint(folder_name))
        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("input:0")
        self.is_train = graph.get_tensor_by_name("is_train:0")
        self.logits = graph.get_tensor_by_name("logits:0")
        self.output = graph.get_tensor_by_name("output:0")
        self.y_true = graph.get_tesnor_by_name("y_true:0")
