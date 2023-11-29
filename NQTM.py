import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization
import sys


def xavier_init(fan_in: int, fan_out: int, constant: float = 1.0):
    print(f'xavier_init: {fan_in}, {fan_out}, {constant}')

    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32, seed=42)


class TopicDisQuant(object):
    def __init__(self, embedding_dim: int, num_embeddings: int, commitment_cost: float):
        print(f'TopicDisQuant.__init__: {embedding_dim}, {num_embeddings}, {commitment_cost}')

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        initializer = tf.compat.v1.uniform_unit_scaling_initializer()
        e1 = tf.compat.v1.Variable(tf.compat.v1.eye(embedding_dim, name='embedding'), trainable=True)
        if num_embeddings > embedding_dim:
            e2 = tf.compat.v1.get_variable('embedding', [embedding_dim, num_embeddings - embedding_dim], initializer=initializer, trainable=True)
            e2 = tf.compat.v1.transpose(e2)
            self._E = tf.compat.v1.Variable(tf.compat.v1.concat([e1, e2], axis=0))
        else:
            self._E = e1

    # This method computes the forward pass. It calculates distances between input
    # embeddings and a set of discrete embeddings, selects the closest embeddings,
    # and computes a quantization loss.
    def forward(self, inputs: tf.Tensor):
        print(f'TopicDisQuant.forward: {inputs}')

        input_shape = tf.compat.v1.shape(inputs)
        with tf.compat.v1.control_dependencies([tf.compat.v1.Assert(tf.compat.v1.equal(input_shape[-1], self.embedding_dim), [input_shape])]):
            flat_inputs = tf.compat.v1.reshape(inputs, [-1, self.embedding_dim])

        distances = (tf.compat.v1.reduce_sum(flat_inputs**2, 1, keepdims=True)
                    - 2 * tf.compat.v1.matmul(flat_inputs, tf.compat.v1.transpose(self._E))
                    + tf.compat.v1.transpose(tf.compat.v1.reduce_sum(self._E ** 2, 1, keepdims=True)))

        encoding_indices = tf.compat.v1.argmax(-distances, 1)
        encodings = tf.compat.v1.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.compat.v1.reshape(tensor=encoding_indices, shape=tf.shape(inputs)[:-1])

        quantized = self.quantize(encoding_indices)

        e_latent_loss = tf.compat.v1.reduce_mean((tf.compat.v1.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = tf.compat.v1.reduce_mean((quantized - tf.compat.v1.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + tf.compat.v1.stop_gradient(quantized - inputs)

        return {
                'quantize': quantized,
                'loss': loss,
                'encodings': encodings,
                'e_latent_loss': e_latent_loss,
                'q_latent_loss': q_latent_loss
            }

    def quantize(self, encoding_indices: tf.Tensor):
        print(f'TopicDisQuant.quantize: {encoding_indices}')

        return tf.compat.v1.nn.embedding_lookup(self._E, encoding_indices, validate_indices=False)


class NQTM(object):

    def __init__(self, config):
        self.config = config
        self.active_fct = config['active_fct']
        self.keep_prob = config['keep_prob']
        self.word_sample_size = config['word_sample_size']
        self.topic_num = config['topic_num']
        self.exclude_topt = 1
        self.select_topic_num = int(self.topic_num - 2)

        self.topic_dis_quant = TopicDisQuant(self.topic_num, self.topic_num, commitment_cost=config['commitment_cost'])

        self.init()

        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=sess_config)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def init(self):
        tf.compat.v1.disable_eager_execution()

        self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, self.config['vocab_size']))
        self.w_omega = tf.compat.v1.placeholder(dtype=tf.float32, name='w_omega')

        self.network_weights = self._initialize_weights()
        self.beta = self.network_weights['weights_gener']['h2']

        self.forward(self.x)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.compat.v1.get_variable('h1', [self.config['vocab_size'], self.config['layer1']]),
            'h2': tf.compat.v1.get_variable('h2', [self.config['layer1'], self.config['layer2']]),
            'out': tf.compat.v1.get_variable('out', [self.config['layer2'], self.topic_num]),
        }
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([self.config['layer1']], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([self.config['layer2']], dtype=tf.float32)),
            'out': tf.Variable(tf.zeros([self.topic_num], dtype=tf.float32)),
        }
        all_weights['weights_gener'] = {
            'h2': tf.compat.v1.Variable(xavier_init(self.topic_num, self.config['vocab_size']))
        }
        all_weights['biases_gener'] = {
            'b2': tf.compat.v1.Variable(tf.zeros([self.config['vocab_size']], dtype=tf.float32))
        }
        return all_weights

    # - The model first encodes the input data through its encoder method.
    # - Input data (x) is passed through fully connected layers, where each layer 
    # involves a matrix multiplication followed by an addition of a bias term and 
    # an activation function (self.active_fct).
    # - After the last fully connected layer, batch normalization is applied, and 
    # the output is transformed using a softmax function to generate a topic 
    # distribution vector (theta). This vector represents the probability 
    # distribution over a predefined number of topics.
    def encoder(self, x):
        weights = self.network_weights["weights_recog"]
        biases = self.network_weights['biases_recog']
        layer_1 = self.active_fct(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = self.active_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_do = tf.nn.dropout(layer_2, rate=(1 - self.keep_prob))
        z_mean = BatchNormalization()(tf.add(tf.matmul(layer_do, weights['out']), biases['out']))

        theta = tf.compat.v1.nn.softmax(z_mean)
        return theta

    # The decoder part of the network reconstructs the input from the topic
    # distribution (theta).
    def decoder(self, theta):
        x_recon = BatchNormalization()(tf.add(tf.matmul(theta, self.network_weights["weights_gener"]['h2']), 0.0))
        x_recon = tf.compat.v1.nn.softmax(x_recon)
        return x_recon

    def negative_sampling(self, theta):
        logits = tf.compat.v1.cast(tf.compat.v1.less(theta, tf.compat.v1.reduce_min(tf.compat.v1.nn.top_k(theta, k=self.exclude_topt).values, axis=1, keepdims=True)), tf.float32)
        topic_indices = tf.compat.v1.one_hot(tf.compat.v1.random.categorical(logits, self.select_topic_num, seed=42), depth=theta.shape[1])  # N*1*K
        indices = tf.compat.v1.nn.top_k(tf.compat.v1.tensordot(topic_indices, self.beta, axes=1), self.word_sample_size).indices
        indices = tf.compat.v1.reshape(indices, shape=(-1, self.select_topic_num * self.word_sample_size))

        _m = tf.compat.v1.one_hot(indices, depth=self.beta.shape[1])
        _m = tf.compat.v1.reduce_sum(_m, axis=1)
        return _m

    # Computes the forward pass of the model. It encodes the input, applies
    # quantization, decodes the quantized representation, and calculates the loss,
    # which includes an auto-encoding error and a quantization loss.
    def forward(self, x):
        self.theta_e = self.encoder(x)
        quantization_output = self.topic_dis_quant.forward(self.theta_e)
        self.theta_q = quantization_output['quantize']
        self.x_recon = self.decoder(self.theta_q)

        if self.word_sample_size > 0:
            print('==> word_sample_size > 0')
            _n_samples = self.negative_sampling(self.theta_q)
            negative_error = -self.w_omega * _n_samples * tf.compat.v1.math.log(1 - self.x_recon)
            self.auto_encoding_error = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(-x * tf.compat.v1.math.log(self.x_recon) + negative_error, axis=1))
            self.loss = self.auto_encoding_error + quantization_output['loss']
        else:
            self.auto_encoding_error = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(-x * tf.compat.v1.math.log(self.x_recon), axis=1))
            self.loss = self.auto_encoding_error + quantization_output['loss']

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        self.train_op = optimizer.minimize(self.loss)
