import numpy as np
import tensorflow as tf
import time
import sys
import os
from time import time
import pickle
import logging
import json
from json import JSONEncoder
import types
import functools
import inspect
import unittest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.addHandler(logging.StreamHandler(sys.stdout))

class DefaultEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, NQTM) or isinstance(obj, TopicDisQuant):
            return dict(obj.__dict__.items())

        elif isinstance(obj, types.FunctionType):
            return 'function'

        elif tf.is_tensor(obj):
            # Get tensor details
            shape = obj.shape
            data_type = obj.dtype

            return {
                'type': 'tf.Tensor',
                'shape': str(shape),
                'dtype': str(data_type),
            }

        elif isinstance(obj, tf.Variable):
            return obj.value() # Returns a tf.Tensor
            return 'tf.Variable'

        try:
            return json.JSONEncoder.default(self, obj)
        except:
            return 'TypeError'


def log_arguments(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        arguments = dict(zip(arg_names, args))
        arguments.update(kwargs)

        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            caller = f'{class_name}::{func.__name__}'
        else:
            caller = func.__name__

        logger.error(json.dumps({
            'caller': caller,
            'arguments': arguments,
        }, cls=DefaultEncoder))

        ret = func(*args, **kwargs)

        logger.error(json.dumps({
            'caller': caller,
            'return': ret,
        }, cls=DefaultEncoder))

        return ret

    return wrapper

def tf_debug(tensor: tf.Tensor, name: str) -> tf.Tensor:
    # if not os.getenv("DEBUG") and not name.startswith("keep."):
        # return tensor

    function = inspect.stack()[1].function

    j = { 'caller': function, 'name': name }

    print_op = tf.compat.v1.print(json.dumps(j), tensor, output_stream=sys.stdout, summarize=-1, sep='\n')

    # Ensure that print_op is executed
    with tf.control_dependencies([print_op]):
        # The operation to ensure print_op executes
        tensor = tf.identity(tensor)

    return tensor


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    ret = tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32, seed=42)

    return tf_debug(ret, 'xavier_out')


class TopicDisQuant(object):
    @log_arguments
    def __init__(self, embedding_dim: int, num_embeddings: int, commitment_cost: float):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        initializer = tf.compat.v1.initializers.variance_scaling(distribution='uniform', seed=42)
        e1 = tf.compat.v1.Variable(tf.eye(embedding_dim, name='embedding'), trainable=True)
        # e1 = tf_debug(e1, name='keep.e1')

        if num_embeddings > embedding_dim:
            e2 = tf.compat.v1.get_variable('embedding', [embedding_dim, num_embeddings - embedding_dim], initializer=initializer, trainable=True)
            # e2 = tf_debug(e2, name='keep.e2')

            e2 = tf.compat.v1.transpose(e2)
            self._E = tf.compat.v1.Variable(tf.compat.v1.concat([e1, e2], axis=0))
        else:
            self._E = e1

        # self._E = tf_debug(self._E, name='maybe_embedding_matrix')

    @log_arguments
    def forward(self, inputs):
        inputs = tf_debug(inputs, 'inputs')

        input_shape = tf.shape(inputs)
        with tf.control_dependencies([tf.Assert(tf.equal(input_shape[-1], self.embedding_dim), [input_shape])]):
            flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                    - 2 * tf.matmul(flat_inputs, tf.transpose(self._E))
                    + tf.transpose(tf.reduce_sum(self._E ** 2, 1, keepdims=True)))

        encoding_indices = tf.argmax(- distances, 1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])

        quantized = self.quantize(encoding_indices)

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + tf.stop_gradient(quantized - inputs)

        quantized = tf_debug(quantized, 'quantized')
        loss = tf_debug(loss, 'loss')

        return {
                'quantize': quantized,
                'loss': loss,
                'encodings': encodings,
                'e_latent_loss': e_latent_loss,
                'q_latent_loss': q_latent_loss
            }

    @log_arguments
    def quantize(self, encoding_indices):
        encoding_indices = tf_debug(encoding_indices, 'encoding_indices')

        ret = tf.nn.embedding_lookup(self._E, encoding_indices, validate_indices=False)

        return tf_debug(ret, 'quantize_out')

class TestTopicDisQuant(tf.test.TestCase):
    def test_initialization(self):
        embedding_dim = 10
        num_embeddings = 20
        commitment_cost = 0.25

        tdq = TopicDisQuant(embedding_dim, num_embeddings, commitment_cost)

        self.assertEqual(tdq.embedding_dim, embedding_dim)
        self.assertEqual(tdq.num_embeddings, num_embeddings)
        self.assertEqual(tdq.commitment_cost, commitment_cost)
        self.assertTrue(isinstance(tdq._E, tf.Variable))
        self.assertEqual(tdq._E.shape.as_list(), [num_embeddings, embedding_dim])

        # tdq._E = tf_debug(tdq._E, 'keep._E')

        expected_embedding_matrix = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [
                0.09162027,
                0.21204221,
                0.32615584,
                -0.11078015,
                0.24337202,
                0.19897145,
                0.39131552,
                0.39388168,
                0.4274729,
                -0.36743802,
            ],
            [
                -0.22212353,
                -0.36713466,
                -0.52108943,
                0.4401281,
                0.22639674,
                0.3024249,
                0.26822084,
                -0.48262185,
                -0.05199763,
                0.49165583,
            ],
            [
                0.1306575,
                -0.061934143,
                -0.19612336,
                -0.27101946,
                0.12554097,
                0.19340438,
                -0.3150789,
                0.21211624,
                -0.049551994,
                0.026616156,
            ],
            [
                -0.017760396,
                -0.1413633,
                0.16142529,
                -0.045005202,
                -0.23712814,
                -0.5415412,
                0.29591304,
                -0.50508505,
                -0.40171874,
                -0.06551796,
            ],
            [
                0.40718067,
                0.13930118,
                -0.43893322,
                -0.12183684,
                -0.53747606,
                0.44685817,
                0.12958145,
                0.16221988,
                -0.16969511,
                -0.0865702,
            ],
            [
                -0.19360852,
                -0.17063913,
                0.20461833,
                0.128007,
                -0.028867722,
                -0.035159707,
                -0.44927338,
                -0.4175684,
                -0.467711,
                0.46463394,
            ],
            [
                0.26711804,
                -0.3061272,
                -0.31804222,
                0.06871563,
                0.5197791,
                -0.3447575,
                -0.40593618,
                -0.4806441,
                -0.43038955,
                -0.36826122,
            ],
            [
                0.38771236,
                0.045807004,
                -0.528653,
                0.35553014,
                0.33580428,
                -0.43672487,
                0.104386866,
                -0.14273733,
                -0.3958457,
                -0.4888562,
            ],
            [
                0.503811,
                0.1618458,
                -0.1468789,
                0.026929677,
                0.2996443,
                0.25494063,
                0.18318701,
                0.46656263,
                0.51119757,
                0.41626287,
            ],
            [
                -0.47605565,
                0.3618819,
                0.3645941,
                -0.54115653,
                0.07553363,
                0.39952016,
                -0.46746078,
                0.43313396,
                0.13382113,
                -0.47895142,
            ],
        ]
        expected_embedding_matrix = tf.convert_to_tensor(expected_embedding_matrix, dtype=tf.float32)
        self.assertEqual(tdq._E.shape.as_list(), expected_embedding_matrix.shape.as_list())

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            evaluated_expected_matrix = sess.run(expected_embedding_matrix)
            evaluated_tdq_matrix = sess.run(tdq._E)
            self.assertAllClose(evaluated_expected_matrix, evaluated_tdq_matrix)

    # def test_forward(self):
        # tdq = TopicDisQuant(embedding_dim=10, num_embeddings=20, commitment_cost=0.25)
        # inputs = tf.random.uniform([100, 10], dtype=tf.float32, seed=42)

        # result = tdq.forward(inputs)
        # self.assertEqual(result['quantize'].shape, inputs.shape)

    # def test_quantize(self):
        # embedding_dim = 10

        # tdq = TopicDisQuant(embedding_dim=embedding_dim, num_embeddings=20, commitment_cost=0.25)
        # encoding_indices = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)

        # quantized_output = tdq.quantize(encoding_indices)

        # self.assertEqual(quantized_output.shape.as_list(), [encoding_indices.shape[0], embedding_dim]) # this was [5, 20]
        # self.assertTrue(isinstance(quantized_output, tf.Tensor))

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

        # sess_config = tf.ConfigProto()
        # sess_config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=sess_config)
        # self.sess.run(tf.global_variables_initializer())

    def init(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.config['vocab_size']))
        self.w_omega = tf.placeholder(dtype=tf.float32, name='w_omega')
        
        self.network_weights = self._initialize_weights()
        self.beta = self.network_weights['weights_gener']['h2']

        self.forward(self.x)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.get_variable('h1', [self.config['vocab_size'], self.config['layer1']]),
            'h2': tf.get_variable('h2', [self.config['layer1'], self.config['layer2']]),
            'out': tf.get_variable('out', [self.config['layer2'], self.topic_num]),
        }
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([self.config['layer1']], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([self.config['layer2']], dtype=tf.float32)),
            'out': tf.Variable(tf.zeros([self.topic_num], dtype=tf.float32)),
        }
        all_weights['weights_gener'] = {
            'h2': tf.Variable(xavier_init(self.topic_num, self.config['vocab_size']))
        }
        all_weights['biases_gener'] = {
            'b2': tf.Variable(tf.zeros([self.config['vocab_size']], dtype=tf.float32))
        }

        for category, variable in all_weights.items():
            for name, variable in variable.items():
                tf_debug(variable, f"{category}.{name}")

        return all_weights

    def encoder(self, x):
        x = tf_debug(x, 'x')

        weights = self.network_weights["weights_recog"]
        biases = self.network_weights['biases_recog']
        layer_1 = self.active_fct(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = self.active_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_do = tf.nn.dropout(layer_2, self.keep_prob)
        z_mean = tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, weights['out']), biases['out']))

        theta = tf.nn.softmax(z_mean)
        theta = tf_debug(theta, 'theta')

        return theta

    def decoder(self, theta):
        theta = tf_debug(theta, 'theta')

        x_recon = tf.contrib.layers.batch_norm(tf.add(tf.matmul(theta, self.network_weights["weights_gener"]['h2']), 0.0))
        x_recon = tf.nn.softmax(x_recon)
        x_recon = tf_debug(x_recon, 'x_recon')

        return x_recon

    def negative_sampling(self, theta):
        theta = tf_debug(theta, 'theta')

        logits = tf.cast(tf.less(theta, tf.reduce_min(tf.nn.top_k(theta, k=self.exclude_topt).values, axis=1, keepdims=True)), tf.float32)
        topic_indices = tf.one_hot(tf.multinomial(logits, self.select_topic_num), depth=theta.shape[1])  # N*1*K
        indices = tf.nn.top_k(tf.tensordot(topic_indices, self.beta, axes=1), self.word_sample_size).indices
        indices = tf.reshape(indices, shape=(-1, self.select_topic_num * self.word_sample_size))

        _m = tf.one_hot(indices, depth=self.beta.shape[1])
        _m = tf.reduce_sum(_m, axis=1)
        _m = tf_debug(_m, '_m')

        return _m

    def forward(self, x):
        x = tf_debug(x, 'x')

        self.theta_e = self.encoder(x)
        quantization_output = self.topic_dis_quant.forward(self.theta_e)
        self.theta_q = quantization_output['quantize']
        self.x_recon = self.decoder(self.theta_q)

        if self.word_sample_size > 0:
            # print('==>word_sample_size > 0')
            _n_samples = self.negative_sampling(self.theta_q)
            negative_error = -self.w_omega * _n_samples * tf.log(1 - self.x_recon)
            self.auto_encoding_error = tf.reduce_mean(tf.reduce_sum(-x * tf.log(self.x_recon) + negative_error, axis=1))
            self.loss = self.auto_encoding_error + quantization_output["loss"]
        else:
            self.auto_encoding_error = tf.reduce_mean(tf.reduce_sum(-x * tf.log(self.x_recon), axis=1))
            self.loss = self.auto_encoding_error + quantization_output['loss']

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        self.train_op = optimizer.minimize(self.loss)

if __name__ == '__main__':
    unittest.main()
