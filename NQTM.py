import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization
from dataclasses import dataclass
import dataclasses
import logging
import functools
import json
from json import JSONEncoder
import types

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('trace.log', mode='w')
logger.addHandler(file_handler)

class DefaultEncoder(JSONEncoder):
    def default(self, obj):
        # return type(obj)
        # if tf.is_tensor(obj) or tf.is_symbolic_tensor(obj):
            # return 'tf.Tensor'

        if isinstance(obj, NQTM) or isinstance(obj, TopicDisQuant):
            return dict(obj.__dict__.items())

        elif isinstance(obj, types.FunctionType):
            return 'function'

        elif tf.is_tensor(obj):
            return 'tf.Tensor'

            session = tf.compat.v1.get_default_session()
            return "session: " + str(type(session))
            # return str(type(tf.compat.v1.get_default_session().run(obj)))

            tensor_shape = obj.shape.as_list()

            if hasattr(obj, 'numpy'):
                # EagerTensor: Serialize the values
                tensor_value = obj.numpy().tolist()
                return {'value': tensor_value, 'shape': tensor_shape}
            else:
                # SymbolicTensor: Serialize only the shape
                return {'shape': tensor_shape}

        elif isinstance(obj, tf.Variable):
            return obj.value() # Returns a tf.Tensor
            return 'tf.Variable'

        elif dataclasses.is_dataclass(obj):
            return 'DataClass'
            return dict(dataclasses.asdict(obj))
        
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

        logger.debug(json.dumps({
            'caller': caller,
            'arguments': arguments,
        }, cls=DefaultEncoder))

        ret = func(*args, **kwargs)

        logger.debug(json.dumps({
            'caller': caller,
            'return': ret,
        }, cls=DefaultEncoder))

        return ret

    return wrapper


# def log_arguments(func):
    # @functools.wraps(func)
    # def wrapper(*args, **kwargs):
        # arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        # arguments = dict(zip(arg_names, args))
        # arguments.update(kwargs)

        # if args and hasattr(args[0], '__class__'):
            # class_name = args[0].__class__.__name__
            # pre = f'$ {class_name}::{func.__name__}():'
        # else:
            # pre = f'$ {func.__name__}():'

        # arguments_str = '\n'.join([f'{name}: {value}' for name, value in arguments.items()])
        # logger.debug(f"{pre} arguments: \n{arguments_str}\n")

        # ret = func(*args, **kwargs)

        # ret_str = str(ret)  # Convert the return value to string
        # logger.debug(f"{pre} ret - {ret_str}\n")

        # return ret

    # return wrapper

@dataclass
class TopicDisQuantForwardResult:
    quantize: tf.Tensor
    loss: tf.Tensor
    encodings: tf.Tensor
    e_latent_loss: tf.Tensor
    q_latent_loss: tf.Tensor

@log_arguments
def xavier_init(fan_in: int, fan_out: int, constant: float = 1.0) -> tf.Tensor:
    # logger.debug(f'fan_in = {fan_in}, fan_out = {fan_out}, constant = {constant}')

    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    ret = tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    # logger.debug('ret =', ret)

    return ret

class TopicDisQuant(object):
    @log_arguments
    def __init__(self, embedding_dim: int, num_embeddings: int, commitment_cost: float):
        # Initialization method for the TopicDisQuant class. It sets up the basic parameters
        # and initializes the embeddings.
        # logger.debug(f'embedding_dim = {embedding_dim}, num_embeddings = {num_embeddings}, commitment_cost = {commitment_cost}')

        self.embedding_dim = embedding_dim  # Dimension of each embedding vector
        self.num_embeddings = num_embeddings  # Total number of discrete embeddings
        self.commitment_cost = commitment_cost  # Cost for embedding commitment used in loss calculation

        initializer = tf.compat.v1.uniform_unit_scaling_initializer()  # Initializer for the embeddings
        e1: tf.Variable = tf.compat.v1.Variable(tf.compat.v1.eye(embedding_dim, name='embedding'), trainable=True)  # Basic identity matrix for embeddings

        if num_embeddings > embedding_dim:
            # If the number of embeddings is greater than the embedding dimension, extend the embedding matrix
            e2: tf.Variable = tf.compat.v1.get_variable('embedding', [embedding_dim, num_embeddings - embedding_dim], initializer=initializer, trainable=True)
            e2: tf.Tensor = tf.compat.v1.transpose(e2)  # Transpose the extended part of the embedding matrix
            self.maybe_embedding_matrix: tf.Variable = tf.compat.v1.Variable(tf.compat.v1.concat([e1, e2], axis=0))  # Combine the basic and extended parts of the embeddings
        else:
            # Use the basic identity matrix as the embedding matrix instead
            self.maybe_embedding_matrix = e1

    @log_arguments
    def forward(self, inputs: tf.Tensor) -> TopicDisQuantForwardResult:
        # The forward method computes the forward pass of the model. It calculates
        # the distances between input embeddings and a set of discrete embeddings,
        # selects the closest embeddings, and computes a quantization loss.
        # logger.debug(f'inputs = {inputs}')

        input_shape = tf.compat.v1.shape(inputs)  # Get the shape of the input tensor
        # Ensure the last dimension of the input tensor matches the embedding dimension
        with tf.compat.v1.control_dependencies([tf.compat.v1.Assert(tf.compat.v1.equal(input_shape[-1], self.embedding_dim), [input_shape])]):
            flat_inputs = tf.compat.v1.reshape(inputs, [-1, self.embedding_dim])  # Flatten the input tensor

        # Calculate distances between flattened inputs and embeddings
        distances: tf.Tensor = (tf.compat.v1.reduce_sum(flat_inputs**2, 1, keepdims=True)
                    - 2 * tf.compat.v1.matmul(flat_inputs, tf.compat.v1.transpose(self.maybe_embedding_matrix))
                    + tf.compat.v1.transpose(tf.compat.v1.reduce_sum(self.maybe_embedding_matrix ** 2, 1, keepdims=True)))

        encoding_indices = tf.compat.v1.argmax(-distances, 1)  # Find the indices of closest embeddings
        encodings = tf.compat.v1.one_hot(encoding_indices, self.num_embeddings)  # Create one-hot encodings
        encoding_indices = tf.compat.v1.reshape(tensor=encoding_indices, shape=tf.shape(inputs)[:-1])  # Reshape encoding indices to match the input shape

        quantized = self.quantize(encoding_indices)  # Quantize the encoding indices

        # Calculate the latent losses for the quantized embeddings
        e_latent_loss: tf.Tensor = tf.compat.v1.reduce_mean((tf.compat.v1.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss: tf.Tensor = tf.compat.v1.reduce_mean((quantized - tf.compat.v1.stop_gradient(inputs)) ** 2)
        loss: tf.Tensor = q_latent_loss + self.commitment_cost * e_latent_loss  # Total loss

        quantized: tf.Tensor = inputs + tf.compat.v1.stop_gradient(quantized - inputs)  # Update quantized embeddings

        # Return a dictionary of the outputs including quantized embeddings and losses
        return TopicDisQuantForwardResult(
            quantize=quantized,
            loss=loss,
            encodings=encodings,
            e_latent_loss=e_latent_loss,
            q_latent_loss=q_latent_loss,
        )

    @log_arguments
    def quantize(self, encoding_indices: tf.Tensor) -> tf.Tensor:
        # This method quantizes the encoding indices by looking up the corresponding embeddings.
        #logger.debug(f'encoding_indices = {encoding_indices}')

        # Perform embedding lookup for the given indices
        return tf.compat.v1.nn.embedding_lookup(self.maybe_embedding_matrix, encoding_indices, validate_indices=False)


class NQTM(object):
    @log_arguments
    def __init__(self, config):
        self.config = config
        self.active_fct = config['active_fct']
        self.keep_prob = config['keep_prob']
        self.word_sample_size = config['word_sample_size']
        self.topic_num = config['topic_num']
        self.exclude_topt = 1
        self.select_topic_num = int(self.topic_num - 2)

        self.sess = None

        self.topic_dis_quant = TopicDisQuant(self.topic_num, self.topic_num, commitment_cost=config['commitment_cost'])

        self.init()

        # self.sess = tf.compat.v1.Session(config=sess_config)
        # self.sess.run(tf.compat.v1.global_variables_initializer())

    def set_session(self, session: tf.compat.v1.Session):
        self.sess = session

    @log_arguments
    def init(self):
        self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, self.config['vocab_size']))
        self.w_omega = tf.compat.v1.placeholder(dtype=tf.float32, name='w_omega')

        self.network_weights = self._initialize_weights()
        self.beta = self.network_weights['weights_gener']['h2']

        self.forward(self.x)

    @log_arguments
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
    @log_arguments
    def encoder(self, x: tf.Tensor) -> tf.Tensor:
        weights = self.network_weights["weights_recog"]
        biases = self.network_weights['biases_recog']
        layer_1 = self.active_fct(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer_2 = self.active_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        layer_do = tf.nn.dropout(layer_2, rate=(1 - self.keep_prob))
        z_mean = BatchNormalization()(tf.add(tf.matmul(layer_do, weights['out']), biases['out']))

        theta = tf.compat.v1.nn.softmax(z_mean)

        logger.debug('Theta:', theta)

        return theta

    # The decoder part of the network reconstructs the input from the topic
    # distribution (theta).
    @log_arguments
    def decoder(self, theta: tf.Tensor) -> tf.Tensor:
        x_recon = BatchNormalization()(tf.add(tf.matmul(theta, self.network_weights["weights_gener"]['h2']), 0.0))
        x_recon = tf.compat.v1.nn.softmax(x_recon)
        return x_recon

    @log_arguments
    def negative_sampling(self, theta: tf.Tensor) -> tf.Tensor:
        logits = tf.compat.v1.cast(tf.compat.v1.less(theta, tf.compat.v1.reduce_min(tf.compat.v1.nn.top_k(theta, k=self.exclude_topt).values, axis=1, keepdims=True)), tf.float32)
        topic_indices = tf.compat.v1.one_hot(tf.compat.v1.random.categorical(logits, self.select_topic_num), depth=theta.shape[1])  # N*1*K
        indices = tf.compat.v1.nn.top_k(tf.compat.v1.tensordot(topic_indices, self.beta, axes=1), self.word_sample_size).indices
        indices = tf.compat.v1.reshape(indices, shape=(-1, self.select_topic_num * self.word_sample_size))

        _m = tf.compat.v1.one_hot(indices, depth=self.beta.shape[1])
        _m = tf.compat.v1.reduce_sum(_m, axis=1)
        return _m

    # Computes the forward pass of the model. It encodes the input, applies
    # quantization, decodes the quantized representation, and calculates the loss,
    # which includes an auto-encoding error and a quantization loss.
    @log_arguments
    def forward(self, x: tf.Tensor):
        self.theta_e = self.encoder(x)
        quantization_output = self.topic_dis_quant.forward(self.theta_e)
        self.theta_q = quantization_output.quantize
        self.x_recon = self.decoder(self.theta_q)

        if self.word_sample_size > 0:
            logging.info('==> word_sample_size > 0')
            _n_samples = self.negative_sampling(self.theta_q)
            negative_error = -self.w_omega * _n_samples * tf.compat.v1.math.log(1 - self.x_recon)
            self.auto_encoding_error = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(-x * tf.compat.v1.math.log(self.x_recon) + negative_error, axis=1))
            self.loss = self.auto_encoding_error + quantization_output.loss
        else:
            self.auto_encoding_error = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(-x * tf.compat.v1.math.log(self.x_recon), axis=1))
            self.loss = self.auto_encoding_error + quantization_output.loss

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        self.train_op = optimizer.minimize(self.loss)
