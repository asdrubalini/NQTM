import tensorflow as tf
import json
import inspect
import sys

def tf_debug(tensor: tf.Tensor, name: str) -> tf.Tensor:
    function = inspect.stack()[1].function

    j = {"caller": function, "name": name}
    print_op = tf.compat.v1.print(json.dumps(j), tensor, output_stream=sys.stdout, summarize=-1, sep="\n")

    # Ensure that print_op is executed
    with tf.control_dependencies([print_op]):
        # The operation to ensure print_op executes
        tensor = tf.identity(tensor)

    return tensor

def main():
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_resource_variables()
    tf.compat.v1.random.set_random_seed(42)
    tf.compat.v1.set_random_seed(42)

    embedding_dim = 10
    num_embeddings = 20

    initializer = tf.compat.v1.initializers.variance_scaling(distribution='uniform', seed=42)
    e1: tf.Variable = tf.compat.v1.Variable(
        tf.compat.v1.eye(embedding_dim, name="embedding"), trainable=True
    ) # Basic identity matrix for embeddings

    e1 = tf_debug(e1, "e1")

    # If the number of embeddings is greater than the embedding dimension, extend the embedding matrix
    e2: tf.Variable = tf.compat.v1.get_variable(
        "embedding",
        [embedding_dim, num_embeddings - embedding_dim],
        initializer=initializer,
        trainable=True,
    )

    e2 = tf_debug(e2, "e2")

    e2: tf.Tensor = tf.compat.v1.transpose(e2)  # Transpose the extended part of the embedding matrix
    maybe_embedding_matrix: tf.Variable = tf.compat.v1.Variable(
        tf.compat.v1.concat([e1, e2], axis=0)
    )  # Combine the basic and extended parts of the embeddings

    maybe_embedding_matrix = tf_debug(maybe_embedding_matrix, name="maybe_embedding_matrix")

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(maybe_embedding_matrix)

if __name__ == '__main__':
    main()
