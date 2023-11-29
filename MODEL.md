# Model description

### 1. **Initialization**
- The `NQTM` class is initialized with a configuration dictionary that specifies various parameters of the model, such as the number of topics, activation function, dropout rate, and others.
- It creates an instance of `TopicDisQuant`, which is responsible for the quantization of embeddings.
- Network weights are initialized using Xavier initialization, which helps in keeping the activations at each layer of the network from becoming too small or too large.

### 2. **Encoding**
- The model first encodes the input data through its `encoder` method.
- Input data (`x`) is passed through fully connected layers, where each layer involves a matrix multiplication followed by an addition of a bias term and an activation function (`self.active_fct`).
- After the last fully connected layer, batch normalization is applied, and the output is transformed using a softmax function to generate a topic distribution vector (`theta`). This vector represents the probability distribution over a predefined number of topics.

### 3. **Quantization**
- The topic distribution vector is then passed to the `TopicDisQuant` object.
- `TopicDisQuant` quantizes these embeddings. It calculates distances between the input embeddings (`theta`) and a set of predefined discrete embeddings. The closest embeddings are selected, and a form of quantization loss is calculated. This step forces the model to represent the input data using a limited set of discrete topic vectors.
- The quantization process involves a trade-off between accurately representing the input data and conforming to the discrete embedding structure, which is controlled by the `commitment_cost` parameter.

### 4. **Decoding**
- The quantized embeddings are then fed into the `decoder` method.
- The decoder attempts to reconstruct the original input from the quantized topic distribution. It does this by again using a fully connected layer with batch normalization.
- The output of the decoder is a reconstructed version of the input data, ideally retaining the essential features of the original input.

### 5. **Negative Sampling (Optional)**
- If `word_sample_size` is greater than 0, the model performs negative sampling. This is a technique commonly used in training neural network models on large datasets. It involves sampling negative examples (i.e., examples not in the training set) to improve the model's discrimination ability.
- The negative sampling method in this model seems to be selecting topics and words based on the topic distribution and then applying this information to calculate a part of the loss function.

### 6. **Loss Calculation and Backpropagation**
- The model computes a loss function that includes both the auto-encoding error (the difference between the input and its reconstruction) and the quantization loss from the `TopicDisQuant` object.
- The model is trained by minimizing this loss using the Adam optimizer, which adjusts the network weights in a way that reduces the loss.

### Summary
The model is essentially learning to represent input data in a compact, discrete topic space. By doing so, it can discover the underlying thematic structure of the data. The quantization step adds an interesting twist by enforcing a discrete representation, which could lead to more interpretable topic distributions. This model could be particularly useful in unsupervised learning scenarios where the goal is to uncover latent topics in a dataset, such as a collection of documents.


# Beta and Theta

Certainly! In the context of topic modeling, especially models like Latent Dirichlet Allocation (LDA) and its neural network extensions (like the Neural Quantized Topic Model, NQTM, mentioned in your script), `theta` and `beta` are key components that represent the distributions of topics over documents and words over topics, respectively.

1. **Theta (θ) - Document-Topic Distribution:**
   - **What It Represents:** Theta is a matrix representing the probability distribution of topics in each document. In other words, it tells you how much of each topic is present in a given document.
   - **Dimensions:** If there are `D` documents in your dataset and `K` topics, theta will be a `D x K` matrix.
   - **Usage:** This is used to understand the topic composition of each document. For instance, in a document about "Technology and Environment," theta would show high probabilities for both the "Technology" and "Environment" topics.

2. **Beta (β) - Topic-Word Distribution:**
   - **What It Represents:** Beta is a matrix that represents the probability distribution of words for each topic. It indicates which words are important for which topics.
   - **Dimensions:** If there are `K` topics and `V` words in your vocabulary, beta will be a `K x V` matrix.
   - **Usage:** This is used to understand what each topic is about. For example, a "Technology" topic might have high probabilities for words like "computer," "software," and "internet."

### Application in Topic Modeling
- In topic modeling algorithms, these matrices are not predefined but are learned from the data during the training process.
- By examining theta, you can understand the thematic structure of your documents.
- Beta helps in interpreting the topics themselves by showing which words are most representative of each topic.

### In Your Script
- The script you provided includes functions for both training the model to learn these distributions (`train`) and for extracting and saving them (`print_top_words` for beta, `get_theta` for theta).
- The model learns to associate words with topics and topics with documents based on the patterns in the training data, which is typically a collection of documents represented in a bag-of-words format.

Understanding theta and beta is crucial for interpreting the results of topic modeling, as they provide insights into the latent thematic structure in your collection of documents.
