import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theta_path')
    parser.add_argument('--beta_path')
    parser.add_argument('--vocab_path')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    return args

def load_vocab(vocab_path: str):
    vocab = list()

    with open(vocab_path) as file:
        for line in file:
            vocab.append(line.strip())

    return vocab

def print_topic_words(beta, theta, feature_names, output_dir, n_topic_selection=2, n_top_words=2):
    top_words = list()
    doc_topic_word = list()

    for i in range(len(beta)):
        top_words.append(' '.join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]]))

    for i in range(len(theta)):
        doc_topic_word.append('//'.join([top_words[j] for j in theta[i].argsort()[:-n_topic_selection - 1:-1]]))

    with open(os.path.join(output_dir, 'topic_words_new_T{}_K{}_{}th'.format(n_top_words, 50, 1)), 'w') as file:
        for line in doc_topic_word:
            file.write(line + '\n')

if __name__ == '__main__':
    args = parse_args()

    vocab = load_vocab(args.vocab_path)

    theta = np.load(args.theta_path)
    beta = np.load(args.beta_path)

    print_topic_words(beta, theta, vocab, args.output_dir)

