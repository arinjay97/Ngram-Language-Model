# imports go here
import sys
import random
import math
import numpy as np

from collections import Counter

"""
Arinjay Jain
Don't forget to put your name and a file comment here
"""


class LanguageModel:
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
    Parameters:
      n_gram (int): the n-gram order of the language model to create
      is_laplace_smoothing (bool): whether to use Laplace smoothing
    """
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.token_counter = {}
        self.training_ngrams = []
        self.training_ngrams_count = {}
        self.training_ngrams_probabilities = {}
        self.vocab = set()

    def __remove_tokens_appearing_once(self):
        """
        Removes the tokens from token counter that only appear once, creates an <UNK> token to count those instead.
        :return: None
        """
        tokens_to_remove = []
        unk_count = 0
        for token in self.token_counter:
            if self.token_counter[token] == 1:
                tokens_to_remove.append(token)
                unk_count += 1
        self.token_counter = {token: self.token_counter[token] for token in self.token_counter if
                              token not in tokens_to_remove}
        self.token_counter = {key: value for key, value in self.token_counter.items() if value != 1}
        if unk_count > 0:
            self.token_counter["<UNK>"] = unk_count

    def __find_and_count_ngrams(self, training_file):
        """
        Goes through the training file and created and stores ngram of specified size in a dictionary.
        :param training_file: The training file object
        :return: None
        """
        training_file.seek(0)
        for line in training_file:
            tokens = line.split()
            tokens_with_unk = [token if token in self.token_counter and self.token_counter[token] > 1 else self.UNK for
                               token in tokens]
            self.vocab.update(tokens_with_unk)
            for i in range(len(tokens) - self.n_gram + 1):
                self.training_ngrams.append(tuple(tokens_with_unk[i: i + self.n_gram]))
        training_file.close()

    def __count_ngrams(self):
        """
        Counts the number of occurrences of each ngram and stores it in a dictionary.
        :return: None
        """
        for ngram in self.training_ngrams:
            if ngram in self.training_ngrams_count:
                self.training_ngrams_count[ngram] += 1
            else:
                self.training_ngrams_count[ngram] = 1

    def __find_probabilities_without_smoothing(self):
        for ngram in self.training_ngrams_count:
            sequence_words = ngram[:-1]
            sequence_count = 0
            for sequence in self.training_ngrams_count:
                if sequence[:-1] == sequence_words:
                    sequence_count += self.training_ngrams_count[sequence]
            self.training_ngrams_probabilities[ngram] = self.training_ngrams_count[ngram] / sequence_count

    def __find_probabilities_with_smoothing(self):
        for ngram in self.training_ngrams_count:
            sequence_words = ngram[:-1]
            sequence_count = 0
            for sequence in self.training_ngrams_count:
                if sequence[:-1] == sequence_words:
                    sequence_count += self.training_ngrams_count[sequence]
            self.training_ngrams_probabilities[ngram] = (self.training_ngrams_count[ngram] + 1) / (
                    sequence_count + len(self.vocab))

    def __train_ngram_dict(self, training_file_path):
        with open(training_file_path, 'r') as f:
            tokens = f.read().split()

            # Replace tokens that appear only once with UNK
            token_counts = {}
            for token in tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
            for i in range(len(tokens)):
                if token_counts[tokens[i]] == 1:
                    tokens[i] = '<UNK>'

            # Create n-grams and store in dictionary
            ngram_dict = {}
            for i in range(len(tokens) - self.n_gram + 1):
                ngram = tokens[i:i + self.n_gram]
                first_ngram_elements = ngram[:-1]
                last_ngram_element = ngram[-1]

                # Check if first n-1 elements of ngram are in dictionary, if not add them
                if tuple(first_ngram_elements) not in ngram_dict:
                    ngram_dict[tuple(first_ngram_elements)] = {}

                # Check if last element of ngram is in inner dictionary, if not add it
                if last_ngram_element not in ngram_dict[tuple(first_ngram_elements)]:
                    ngram_dict[tuple(first_ngram_elements)][last_ngram_element] = 0

                # Increment count of ngram in inner dictionary
                ngram_dict[tuple(first_ngram_elements)][last_ngram_element] += 1
        print(ngram_dict)

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Parameters:
      training_file_path (str): the location of the training data to read

    Returns:
    None
    """
        training_file = open(training_file_path, "r")
        self.token_counter = Counter(training_file.read().split())
        self.__remove_tokens_appearing_once()
        self.__find_and_count_ngrams(training_file)
        self.__count_ngrams()
        if self.is_laplace_smoothing:
            self.__find_probabilities_with_smoothing()
        else:
            self.__find_probabilities_without_smoothing()

    def __score_with_smoothing(self, sentence):
        """Helper method that calculates the probability score for a given string representing a single sentence.
        For unknown n-grams the probability is the inverse of the size of vocabulary.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        sentence_tokens = [word if word in self.vocab else "<UNK>" for word in sentence.split()]
        ngrams = []
        for i in range(len(sentence_tokens) - self.n_gram + 1):
            ngrams.append(tuple(sentence_tokens[i: i + self.n_gram]))
        prob = 1
        for ngram in ngrams:
            if ngram in self.training_ngrams_count:
                if self.n_gram == 1:
                    prob *= (1 + self.training_ngrams_count[ngram]) / (
                            len(self.vocab) + sum(self.token_counter.values()))
                else:
                    prob *= (1 + self.training_ngrams_count[ngram]) / (len(self.vocab) + self.token_counter[ngram[0]])
            else:
                prob *= 1 / (len(self.vocab) + self.token_counter[ngram[0]])
        return prob

    def __score_without_smoothing(self, sentence):
        """Helper method that calculates the probability score for a given string representing a single sentence.
        Assigns zero probability to new n-grams.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        sentence_tokens = [word if word in self.vocab else "<UNK>" for word in sentence.split()]
        ngrams = []
        for i in range(len(sentence_tokens) - self.n_gram + 1):
            ngrams.append(tuple(sentence_tokens[i: i + self.n_gram]))
        prob = 1
        for ngram in ngrams:
            if ngram in self.training_ngrams_probabilities:
                prob *= self.training_ngrams_probabilities[ngram]
            else:
                return 0
        return prob

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of
      
    Returns:
      float: the probability value of the given string for this model
    """

        return self.__score_with_smoothing(sentence) if self.is_laplace_smoothing else self.__score_without_smoothing(
            sentence)

    def sample_next_word(self, prev_ngram):
        """
        Sample the next word in the sequence by sampling from the n-grams beginning with the previous word

        :param prev_ngram: (str) previous ngram in the sequence
        :return: (str) the next word of the sequence sampled
        """
        if self.n_gram > 1:
            possible_words = [ngram[-1] for ngram in self.training_ngrams if ngram[0] == prev_ngram]
        else:
            possible_words = [ngram for ngram in self.training_ngrams]
        chosen = random.choice(possible_words)
        return chosen

    def __generate_sentence_unigram(self):
        """
        Generates a single sentence from a trained unigram language model using the Shannon technique.

        Returns:
            str: the generated sentence
        """
        sentence = ["<s>"]
        while sentence[-1] != "</s>":
            next_word = self.sample_next_word(sentence[-1])
            if next_word[0] == "</s>":
                sentence.append(next_word[0])
                break
            if next_word[0] == "<s>":
                continue
            sentence.append(next_word[0])
        return " ".join(sentence)

    def generate_sentence(self):
        """Generates a single sentence from a trained ngram language model using the Shannon technique.
      
    Returns:
      str: the generated sentence
    """
        if self.n_gram == 1:
            return self.__generate_sentence_unigram()
        sentence = ["<s>"]
        while sentence[-1] != "</s>":
            next_word = self.sample_next_word(sentence[-1])
            sentence.append(next_word)
        tokens = []
        for token in sentence:
            if token == "<s>" or token == "</s>":
                pass
            tokens.append(token)
        return " ".join(tokens)

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing strings, one per generated sentence
    """
        generated_sentences = []
        while n > 0:
            sentence = self.generate_sentence()
            generated_sentences.append(sentence)
            n -= 1
        return generated_sentences

    def perplexity(self, test_sequence):
        """
        Measures the perplexity for the given test sequence with this trained model.
        As described in the text, you may assume that this sequence may consist of many sentences "glued together".

        Parameters:
          test_sequence (string): a sequence of space-separated tokens to measure the perplexity of
        Returns:
          float: the perplexity of the given sequence
        """
        tokens = test_sequence.split()
        if self.n_gram == 1:
            test_sentence = ["<s>"] + tokens + ["</s>"]
        else:
            test_sentence = ["<s>"] * (self.n_gram - 1) + tokens + ["</s>"] * (self.n_gram - 1)
        test_sentence = [token if token in self.vocab else "<UNK>" for token in test_sentence]
        ngrams = [tuple(test_sentence[i: i + self.n_gram]) for i in range(len(test_sentence) - self.n_gram + 1)]
        perplexity = 1
        for ngram in ngrams:
            if ngram in self.training_ngrams:
                perplexity *= self.training_ngrams_probabilities[ngram]
            else:
                perplexity *= 1 / (len(self.vocab) + sum(self.token_counter.values()))
        perplexity = math.pow((1 / perplexity), 1 / sum(self.token_counter.values()))
        return perplexity


def main():
    # TODO: implement
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]

    unigram_lm = LanguageModel(1, True)
    bigram_lm = LanguageModel(2, True)
    unigram_lm.train(training_path)
    bigram_lm.train(training_path)

    unigram_generated_sentences = unigram_lm.generate(50)
    bigram_generated_sentences = bigram_lm.generate(50)
    unigram_scores = {"File 1": [], "File 2": []}
    bigram_scores = {"File 1": [], "File 2": []}

    print("Unigram generated sentences")
    for sentence in unigram_generated_sentences:
        print(sentence)

    print("Bigram generated sentences")
    for sentence in bigram_generated_sentences:
        print(sentence)

    test_file1 = open(testing_path1)
    for sentence in test_file1:
        unigram_scores["File 1"].append(unigram_lm.score(sentence))
        bigram_scores["File 1"].append(bigram_lm.score(sentence))
    test_file1.close()

    test_file2 = open(testing_path2)
    for sentence in test_file2:
        unigram_scores["File 2"].append(unigram_lm.score(sentence))
        bigram_scores["File 2"].append(bigram_lm.score(sentence))
    test_file1.close()

    print("Scores for 1-grams model:")
    print("For file hw2-test.txt:", "Average score of set =", np.average(unigram_scores["File 1"]), "Standard deviation of set =", np.std(unigram_scores["File 1"]))
    print("For file hw2-my-test.txt:", "Average score of set =", np.average(unigram_scores["File 2"]),
          "Standard deviation of set =", np.std(unigram_scores["File 2"]))

    print("Scores for 2-grams model:")
    print("For file hw2-test.txt:", "Average score of set =", np.average(bigram_scores["File 1"]),
          "Standard deviation of set =", np.std(bigram_scores["File 1"]))
    print("For file hw2-my-test.txt:", "Average score of set =", np.average(bigram_scores["File 2"]),
          "Standard deviation of set =", np.std(bigram_scores["File 2"]))


if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python hw2_lm.py training_file.txt testingfile1.txt testingfile2.txt")
        sys.exit(1)

    main()
