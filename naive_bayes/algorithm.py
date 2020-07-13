import nltk
from pathlib import Path
import string
import math
import random
from sklearn.model_selection import train_test_split


class naive_bayes_algorithm:

    # k: Number of folds
    # gram_number: 1 is uni-gram, 2 is bi-gram.

    def __init__(self, gram_number):
        self.gram_number = gram_number
        self.pos_path = 'data/POS'
        self.neg_path = 'data/NEG'
        self.pos_examples = [list(), list()]
        self.neg_examples = [list(), list()]
        self.raw_pos_train = list()
        self.pos_train = list()
        self.pos_test = list()
        self.raw_neg_train = list()
        self.neg_train = list()
        self.neg_test = list()
        self.pos_word_counts = dict()
        self.neg_word_counts = dict()
        self.total_pos = None
        self.total_neg = None
        self.vocabulary = list()
        self.pos_folds = list()
        self.neg_folds = list()
        self.cv_pos_set = list()
        self.cv_neg_set = list()
        self.accuracy = list()

    def run_k_fold_naive_bayes(self, k):

        self.data_setup(True, k)
        for fold_number in range(k):
            self.k_fold_train(fold_number)
            self.test()

        average_accuracy = int(sum(self.accuracy) / len(self.accuracy))

        return "{}-gram accuracy with {} folds is {}%.".format(self.gram_number, k, average_accuracy)

    def train_algorithm(self):

        self.data_setup(False)
        self.word_counts_positive()
        self.word_counts_negative()
        self.calculate_total_numbers_in_pos()
        self.calculate_total_numbers_in_neg()
        self.calculate_vocabulary()

    def data_setup(self, k_fold, k=None):

        self.setup_class_examples()

        if self.gram_number == 2:
            self.bigram_model_class_examples()

        self.data_split(k_fold)

        if k_fold is True:
            self.k_fold_cv_make_groups(k)

        return

    def setup_class_examples(self):

        # Sets up examples for each class and handles data pre-processing.
        self.pos_examples = [list(), list()]
        self.neg_examples = [list(), list()]
        count = 0
        while count < 2:
            if count == 0:
                paths = Path(self.pos_path).glob('**/*.txt')

            else:
                paths = Path(self.neg_path).glob('**/*.txt')

            for file in paths:
                path_in_str = str(file)
                file_content = open(path_in_str, encoding='utf-8').read()
                tokens = nltk.word_tokenize(file_content)
                tokens = [token for token in tokens if token not in string.punctuation]
                tokens = [token.lower() for token in tokens]
                if count == 0:
                    self.pos_examples[0].append(tokens)
                else:
                    self.neg_examples[0].append(tokens)

            count += 1

    def bigram_model_class_examples(self):

        for review in range(len(self.pos_examples[0])):
            self.pos_examples[1].append(list((zip(self.pos_examples[0][review][:-1],
                                                  self.pos_examples[0][review][1:]))))

        for review in range(len(self.neg_examples[0])):
            self.neg_examples[1].append(list(zip(self.neg_examples[0][review][:-1],
                                                 self.neg_examples[0][review][1:])))

    def data_split(self, k_fold):
        self.raw_pos_train = list()
        self.raw_neg_train = list()
        self.pos_train = list()
        self.neg_train = list()

        self.raw_pos_train, self.pos_test, self.raw_neg_train, self.neg_test = train_test_split(
            self.pos_examples[self.gram_number - 1], self.neg_examples[self.gram_number - 1], test_size=(1 / 9))

        if k_fold is False:
            self.pos_train = self.raw_pos_train
            self.neg_train = self.raw_neg_train

    def k_fold_cv_make_groups(self, k):

        self.pos_folds = list()
        self.neg_folds = list()

        random.shuffle(self.raw_pos_train)
        random.shuffle(self.raw_neg_train)
        raw_pos_train_length = len(self.raw_pos_train)
        raw_neg_train_length = len(self.raw_neg_train)

        pos_remainder = raw_pos_train_length % k
        neg_remainder = raw_neg_train_length % k

        pos_fold_length = int((raw_pos_train_length - pos_remainder) / k)
        neg_fold_length = int((raw_neg_train_length - neg_remainder) / k)
        for _ in range(k):
            self.pos_folds.append(self.raw_pos_train[(_ * pos_fold_length):((_ + 1) * pos_fold_length)])
            self.neg_folds.append(self.raw_neg_train[(_ * neg_fold_length):((_ + 1) * neg_fold_length)])

        if pos_remainder != 0:
            self.pos_folds.append(self.raw_pos_train[len(self.raw_pos_train) - pos_remainder:])

        if neg_remainder != 0:
            self.neg_folds.append(self.raw_neg_train[len(self.raw_neg_train) - neg_remainder:])

        return

    def k_fold_train(self, fold_number):
        # Train algorithms separately
        self.k_fold_cv_assign_train_test(fold_number)
        self.word_counts_positive()
        self.word_counts_negative()
        self.calculate_total_numbers_in_pos()
        self.calculate_total_numbers_in_neg()
        self.calculate_vocabulary()

    def k_fold_cv_assign_train_test(self, fold):
        self.pos_train = list()
        self.neg_train = list()
        self.cv_pos_set = list()
        self.cv_neg_set = list()

        for group in self.pos_folds[:fold]:
            for word in group:
                self.pos_train.append(word)

        for group in self.pos_folds[fold + 1:]:
            for word in group:
                self.pos_train.append(word)

        for group in self.neg_folds[:fold]:
            for word in group:
                self.neg_train.append(word)

        for group in self.neg_folds[fold + 1:]:
            for word in group:
                self.neg_train.append(word)

        for word in self.pos_folds[fold]:
            self.cv_pos_set.append(word)

        for word in self.neg_folds[fold]:
            self.cv_neg_set.append(word)

        return

    def word_counts_positive(self):
        self.pos_word_counts = dict()

        for file in range(len(self.pos_train)):
            for word in self.pos_train[file]:

                if word in self.pos_word_counts.keys():
                    self.pos_word_counts[word] += 1

                else:
                    self.pos_word_counts[word] = 1

        return

    def word_counts_negative(self):

        self.neg_word_counts = dict()
        for file in range(len(self.neg_train)):
            for word in self.neg_train[file]:

                if word in self.neg_word_counts.keys():
                    self.neg_word_counts[word] += 1

                else:
                    self.neg_word_counts[word] = 1

        return

    def calculate_total_numbers_in_pos(self):
        self.total_pos = None
        total_pos = list(self.pos_word_counts.values())
        total_pos = sum(total_pos)
        self.total_pos = total_pos

    def calculate_total_numbers_in_neg(self):
        self.total_neg = None
        total_neg = list(self.neg_word_counts.values())
        total_neg = sum(total_neg)
        self.total_neg = total_neg

    def p_word_given_positive(self, word):

        if word not in self.pos_word_counts.keys():

            probability = 1 / (self.total_pos + len(self.vocabulary))

        else:
            probability = (self.pos_word_counts[word] + 1) / (self.total_pos + len(self.vocabulary))

        return probability

    def p_positive(self):
        probability = len(self.pos_train) / (len(self.pos_train) + len(self.neg_train))
        # print(probability)
        return probability

    def p_word_given_negative(self, word):
        if word not in self.neg_word_counts.keys():

            probability = 1 / (self.total_neg + len(self.vocabulary))
        else:
            probability = (self.neg_word_counts[word] + 1) / (self.total_neg + len(self.vocabulary))

        return probability

    def p_negative(self):
        probability = len(self.neg_train) / (len(self.neg_train) + len(self.pos_train))
        # print(probability)
        return probability

    def calculate_vocabulary(self):
        self.vocabulary = list()
        pos_words = list(self.pos_word_counts.keys())
        neg_words = list(self.neg_word_counts.keys())

        pos_words.extend(neg_words)
        for word in set(pos_words):
            self.vocabulary.append(word)

    def test_set(self, file_index, files):

        prior_pos_probability = math.log(self.p_positive())
        prior_neg_probability = math.log(self.p_negative())
        likelihood_pos_probability = 0
        likelihood_neg_probability = 0

        for word in files[file_index]:
            likelihood_pos_probability += math.log((self.p_word_given_positive(word)))
            likelihood_neg_probability += math.log((self.p_word_given_negative(word)))

        pos_probability = prior_pos_probability + likelihood_pos_probability
        neg_probability = prior_neg_probability + likelihood_neg_probability

        if max(pos_probability, neg_probability) == pos_probability:
            return 'POS'

        else:
            return 'NEG'

    def test(self):

        errors = 0
        for file in range(len(self.cv_pos_set)):
            result = self.test_set(file, self.cv_pos_set)
            if result != 'POS':
                errors += 1

        for file in range(len(self.cv_neg_set)):
            result = self.test_set(file, self.cv_neg_set)
            if result != 'NEG':
                errors += 1

        error_ratio = (errors / (len(self.cv_pos_set) + len(self.cv_neg_set)))
        self.accuracy.append(int(100 * (1 - error_ratio)))


# uni_gram_nlp = naive_bayes_algorithm(1)
# bi_gram_nlp = naive_bayes_algorithm(2)
# print(uni_gram_nlp.run_k_fold_naive_bayes(8))
# print(bi_gram_nlp.run_k_fold_naive_bayes(8))
