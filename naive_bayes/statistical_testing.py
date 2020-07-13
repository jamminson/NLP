import math
from math import comb
import algorithm

class statistical_testing:

    def __init__(self, base_algorithm, comparison_algorithm):

        self.algorithm1 = base_algorithm
        self.algorithm2 = comparison_algorithm
        self.plus = None
        self.minus = None
        self.null = None

    def sign_test_setup(self):
        null = 0
        plus = 0
        minus = 0

        self.algorithm1.pos_test.extend(self.algorithm2.pos_test)
        self.algorithm1.neg_test.extend(self.algorithm2.neg_test)
        total_pos_test_set = self.algorithm1.pos_test
        total_neg_test_set = self.algorithm2.neg_test

        for file in range(len(total_pos_test_set)):

            algorithm1_result = self.algorithm1.test_set(file, total_pos_test_set)
            algorithm2_result = self.algorithm2.test_set(file, total_pos_test_set)
            if algorithm1_result != algorithm2_result:
                if algorithm1_result != 'POS':
                    plus += 1
                else:
                    minus += 1

            else:
                if algorithm1_result == 'POS':
                    null += 1

        for file in range(len(total_neg_test_set)):

            algorithm1_result = self.algorithm1.test_set(file, total_neg_test_set)
            algorithm2_result = self.algorithm2.test_set(file, total_neg_test_set)

            if algorithm1_result != algorithm2_result:
                if algorithm1_result != 'NEG':
                    plus += 1
                else:
                    minus += 1

            else:
                if algorithm1_result == 'NEG':
                    null += 1

        return plus, minus, null

    def calculate_p_value(self, q):
        p_value = 0
        (self.plus, self.minus, self.null) = self.sign_test_setup()
        n = 2 * int(math.ceil(self.null / 2)) + self.plus + self.minus
        k = int(math.ceil(self.null / 2)) + min(self.plus, self.minus)

        for i in range(k + 1):
            p_value += (comb(n, i) * (q ** i) * ((1 - q) ** (n - i)))
            # p_value += (comb(n, i) * (q ** n))

        return 2 * p_value

    def sign_test(self, q):

        p_value = round(self.calculate_p_value(q) * 100, 1)

        return p_value


unigram = algorithm.naive_bayes_algorithm(1)
bigram = algorithm.naive_bayes_algorithm(2)
unigram.train_algorithm()
bigram.train_algorithm()
please_work = statistical_testing(unigram, bigram)
print(please_work.sign_test(0.5))
