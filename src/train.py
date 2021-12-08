import operator

from util import get_counts, check_accuracy
from sklearn.model_selection import KFold


def calculate_likelihood(word_count, feature_word_count, vocab):
    # handle unknown words
    likelihood = {}
    for feature in word_count.keys():
        likelihood[feature] = {}
        likelihood[feature]['<UNK>'] = 1 / (feature_word_count[feature] + vocab)
        for word, count in word_count[feature].items():
            likelihood[feature][word] = (count + 1) / (feature_word_count[feature] + vocab)
    return likelihood


def calculate_prior(feature_count, document_count):
    prior = {}
    for feature, count in feature_count.items():
        # print(count)
        prior[feature] = int(count) / int(document_count)
    return prior


class NaiveBayes:

    def __init__(self):
        self.likelihood = {}
        self.prior_probability = {}
        self.vocab_list = {}

    def train_bayes(self, processed_token):
        # get counts for each feature
        feature_count, word_count, feature_word_count, vocab_len, vocab_list = get_counts(processed_token)

        self.prior_probability = calculate_prior(feature_count, len(processed_token))

        self.likelihood = calculate_likelihood(word_count, feature_word_count, vocab_len)

        self.vocab_list = vocab_list

    # predict max probability from document sentence

    def train_accuracy(self, train_set):
        feature = list(self.prior_probability.keys())
        predict_probs = {}

        for class_ in feature:
            predict_prob = 1
            for word in train_set:
                # processes only words in vocab and smooth unknown words
                if word in self.vocab_list:
                    if word not in self.likelihood[class_].keys():
                        predict_prob *= self.likelihood[class_]['<UNK>']
                    else:
                        predict_prob *= self.likelihood[class_][word]

                predict_probs[class_] = self.prior_probability[class_] * predict_prob

        return max(predict_probs.items(), key=operator.itemgetter(1))[0]


def train(sentence):
    kf = KFold(n_splits=3)
    kf.get_n_splits(sentence)
    mean_accuracy = 0
    for train_index, test_index in kf.split(sentence):

        train_set = []
        test_set = []
        result = []
        for i in train_index:
            train_set.append(sentence[i])
        for j in test_index:
            test_set.append(sentence[j])

        # call naiveBayes
        NB = NaiveBayes()
        NB.train_bayes(train_set)

        for i in range(len(test_set)):
            predicted_val = NB.train_accuracy(test_set[i][0])

            original = test_set[i][1]

            result.append((original, predicted_val))

        mean_accuracy += check_accuracy(result) / 3

    return mean_accuracy
