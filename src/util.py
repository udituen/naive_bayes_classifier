import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download("stopwords")


def preprocess(path):
    with open(path, 'r', encoding="utf8") as file:
        sentence = csv.reader(file, quotechar='"')
        next(sentence)
        word_feature = []
        row_id = []
        stops = set(stopwords.words('english'))
        for row in sentence:
            w = []
            for word in row[1].split(' '):
                # check if work is alphanumeric, else it skips
                if word.isalnum() and word not in stops:
                    # convert words to lower case
                    w.append(word.lower())
            word_feature.append((w, row[2]))
            row_id.append(row[0])
    return word_feature, row_id


def get_counts(token_):
    # dictionary containing count of sentences belonging to individual class
    feature_count = {}
    # dic containing number of words in each class. this would be a two dimensional dictionary
    word_count = {}
    # count of words belonging to a feature
    feature_word_count = {}

    vocab_len = 0

    vocabulary = []

    for words, feature in token_:
        # if a new feature is identified, add to the dictionary
        if feature not in feature_count.keys():
            feature_count[feature] = 1
            #
            feature_word_count[feature] = 0
            # instantiate dictionary to store sentence count of each feature
            word_count[feature] = {}
        else:
            feature_count[feature] += 1

        for word in words:
            # add to vocab_len

            feature_word_count[feature] += 1

            if word in word_count[feature].keys():
                word_count[feature][word] += 1
            else:
                word_count[feature][word] = 1
                vocab_len += 1
                vocabulary.append(word)

    return feature_count, word_count, feature_word_count, vocab_len, vocabulary


def check_accuracy(result):
    right_pred = 0
    wrong = []
    for actual_pred in result:
        if actual_pred[0] == actual_pred[1]:
            right_pred += 1
        else:
            wrong.append(actual_pred)
    return right_pred / len(result)
