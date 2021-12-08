# step 1: read argument src/main.py --train data/train.csv --test data/test.csv --output output/test.csv
# Step 2: Train model with training file, and print accuracy using 3-fold cross validation
# Step 3: test model with test file, make predictions output in file.. print accuracy on test set

# report --- confusion matrix with precision and recall,, aggregated micro and macro average precision

# tokenization, model parameters, unknown words, stop words
# explain choices for handling above.

# explain classification, giving justification
#  output column: original_label, output_label, row_id


import argparse
from train import NaiveBayes, train
from test import test
from util import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="specify the train path")
    parser.add_argument("--test", help="specify the test path")
    parser.add_argument("--output", help="specify the output path")
    args = parser.parse_args()
    # print(args.train, args.test, args.output)
    # extract word, class and IDs from input file

    token_relation, row_id = preprocess(args.train)

    # apply 3-fold cross validation
    # train_accuracy = train(token_relation)
    # print(f'accuracy of the train set is {train_accuracy}')

    test(args.test, args.output, args.train)


if __name__ == "__main__":
    main()
