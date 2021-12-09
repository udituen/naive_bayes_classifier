
import argparse
from train import train
from test import test
from util import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="specify the train path")
    parser.add_argument("--test", help="specify the test path")
    parser.add_argument("--output", help="specify the output path")
    args = parser.parse_args()

    # extract word, class and IDs from input file
    token_relation, row_id = preprocess(args.train)

    # apply 3-fold cross validation
    train_accuracy = train(token_relation)
    print(f'Average accuracy of the train set is {train_accuracy}')

    # # train bayes for use in the test set
    test(args.test, args.output, args.train)


if __name__ == "__main__":
    main()
