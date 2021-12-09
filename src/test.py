import csv
from util import preprocess, check_accuracy
from train import NaiveBayes, train
from sklearn.metrics import confusion_matrix, classification_report


def test(test_path, output_path, train_path):
    token_relation, row_id = preprocess(train_path)
    test_token_relation, test_row_id = preprocess(test_path)

    result = []
    accuracy_result = []

    # initialise classifier
    NB = NaiveBayes()
    NB.train_bayes(token_relation)

    for i in range(len(test_token_relation)):

        # print(test_token_relation[i][0])
        predicted_feature = NB.train_accuracy(test_token_relation[i][0])

        original = test_token_relation[i][1]
        # print(len(test_token_relation))
        row_id = test_row_id[i]
        accuracy_result.append([original, predicted_feature])
        result.append([original, predicted_feature, row_id])

    # send final dataset and args object containing the output path to the write_file func
    write_file(output_path, result)

    accuracy = check_accuracy(accuracy_result)

    print(f'Accuracy of the test set is {accuracy}')

    ground_truth = [pair[0] for pair in accuracy_result]
    system = [pair[1] for pair in accuracy_result]

    print(f'Confusion metric: {confusion_matrix(ground_truth, system).transpose()}')

    print(f'Report: {classification_report(ground_truth, system, digits=3)}')


def write_file(path, rows):
    # initialise csv headers
    headers = ['original_label', 'output_label', 'row_id']

    # csv data
    row_total = rows
    with open(path, 'w', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(row_total)
        print(f'Successfully saved to {path}')
