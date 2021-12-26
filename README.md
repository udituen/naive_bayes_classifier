
## 3-rd Party Libraries
You do not need to list `nltk` and `pandas` here.

* `test.py L:[47,49]` used `[sklearn]` for [displaying confusion metric and classification report].
* `train.py L:[64]` used `[sklearn]` for [implementing kfold cross validation].


## Execution

To execute, simply run the sample command line argument below. paths are dynamic and thus can be changed.

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`

## Data

The project's training data can be found in [data/train.txt](data/train.txt),and the in-domain test data can be found in [data/test.txt](data/test.txt).


## Acknowledgement 

https://web.stanford.edu/~jurafsky/slp3/4.pdf
https://www.programiz.com/python-programming/methods/string/isalnum
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
https://github.com/jmkovachi/sent-classifier/blob/master/classifiers/NaiveBayes.py

Done in partnership with Bintemoin Maisha 
