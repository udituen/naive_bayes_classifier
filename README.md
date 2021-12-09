# Intro to NLP - Assignmen5

## Team
| Student name     |  CCID     | 
|------------------|-----------|
| Uduak Ituen      | Ituen     |
| Maisha Binte Moin| Bintemoi|      

## TODOs

In this file you **must**:
- [ ] Fill out the team table above. 
- [ ] Make sure you submitted the URL on eClass.
- [ ] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment.
- [ ] Provide clear installation and execution instructions that TAs must follow to execute your code.
- [ ] List where and why you used 3rd-party libraries.
- [ ] Delete the line that doesn't apply to you in the Acknowledgement section.

## 3-rd Party Libraries
You do not need to list `nltk` and `pandas` here.

* `test.py L:[47,49]` used `[sklearn]` for [displaying confusion metric and classification report].
* `train.py L:[64]` used `[sklearn]` for [implementing kfold cross validation].


## Execution

To execute, simply run the sample command line argument below. paths are dynamic and thus can be changed.

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`

## Data

The assignment's training data can be found in [data/train.txt](data/train.txt),and the in-domain test data can be found in [data/test.txt](data/test.txt).


## Acknowledgement 
In accordance with the UofA Code of Student Behaviour, we acknowledge that  
(**delete the line that doesn't apply to you**)

- We have listed all external resources we consulted for this assignment.

https://web.stanford.edu/~jurafsky/slp3/4.pdf
https://www.programiz.com/python-programming/methods/string/isalnum
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
https://github.com/jmkovachi/sent-classifier/blob/master/classifiers/NaiveBayes.py
