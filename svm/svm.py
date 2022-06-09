import sys
sys.path.append("..")
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from load_data import load_cifar10, extract_hog

trainset, _, testset, _, classes = load_cifar10()

classifier = svm.SVC(C=0.1, gamma=1e-7, decision_function_shape="ovr", max_iter=100000, probability=True)

train_data = extract_hog(trainset.data)
classifier.fit(train_data, trainset.targets)

test_data = extract_hog(testset.data)
y_true = testset.targets
y_pred = classifier.predict(test_data)
y_score = classifier.predict_proba(test_data)
print(len(y_pred), len(y_true))

res = roc_auc_score(y_true=y_true, y_score=y_score, multi_class="ovr", labels=[i for i in range(10)])
print(res)