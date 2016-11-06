%matplotlib inline

# General libraries.
import re
import numpy as np
import matplotlib.pyplot as plt

# SK-learn libraries for learning
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn library for importing the newsgroup data.
from sklearn.datasets import fetch_20newsgroups

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)

num_test = len(newsgroups_test.target)
test_data, test_labels = newsgroups_test.data[int((num_test-1)/2):], newsgroups_test.target[int((num_test-1)/2):]
dev_data, dev_labels = newsgroups_test.data[:int((num_test-1)/2)], newsgroups_test.target[:int((num_test-1)/2)]
train_data, train_labels = newsgroups_train.data, newsgroups_train.target

print('training label shape:', train_labels.shape)
print('test label shape:', test_labels.shape)
print('dev label shape:', dev_labels.shape)
print('labels names:', newsgroups_train.target_names)


def P1(train_data, train_labels, num_examples=5):
    for i in range(0,num_examples):
        print("training example: ", train_data[i])
        print("label: ", train_labels[i])
        print()

def P2(train_data, test_data):
    # default vectorizer
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    print("Size of the vocabulary: ", X_train.shape[1])

    print("Number of non-zero features per example: ", X_train.nnz/(X_train.shape[1]*X_train.shape[0]))
    print("Fraction of non-zero entries:", (X_train.nnz/(X_train.shape[1]*X_train.shape[0])*100))

    print("First feature string: ", vectorizer.get_feature_names()[0])
    print("Last feature string: ", vectorizer.get_feature_names()[-1])

    vocab = ["atheism", "graphics", "space", "religion"]
    vectorizer = CountVectorizer()
    vocab_train = vectorizer.fit_transform(vocab)
    print("Number of non-zero features per example for given vocabulary: ", vocab_train.nnz/(vocab_train.shape[1]*vocab_train.shape[0]))

    # bigram features
    ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
    X_train_bigram = ngram_vectorizer.fit_transform(train_data)
    print("Size of vocabulary for bigram features: ", X_train_bigram.shape[1])

    # trigram features
    ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3))
    X_train_trigram = ngram_vectorizer.fit_transform(train_data)
    print("Size of vocabulary for trigram features: ", X_train_trigram.shape[1])

    # prune words that appear in less than 10 documents
    vectorizer10 = CountVectorizer(min_df = 10)
    X_train_10 = vectorizer10.fit_transform(train_data) 
    print("Size of vocabulary for pruned data: ", X_train_10.shape[1])

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_dev = vectorizer.transform(dev_data)
    print("Fraction missing: ", abs((X_dev.shape[0] - X_train.shape[0])/X_train.shape[0]*100))
    return X_train, X_test

[X_train, X_test] = P2(train_data, test_data)


def P3(X_train, X_test, train_labels, test_labels):
    warnings.filterwarnings('ignore')

    # find optimal k value for knn
    max_score = 0
    best_n = 0

    for i in range(1,10):
        knn = KNeighborsClassifier(i).fit(X_train, train_labels)
        y_fitted = knn.predict(X_test)

        f_score = metrics.f1_score(test_labels, y_fitted)
        if f_score > max_score:
            max_score = f_score
            best_n = i

    print("Maximum f1 score for k nearest neighbors: ", max_score)
    print("Best number of neighbors: ", best_n)

    # find optimal alpha value for multinomial naive Bayes
    max_score = 0
    best_a = 0
    alpha_list = [0, 0.1, 0.3, 0.5, 0.8, 1, 1.3, 1.5, 1.8, 2]

    for i in alpha_list:
        naive_Bayes = MultinomialNB(alpha = 0).fit(X_train, train_labels)
        y_fitted = naive_Bayes.predict(X_test)

        f_score = metrics.f1_score(test_labels, y_fitted)
        if f_score > max_score:
            max_score = f_score
            best_a = i

    print("Maximum f1 score for multinomial naive Bayes: ", max_score)
    print("Best alpha value: ", best_a)

    # find optimal C reguralization term
    max_score = 0
    best_C = 0
    C_list = [0.1, 0.3, 0.5, 0.8, 1, 1.3, 1.5, 1.8, 2]

    for i in C_list:
        log_regression = LogisticRegression(penalty='l2', C=i).fit(X_train, train_labels)
        y_fitted = log_regression.predict(X_test)

        f_score = metrics.f1_score(test_labels, y_fitted)
        if f_score > max_score:
            max_score = f_score
            best_C = i
            best_log_regression = log_regression

        print("Current C value: ", i)
        print("Squared sum of coefficients for each label: ")
        for j in list(log_regression.coef_):
            print(float(sum(list(j)))**2)

    print("Maximum f1 score for logistic regression: ", max_score)
    print("Best C value: ", best_C)
    return best_C, best_log_regression

[best_C, log_regression] = P3(X_train, X_test, train_labels, test_labels)

def P4(train_data, log_regression, X_train, train_labels, best_C):
    print("Unigram features table: ")
    iter = 0
    for i in list(log_regression.coef_):
        print("Label: ", iter)
        keys_per_label = sorted(range(len(i)),key=i.__getitem__, reverse=True)
        keys_per_label = keys_per_label[:5]

        for k in list(log_regression.coef_):
            for j in keys_per_label:
                print(k[j], end=" ")
            print()
        iter += 1
        print()

    ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
    X_train_bigram = ngram_vectorizer.fit_transform(train_data)
    log_regression = LogisticRegression(penalty='l2', C=best_C).fit(X_train_bigram, train_labels)

    print("Bigram features table: ")
    iter = 0
    for i in list(log_regression.coef_):
        print("Label: ", iter)
        keys_per_label = sorted(range(len(i)),key=i.__getitem__, reverse=True)
        keys_per_label = keys_per_label[:5]

        for k in list(log_regression.coef_):
            for j in keys_per_label:
                print(k[j], end=" ")
            print()
        iter += 1
        print()

P4(train_data, log_regression, X_train, train_labels, best_C)

def empty_preprocessor(train_data, test_data):
    return train_data, test_data

def better_preprocessor(train_data, test_data):
    for i in train_data:
        i = i.lower()

    for i in train_data:
        regex = re.compile('[^a-zA-Z 0-9]')
        i = regex.sub('', i)

    for i in test_data:
        i = i.lower()
    for i in test_data:
        regex = re.compile('[^a-zA-Z 0-9]')
        i = regex.sub('', i)

    return train_data, test_data

[train_data2, test_data2] = better_preprocessor(train_data, test_data)

def P5(X_train_orig, train_data2,test_data2,train_labels):
    # preprocessed data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data2)
    X_test = vectorizer.transform(test_data2)

    log_regression = LogisticRegression(penalty='l2', C=best_C).fit(X_train, train_labels)
    y_fitted = log_regression.predict(X_test)

    f_score = metrics.f1_score(test_labels, y_fitted)
    print("Vocabulary size for original training X: ", X_train_orig.shape[1])
    print("Vocabulary size for preprocessed training X: ", X_train.shape[1])
    return f_score

f_new_score = P5(X_train, train_data2, test_data2, train_labels)

def P6(X_train, X_test, train_labels, test_labels):
    log_regression = LogisticRegression(penalty='l1', C=best_C).fit(X_train, train_labels)
    listPos = [x != 0 for x in log_regression.coef_]
    sumTruel1 = 0
    for i in listPos:
        sumTruel1 += sum(list(i))

    # remove features that are 0 for all the labels
    columns_to_del = []
    for iter in range(len(log_regression.coef_[0])):
        counter = 0
        for i in log_regression.coef_:
            if list(i)[iter] == 0:
                counter += 1
        
        if counter == 4:
            columns_to_del.append(iter)

    all_cols = np.arange(X_train.shape[1])
    cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, columns_to_del)))[0]
    new_X_train = X_train[:, cols_to_keep]
    new_X_test = X_test[:, cols_to_keep]

    # retrain the model with new sparse matrix and plot f1 score for each C value
    log_regression = LogisticRegression(penalty='l2', C=best_C).fit(new_X_train, train_labels)
    
    C_list = [0.1, 0.3, 0.5, 0.8, 1, 1.3, 1.5, 1.8, 2]
    f_score_list = []
    C_param_list = []
    for c in C_list:
        log_regression = LogisticRegression(penalty='l2', C=c).fit(new_X_train, train_labels)
        y_fitted = log_regression.predict(new_X_test)
        f_score = metrics.f1_score(test_labels, y_fitted)

        f_score_list.append(f_score)
        C_param_list.append(c)

    plt.plot(C_param_list, f_score_list)        

P6(X_train, X_test, train_labels, test_labels)

def P7(train_data, test_data):
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)

    log_regression = LogisticRegression(penalty='l2', C=100).fit(X_train, train_labels)
    y_fitted = log_regression.predict(X_test)

    ratio_list = []
    iter = 0
    for i in log_regression.predict_proba(X_train):
        ratio_list.append(max(list(i))/list(i)[y_fitted[iter]])
        iter = iter + 1
        if iter == 677:
            break
    
    keys = sorted(range(len(ratio_list)),key=ratio_list.__getitem__, reverse=True)
    ratio_list = sorted(ratio_list, reverse=True)
    ratio_top3 = []
    for i in range(3):
        ratio_top3.append(train_data[keys[i]])

    for i in range(3):
        print(ratio_top3[i])

P7(train_data, test_data)

def P8(train_data, test_data, train_labels, test_labels):
    # use tf-idf vectorizer together with data preprocessing and choosing optimal C value
    [train_data, test_data] = better_preprocessor(train_data, test_data)
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    
    C_list = [0.1, 0.3, 0.5, 0.8, 1, 1.3, 1.5, 1.8, 2, 2.3, 2.5, 2.8, 3]
    f_score_list = []
    C_param_list = []
    for c in C_list:
        log_regression = LogisticRegression(penalty='l2', C=c).fit(X_train, train_labels)
        y_fitted = log_regression.predict(X_test)
        f_score = metrics.f1_score(test_labels, y_fitted)

        f_score_list.append(f_score)
        C_param_list.append(c)

    plt.plot(C_param_list, f_score_list)

P8(train_data, test_data, train_labels, test_labels)