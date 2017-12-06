'''
Question 1 Skeleton Code


'''
import sklearn
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from time import time

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    print("============== feature names ===============", feature_names)
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


# helper function for confusion matrix
def confusion_matrix()
    pass


if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)


    # Algo 1: Linear SVM classification model with Pipeline
    to_remove = ('headers', 'footers', 'quotes')  # remove for more realistic prediction
    newsgroups_train = fetch_20newsgroups(subset='train', remove=to_remove)
    newsgroups_test = fetch_20newsgroups(subset='test', remove=to_remove)

    for ind, category in enumerate(fetch_20newsgroups(subset='all', remove=to_remove).target_names):
        print(ind, category)

    clf2 = Pipeline([('vector', CountVectorizer()),
                     ('tfidf_trans', TfidfTransformer()),
                     ('clf', SGDClassifier())])
    # Tuning parameter using grid search
    parameters = {'clf__alpha': (1e-2, 1e-3),
                  'clf__penalty': ('l2', 'elasticnet'),
                  'clf__n_iter': (5, 10),
                  'tfidf_trans__use_idf': (True, False),
                  'tfidf_trans__norm': ('l1', 'l2'),
                  'vector__ngram_range': ((1, 1), (1, 2)),
                  'vector__max_df': (0.5, 0.75, 1.0),
                  'vector__max_features': (5000, 10000),
                  }
    g_clf2 = GridSearchCV(clf2, parameters, n_jobs=-1)
    t0 = time()
    g_clf2.fit(newsgroups_train.data, newsgroups_train.target)
    # print('Finish fitting in %.5fs' % (time() - t0))
    # TODO: confusion matrix
    test_preds = g_clf2.predict(newsgroups_test.data)