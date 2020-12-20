import random
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json
import pickle


class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentement = self.get_sentement()

    def get_sentement(self):
        if self.score <= 2:
            return "NEGATIVE"
        elif self.score == 3:
            return "NEUTRAL"
        else:
            return "POSITIVE"


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentement(self):
        return [x.sentement for x in self.reviews]

    def distrbuted(self):
        # do a map throw all the data and find the data that has a sentemant of negative
        negative = list(filter(lambda x: x.sentement == 'NEGATIVE', self.reviews))
        positive = list(filter(lambda x: x.sentement == 'POSITIVE', self.reviews))
        positive_shunk = positive[:len(negative)]
        self.reviews = negative + positive_shunk
        random.shuffle(self.reviews)


if __name__ == '__main__':
    reviews = []
    # read the fill line by line
    with open('Books_small_10000.json') as f:
        for line in f:
            review = json.loads(line)
            reviews.append(Review(review['reviewText'], review['overall']))

    # spilt the data into two part test and train
    train, test = train_test_split(reviews, test_size=0.33, random_state=42)
    # pass the train data to the class
    train_container = ReviewContainer(train)
    # pass the test data to the class
    test_container = ReviewContainer(test)
    # call the method that make the data balance
    train_container.distrbuted()
    # call the function that will make the test data balance
    test_container.distrbuted()
    # get the text from the data and save it as the train  data
    train_x = train_container.get_text()
    # get the score and sentement and save it as the label
    train_y = train_container.get_sentement()
    # get the test data and  save it
    test_X = test_container.get_text()
    # get the test label and save it
    test_y = test_container.get_sentement()
    # now we will use bag of words to convert the text to a numerical numbers
    vec = TfidfVectorizer()
    X = vec.fit_transform(train_x)
    X_test = vec.transform(test_X)
    # use the svm model to train our data
    clf = svm.SVC(kernel='rbf', C=10)
    clf.fit(X, train_y)
    clf.predict(X_test)

#     we can use another algo to train our model like logistic re

    clf_log = LogisticRegression(max_iter=1000).fit(X, train_y)
    pre = clf_log.predict(X_test)
#     evaluation for your model to see how it work
    print(clf.score(X_test, test_y))
    print(clf_log.score(X_test, test_y))
#     the score function give you the mean accuracy of the model
    pres_log = clf_log.score(X_test, test_y)
    pres_svm = clf.score(X_test, test_y)
#     f1 score
    f1 = f1_score(test_y, clf.predict(X_test), average=None)
#     use GridSearchCV to select the best parameters for you
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 23, 24, 32]}
    svc = svm.SVC()
    clf2 = GridSearchCV(svc, parameters, cv=5)
    clf2.fit(X, train_y)
    # this is how to get the best parameters from gridsearch
    pre = clf2.best_params_
    print(pre)
#     saving our model
    with open('./models/sem.pkl', 'wb') as f:
        pickle.dump(clf, f)

#   how to load the fill
    with open('./models/sem.pkl', 'rb') as f:
        loaded_clf = pickle.load(f)

#     use the loaded model to predict the answer
    print(test_X[0])
    lpre = loaded_clf.predict(X[0])
    print(lpre)










