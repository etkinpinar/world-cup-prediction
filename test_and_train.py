import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

class SVM:
    def __init__(self, team_name, championships):
        self.team_name = team_name
        self.championships = championships
        self.X = None
        self.y = None

    def train_test_split(self, teamid_matches, dropped_matches) -> object:
        X = teamid_matches.loc[:,
            ['Home Team Name', 'Away Team Name', 'Home Team Championship', 'Away Team Championship']]
        X = np.array(X).astype('float64')

        # exchange 'home team name' with 'away team name', 'home team championship' with 'away team championship', and replace the result
        _X = X.copy()

        _X[:, 0] = X[:, 1]
        _X[:, 1] = X[:, 0]
        _X[:, 2] = X[:, 3]
        _X[:, 3] = X[:, 2]

        y = dropped_matches.loc[:, ['Winner']]
        y = np.array(y).astype('float64')
        y = np.reshape(y, (1, 850))

        y = y[0]

        _y = y.copy()

        for i in range(len(_y)):
            if _y[i] == 1:
                _y[i] = 2
            elif _y[i] == 2:
                _y[i] = 1

        X = np.concatenate((X, _X), axis=0)

        y = np.concatenate((y, _y))

        # shuffle
        self.X, self.y = shuffle(X, y)
        # split test, train
        return train_test_split(X, y, test_size=0.2)


    def fit(self, X, y, x_train, x_test, y_train, y_test):
        param_grid = {'C': [1e3],
                      'gamma': [0.0001]}
        self.svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True)
        self.svm_model.fit(X, y)

        print("Predicting on the test set")
        # t0 = time()
        y_pred = self.svm_model.predict(x_test)
        # print("done in %0.3fs" % (time() - t0))
        print(self.svm_model.score(x_test, y_test))
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred, labels=range(3)))

    def prediction(self, team1, team2):
        id1 = self.team_name[team1]
        id2 = self.team_name[team2]
        championship1 = self.championships.get(team1) if self.championships.get(team1) != None else 0
        championship2 = self.championships.get(team2) if self.championships.get(team2) != None else 0
        x = np.array([id1, id2, championship1, championship2]).astype('int')
        x = np.reshape(x, (1, -1))
        _y = self.svm_model.predict_proba(x)[0]
        text = (
                'Chance for ' + team1 + ' to win ' + team2 + ' is {}<br>\nChance for ' + team2 + ' to win ' + team1 + ' is {}<br>\nChance for ' + team1 + ' and ' + team2 + ' draw is {}').format(
            _y[1] * 100, _y[2] * 100, _y[0] * 100)
        return _y[0], text