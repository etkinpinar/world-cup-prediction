import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import *
from math import sqrt

class KNN:
    def __init__(self, team_name, championships, num_neighbors):
        self.team_name = team_name
        self.championships = championships
        self.num_neighbours = num_neighbors
        self.X = None
    def train_set_process(self, teamid_matches) -> object:
        X = teamid_matches.loc[:,
            ['Home Team Name', 'Away Team Name', 'Home Team Championship', 'Away Team Championship', 'Winner']]
        X = np.array(X).astype('float64')

        _X = X.copy()

        _X[:, 0] = X[:, 1]
        _X[:, 1] = X[:, 0]
        _X[:, 2] = X[:, 3]
        _X[:, 3] = X[:, 2]

        for i in range(len(_X)):
            if _X[i][-1] == 1:
                _X[i][-1] = 2
            elif _X[i][-1] == 2:
                _X[i][-1] = 1

        X = np.concatenate((X, _X), axis=0)
        # shuffle
        self.X = shuffle(X)
        return self.X

    def euclidean_distance(self, row1, row2):
        distance = 0.0
        row1 = row1.flatten()
        for i in range(len(row2) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    def get_neighbors(self, train, test_row):
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.num_neighbours):
            neighbors.append(distances[i][0])
        return neighbors

    def predict_classification(self, test_row):
        neighbors = self.get_neighbors(self.X, test_row)
        output_values = [row[-1] for row in neighbors]
        win_prob = output_values.count(1) / self.num_neighbours
        lose_prob = output_values.count(2) / self.num_neighbours
        draw_prob = output_values.count(0) / self.num_neighbours
        return draw_prob, win_prob, lose_prob

    def prediction(self, team1, team2):
        id1 = self.team_name[team1]
        id2 = self.team_name[team2]
        championship1 = self.championships.get(team1) if self.championships.get(team1) != None else 0
        championship2 = self.championships.get(team2) if self.championships.get(team2) != None else 0
        x = np.array([id1, id2, championship1, championship2]).astype('int')
        x = np.reshape(x, (1, -1))
        draw_prob, win_prob, lose_prob = self.predict_classification(x)
        text = (
                'Chance for ' + team1 + ' to win ' + team2 + ' is {}\nChance for ' + team2 + ' to win ' + team1 + ' is {}\nChance for ' + team1 + ' and ' + team2 + ' draw is {}').format(
            win_prob * 100, lose_prob * 100, draw_prob * 100)
        return draw_prob, text

    def accuracy(self):
        y_true = self.X[:, 4]
        y_pred = list()
        for row in self.X:
            probs = self.predict_classification(row[:4])
            max_index = probs.index(max(probs))
            y_pred.append(max_index)
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        confusion_mtx = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        return acc, recall,  precision, confusion_mtx