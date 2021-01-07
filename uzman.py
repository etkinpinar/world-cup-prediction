from interface.interface import run
from preprocessing import *
from test_and_train import SVM


if __name__ == "__main__":
    preprocessing = Preprocessing()
    preprocessing.read_csv_files("./dataset/WorldCupMatches.csv", './dataset/WorldCupPlayers.csv',
                                                 './dataset/WorldCups.csv')

    preprocessing.fix_wrong_team_names()
    team_name = preprocessing.create_dictionary_of_football_team()
    preprocessing.drop_unnecessary_columns()
    championships = preprocessing.fix_cups_names()
    preprocessing.count_championships()
    dropped_matches = preprocessing.find_winners()
    teamid_matches = preprocessing.replace_team_names_by_id()

    svm = SVM(team_name, championships)
    X_train, X_test, y_train, y_test = svm.train_test_split(teamid_matches, dropped_matches)
    print(X_train, y_train)
    svm.fit(svm.X, svm.y, X_train, X_test, y_train, y_test)

    # predict match between France and Uruguay
    print(list(team_name.keys()))
    run(list(team_name.keys()), svm)

    prob1, text1 = svm.prediction('France', 'Uruguay')
    print(text1)

