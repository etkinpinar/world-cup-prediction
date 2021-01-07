from interface.interface import run
from preprocessing import *
from test_and_train import KNN


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

    knn = KNN(team_name, championships, 17)
    X = knn.train_set_process(teamid_matches)
    print(X)

    # predict match between France and Uruguay
    print(list(team_name.keys()))
    #run(list(team_name.keys()), knn)

    prob1, text1 = knn.prediction('Sweden', 'Uruguay')
    print(text1)
    prob1, text1 = knn.prediction('France', 'Romania')
    print(text1)
    acc, recall, precision, conf_mtx = knn.accuracy()
    print("Accuracy: ", acc)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Confusion Matrix:\n", conf_mtx)

