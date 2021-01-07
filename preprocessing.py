import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import numpy as np

class Preprocessing:
    def read_csv_files(self, matches_path, players_path, cups_path):
        matches = pd.read_csv(matches_path)
        players = pd.read_csv(players_path)
        cups = pd.read_csv(cups_path)
        self.matches = matches.dropna()
        self.players = players.dropna()
        self.cups = cups.dropna()
        matches.head()

    @staticmethod
    def replace_name(df):
        if (df['Home Team Name'] in ['German DR', 'Germany FR']):
            df['Home Team Name'] = 'Germany'
        elif (df['Home Team Name'] == 'Soviet Union'):
            df['Home Team Name'] = 'Russia'

        if (df['Away Team Name'] in ['German DR', 'Germany FR']):
            df['Away Team Name'] = 'Germany'
        elif (df['Away Team Name'] == 'Soviet Union'):
            df['Away Team Name'] = 'Russia'
        return df


    def fix_wrong_team_names(self):
        names = self.matches[self.matches['Home Team Name'].str.contains('rn">')]['Home Team Name'].value_counts()
        wrong = list(names.index)
        correct = [name.split('>')[1] for name in wrong]
        old_name = ['Maracan� - Est�dio Jornalista M�rio Filho', 'Estadio do Maracana']
        new_name = ['Maracan Stadium', 'Maracan Stadium']
        wrong = wrong + old_name
        correct = correct + new_name

        for index, wr in enumerate(wrong):
            self.matches = self.matches.replace(wrong[index], correct[index])
            self.matches = self.matches.apply(Preprocessing.replace_name, axis='columns')

    def create_dictionary_of_football_team(self):
        # create a dictionary of football team
        team_name = {}
        index = 0
        for idx, row in self.matches.iterrows():
            name = row['Home Team Name']
            if (name not in team_name.keys()):
                team_name[name] = index
                index += 1
            name = row['Away Team Name']
            if (name not in team_name.keys()):
                team_name[name] = index
                index += 1
        self.team_name = team_name
        return team_name

    def drop_unnecessary_columns(self):
        self.dropped_matches = self.matches.drop(
            ['Datetime', 'Stadium', 'Referee', 'Assistant 1', 'Assistant 2', 'RoundID', 'Win conditions',
             'Home Team Initials', 'Away Team Initials', 'Half-time Home Goals', 'Half-time Away Goals',
             'Attendance', 'City', 'MatchID', 'Stage'], 1)

    def fix_cups_names(self):
        self.championships = self.cups['Winner'].map(lambda p: 'Germany' if p == 'Germany FR' else p).value_counts()
        plt.figure(figsize=(18, 6))
        sns.countplot(self.cups['Winner'].map(lambda p: 'Germany' if p == 'Germany FR' else p))
        return self.championships

    def count_championships(self):
        self.dropped_matches['Home Team Championship'] = 0
        self.dropped_matches['Away Team Championship'] = 0

        def count_championship(df):
            if (self.championships.get(df['Home Team Name']) != None):
                df['Home Team Championship'] = self.championships.get(df['Home Team Name'])
            if (self.championships.get(df['Away Team Name']) != None):
                df['Away Team Championship'] = self.championships.get(df['Away Team Name'])
            return df

        self.dropped_matches = self.dropped_matches.apply(count_championship, axis='columns')
        print(self.dropped_matches)
        self.dropped_matches['Winner'] = '-'

    def find_winners(self):
        def find_winner(df):
            if (int(df['Home Team Goals']) == int(df['Away Team Goals'])):
                df['Winner'] = 0
            elif (int(df['Home Team Goals']) > int(df['Away Team Goals'])):
                df['Winner'] = 1
            else:
                df['Winner'] = 2
            return df
        self.dropped_matches = self.dropped_matches.apply(find_winner, axis='columns')
        print(self.dropped_matches)
        return self.dropped_matches


    def replace_team_name_by_id(self, df):
        df['Home Team Name'] = self.team_name[df['Home Team Name']]
        df['Away Team Name'] = self.team_name[df['Away Team Name']]
        # df['Winner'] = team_name[df['Winner']]
        return df

    def replace_team_names_by_id(self):
        self.teamid_matches = self.dropped_matches.apply(self.replace_team_name_by_id, axis='columns')
        print(self.teamid_matches)

        teamid_matches = self.teamid_matches.drop(['Year', 'Home Team Goals', 'Away Team Goals'], 1)
        return teamid_matches

