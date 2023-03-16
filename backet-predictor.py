# imports
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
from sklearn.metrics import mean_squared_error, r2_score
import requests
import bs4

# All collected variables for each team
TEAM_DATA_FORMAT = {'Name': 0, 'GP': 0, 'MPG': 0, 'PPG': 0, 'FGM': 0, 'FGA': 0, 'FGP': 0, 'TPM': 0, 'TPA': 0, 'TPP': 0,
                    'FTM': 0, 'FTA': 0, 'FTP': 0, 'ORB': 0, 'DRB': 0, 'RPG': 0, 'APG': 0, 'SPG': 0, 'BPG': 0, 'TOV': 0,
                    'PF': 0, 'Rank': 0}

# Variables used as input
X_FORMAT = ['GP', 'MPG', 'PPG', 'FGM', 'FGA', 'FGP', 'TPM', 'TPA', 'TPP', 'FTM', 'FTA', 'FTP', 'ORB', 'DRB', 'RPG',
            'APG', 'SPG', 'BPG', 'TOV', 'PF']
# Variables used as output
Y_FORMAT = ['Rank']

# ======= Collecting Data =======


# Requires the year of the games which data you want (ex 2022) as well as a dictionary with the key being team names and
# the value being how many games they played at in a given years march madness tournament (with an extra point going to
# the champion team (ex: if a team played their first game and lost their value is 1 and if another team won their first
# game and got knocked out in their second then their value should be 2 and if a team lost the finale it's value
# should be 6 and the winner should be 7) This would mean values are bound from 1-7. The order of the entries does not
# matter. It also requires df, a dataframe following TEAM_DATA FORMAT
# Returns an updated version of the df dataframe with all the year_data entries added as well as their rank


# Requires a dictionary year_data_and_year with the key being the year and the value being year_data Individual year
# dictionary with the key being team names and the value being how many games they played at in a given
# years march madness tournament (with an extra point going to the champion team (ex: if a team played their first game
# # and lost their value is 1 and if another team won their first game and got knocked out in their second then their
# # value should be 2 and if a team lost the finale it's value
# Also has an optional printing variable
# Returns - the dataframe with all the year data added
# Unless printing is turned off, it will also print out the dataframe
def add_data(year_data_and_year, df, printing=True):
    for year in year_data_and_year:
        year_data = year_data_and_year[year]
        df = add_year_data(year, year_data, df)

    if printing:
        print(df)

    return df


def add_year_data(year, year_data, df):
    soup = get_soup("https://basketball.realgm.com/ncaa/team-stats/{}/Averages/Team_Totals/0".format(year))
    table = soup.find('tbody')

    for row in table:
        cols = row.find_all('td')

        name = str(cols[1].a.text)
        GP = str(cols[2].text)
        MPG = str(cols[3].text)
        PPG = str(cols[4].text)
        FGM = str(cols[5].text)
        FGA = str(cols[6].text)
        FGP = str(cols[7].text)
        TPM = str(cols[8].text)
        TPA = str(cols[9].text)
        TPP = str(cols[10].text)
        FTM = str(cols[11].text)
        FTA = str(cols[12].text)
        FTP = str(cols[13].text)
        ORB = str(cols[14].text)
        DRB = str(cols[15].text)
        RPG = str(cols[16].text)
        APG = str(cols[17].text)
        SPG = str(cols[18].text)
        BPG = str(cols[19].text)
        TOV = str(cols[20].text)
        PF = str(cols[21].text)

        # Checks if entry is one of the teams in the array
        if name in year_data:
            # Creates new frame
            df_entry = create_frame([name, GP, MPG, PPG, FGM, FGA, FGP, TPM, TPA, TPP, FTM, FTA, FTP, ORB, DRB, RPG,
                                     APG, SPG, BPG, TOV, PF, year_data[name]])
            # Adds new frame to current dataframe
            df = pd.concat([df, df_entry], ignore_index=True)
    return df


# Requires - Array consisting of len 21: name, GP, MPG, PPG, FGM, FGA, FGP, TPM, TPA, TPP, FTM, FTA, FTP, ORB, DRB, RPG,
# APG, SPG, BPG, TOV, PF, rank indicating statistics for a specific team
# Returns - the data in a dataframe format following TEAM_DATA_FORMAT
def create_frame(team_data):
    new_data_entry = {'Name': team_data[0], 'GP': team_data[1], 'MPG': team_data[2], 'PPG': team_data[3], 'FGM': team_data[4], 'FGA': team_data[5], 'FGP': team_data[6],
                      'TPM': team_data[7], 'TPA': team_data[8], 'TPP': team_data[9], 'FTM': team_data[10], 'FTA': team_data[11], 'FTP': team_data[12], 'ORB': team_data[13],
                      'DRB': team_data[14], 'RPG': team_data[15], 'APG': team_data[16], 'SPG': team_data[17], 'BPG': team_data[18], 'TOV': team_data[19], 'PF': team_data[20],
                      'Rank': team_data[21]}
    new_data_entry = pd.DataFrame([new_data_entry])
    return new_data_entry


# ======= Creating Model =======


# Requires - x and y data for creating a machine learning model
# Both x and y should be rows from the same dataframe with x being the data that when predicting will be used as input
# and the y data should be filled and represent row(s) that will be output
# It also has an optional printing model
# This specific model uses sklearn linear regression
# Returns - linear regression model which can run predictions based on model trained with the x and y dataframe
# Unless printing is turned off it also prints out the slope, intercept, root mean square error, and R^2 results
def create_model(x, y, printing=True):
    # Initializes Linear Regression Model
    model = sklearn.linear_model.LinearRegression()

    # Trains/Fits the data to the model used
    model.fit(x, y.values)

    # Creates a prediction for the model
    y_predicted = model.predict(x)

    if printing:
        # Calculates Models Root Mean Square Error and R^2 values
        rmse = mean_squared_error(y, y_predicted)
        r2 = r2_score(y, y_predicted)

        # Printing values
        print('Slope:', model.coef_)
        print('Intercept:', model.intercept_)
        print('Root mean squared error: ', rmse)
        print('R2 score: ', r2)

    return model


# ======= Predicting New Results =======


# Requires - a string team_name  (following realgm's website's NCAA basketball naming scheme as well as the machine
# learning model used to make the prediction
# Also has an optional year variable if the year variable to set which year the team data is collected for which is set
# to 2023 by default
# Returns - returns the predicted results being a value representing a teams likelihood of winning with the higher
# value meaning that the team is more likely to do better and win more games
def predict_results(team_name, model, year=2023):

    soup = get_soup("https://basketball.realgm.com/ncaa/team-stats/{}/Averages/Team_Totals/0".format(year))
    table = soup.find('tbody')

    for row in table:
        cols = row.find_all('td')
        name = str(cols[1].a.text)
        if name == team_name:
            GP = float(cols[2].text)
            MPG = float(cols[3].text)
            PPG = float(cols[4].text)
            FGM = float(cols[5].text)
            FGA = float(cols[6].text)
            FGP = float(cols[7].text)
            TPM = float(cols[8].text)
            TPA = float(cols[9].text)
            TPP = float(cols[10].text)
            FTM = float(cols[11].text)
            FTA = float(cols[12].text)
            FTP = float(cols[13].text)
            ORB = float(cols[14].text)
            DRB = float(cols[15].text)
            RPG = float(cols[16].text)
            APG = float(cols[17].text)
            SPG = float(cols[18].text)
            BPG = float(cols[19].text)
            TOV = float(cols[20].text)
            PF = float(cols[21].text)

            # Creates new frame
            prediction = create_frame([name, GP, MPG, PPG, FGM, FGA, FGP, TPM, TPA, TPP, FTM, FTA, FTP, ORB, DRB, RPG,
                                      APG, SPG, BPG, TOV, PF, -1])

            prediction = str(model.predict(prediction[X_FORMAT]))
            prediction = parse_string(prediction, ['[', ']'])

            return prediction


# Requires - a team_list dictionary list with keys being the team names (following realgm's website's NCAA basketball
# team naming scheme) as well as the machine learning model for the prediction
# Also has an optional print variable which can be turned off
# Returns - Doesn't return anything, although it does modify the team_list variable by adding the value to each team
# representing the ML algorithm's estimate on how well the team will do
# Because it runs many large predictions, the algorithm can take a long time to run (maybe 1/2 second per element in
# team_list, so having printing turned on will print its progress to the console
def fill_teams_score(team_list, model, printing=True):
    if printing:
        print('Computing teams', end=': ')
        i = 0
    for team in team_list:
        # Runs the code to calculate the value for each code
        team_list[team] = predict_results(team, model)

        if printing:
            i = i + 1
            print(str(i), end=' -> ')
    if printing:
        print(end="\r\n\n")


# ======= Showing End Results =======


# Requires - a team_list dictionary list with keys being the team names (following realgm's website's NCAA basketball
# naming scheme) organized by how they appear in the bracket (ex team 1 and team 2 play each other and team 3 and team 4
# play each other) and the value representing a teams likelihood of winning with the higher value meaning that the team
# is more likely to do better. The list must also have an even number of teams (to continuously run, it needs to be base
# 2 number of teams)
# Also has an optional print variable which can be turned off
# Returns - doesn't return anything, although it modifies the team_list variable to reduce it by 1/2 by comparing teams
# (team 1 vs team 2, team 3 vs team 4, etc's) value and keeping whichever one has a higher value.
# Unless the print variable is turned off, it also prints each team that remained to the console
def round_of_games(team_list, printing=True):
    # Choose which team wins
    # Based on score from prediction
    delete_list = compare_losing_teams(team_list, printing=False)

    # Remove lost teams from list
    remove_losing_teams(team_list, delete_list)

    # Prints each team remaining in the bracket (didn't lose this round)
    if printing:
        for team in team_list:
            print(team)
        print("")


# Requires - a team_list dictionary list with keys being the team names (following realgm's website's NCAA basketball
# naming scheme) organized by how they appear in the bracket (ex team 1 and team 2 play each other and team 3 and team 4
# play each other) and the value representing a teams likelihood of winning with the higher value meaning that the team
# is more likely to do better. The list must also have an even number of teams (to continuously run, it needs to be base
# 2 number of teams)
# Also has an optional printing variable
# Returns - Returns a list of teams that should be removed based on the team_list dictionary's values
# Unless printing is turned off it will also write to console team 1 vs team 2 and then team 1 wins or team 2 wins
def compare_losing_teams(team_list, printing=True):
    # Array of losing teams to be returned
    losing_teams = []

    # Iterator to access the next team in the list
    # Will throw an error if the number of teams in team_list is not even
    next_team_iterator = iter(team_list)
    next_team = next(next_team_iterator)

    # Used to make it a for by 2 loop
    i = 1

    for team in team_list:
        # Will prevent there be being an error during the last run through
        try:
            next_team = next(next_team_iterator)
            i = i + 1
            if i % 2 == 0:
                if printing:
                    print(team + " vs " + next_team)

                if team_list[team] > team_list[next_team]:
                    losing_teams.append(next_team)
                    if printing:
                        print(team + ' wins', end="\n\n")
                else:
                    losing_teams.append(team)
                    if printing:
                        print(next_team + ' wins', end="\n\n")
        except StopIteration:
            pass
    return losing_teams


# Requires - an all_items dictionary and an array named removed_items
# removed_items should be a subset of all_items (all elements in removed_items must appear in all_items, but not every
# element in all_items needs to be in removed_items)
# Returns - Doesn't return anything, although it does modify the all_items dictionary and deletes every element from it
# that is also in removed_items
def remove_losing_teams(all_items, removed_items):
    for ri in removed_items:
        del all_items[ri]


# ======= Helper Functions =======


# Requires - a url
# Returns - the soup commands to the given url
def get_soup(url):
    r = requests.get(url)
    return bs4.BeautifulSoup(r.content, 'html5lib')


# Requires a string data and a character array removedElements which should have any characters you want removed from
# the string
# Returns the string without any of the characters in removedElements
def parse_string(data, removed_elements):
    parsed_data = ""
    for char in data:
        if char not in removed_elements:
            parsed_data += char
    return parsed_data


# Individual year dictionary with the key being team names and the value being how many games they played at in a given
# years march madness tournament (with an extra point going to the champion team (ex: if a team played their first game
# and lost their value is 1 and if another team won their first game and got knocked out in their second then their
# value should be 2 and if a team lost the finale it's value
year_data_2017 = {'North Carolina': 7, ' Gonzaga': 6, 'South Carolina': 5, 'Oregon': 5, 'Florida': 4, 'Xavier': 4,
                  'Kansas': 4, 'Kentucky': 4, 'Wisconsin': 3, 'Baylor': 3, 'West Virginia': 3, 'Arizona': 3,
                  'Purdue': 3, 'Michigan': 3, 'Butler': 3, 'UCLA': 3, 'Villanova': 2, 'Virginia': 2, 'USC': 2,
                  'Duke': 2, 'Northwestern': 2, 'Notre Dame': 2, 'Florida State': 2, 'Saint Mary\'s': 2,
                  'Michigan State': 2, 'Iowa State': 2, 'Rhode Island': 2, 'Louisville': 2, 'Arkansas': 2,
                  'Middle Tennessee State': 1, 'Cincinnati': 2, 'Wichita State': 2, 'Mount St. Mary\'s': 1,
                  'Virginia Tech': 1, 'UNC Wilmington': 1, 'East Tennessee State': 1, 'Southern Methodist': 1,
                  'New Mexico State': 1, 'Marquette': 1, 'Troy': 1, 'South Dakota State': 1, 'Vanderbilt': 1,
                  'Princeton': 1, 'Bucknell': 1, 'Maryland': 1, 'Florida Gulf Coast': 1, 'Bucknell': 1, 'Maryland': 1,
                  'VCU': 1, 'North Dakota': 1, 'UC Davis': 1, 'Miami (FL)': 1, 'Nevada': 1, 'Vermont': 1,
                  'Creighton': 1, 'Iona': 1, 'Oklahoma State': 1, 'Jacksonville State': 1,
                  'Texas Southern': 1, 'Seton Hall': 1, 'Minnesota': 1, 'Winthrop': 1, 'Kansas State': 1,
                  'Kent State': 1, 'Dayton': 1, 'Northern Kentucky': 1}
year_data_2018 = {'Villanova': 7, 'Michigan': 6, 'Loyola (IL)': 5, 'Kansas': 5, 'Kansas State': 4, 'Florida State': 4,
                  'Texas Tech': 4, 'Duke': 4, 'Kentucky': 3, 'Nevada': 3, 'Gonzaga': 3, 'Texas A&M': 3,
                  'West Virginia': 3, 'Purdue': 3, 'Clemson': 3, 'Syracuse': 3, 'UMBC': 2, 'Buffalo': 2, 'Tennessee': 2,
                  'Cincinnati': 2, 'Xavier': 2, 'Ohio State': 2, 'Houston': 2, 'North Carolina': 2, 'Alabama': 2,
                  'Marshall': 2, 'Florida': 2, 'Butler': 2, 'Seton Hall': 2, 'Auburn': 2, 'Michigan State': 2,
                  'Rhode Island': 2, 'Virginia': 1, 'Creighton': 1, 'Davidson': 1, 'Arizona': 1, 'Miami (FL)': 1,
                  'Wright State': 1, 'Texas': 1, 'Georgia State': 1, 'Texas Southern': 1, 'Missouri': 1,
                  'South Dakota State': 1, 'UNC Greensboro': 1, 'Radford': 1, 'Virginia Tech': 1, 'Murray State': 1,
                  'Wichita State': 1, 'St. Bonaventure': 1, 'Stephen F. Austin': 1, 'Arkansas': 1,
                  'Cal State Fullerton': 1, 'Penn State': 1, 'North Carolina State': 1, 'New Mexico State': 1,
                  'Charleston': 1, 'Texas Christian': 1, 'Bucknell': 1, 'Oklahoma': 1, 'Iona': 1}
year_data_2019 = {'Virginia': 7, 'Texas Tech': 6, 'Michigan State': 5, 'Auburn': 5, 'Purdue': 4, 'Kentucky': 4,
                  'Duke': 4, 'Gonzaga': 4, 'Virginia Tech': 3, 'LSU': 3, 'Florida State': 3, 'Michigan': 3, 'Oregon': 3,
                  'Tennessee': 3, 'North Carolina': 3, 'Houston': 3, 'Oklahoma': 2, 'UC Irvine': 2, 'Villanova': 2,
                  'Iowa': 2, 'Washington': 2, 'Kansas': 2, 'Ohio State': 2, 'Wofford': 2, 'UFC': 2, 'Liberty': 2,
                  'Maryland': 2, 'Minnesota': 2, 'Baylor': 2, 'Murray State': 2, 'Buffalo': 2, 'Florida': 2,
                  'Gardner-Webb': 1, 'Mississippi': 1, 'Wisconsin': 1, 'Kansas State': 1, 'Saint Mary\'s': 1,
                  'Old Dominion': 1, 'Cincinnati': 1, 'Colgate': 1, 'Iona': 1, 'Utah State': 1, 'New Mexico State': 1,
                  'Northeastern': 1, 'Iowa State': 1, 'Georgia State': 1, 'Seton Hall': 1, 'Abilene Christian': 1,
                  'North Dakota State': 1, 'VCU': 1, 'Mississippi State': 1, 'Saint Louis': 1, 'Belmont': 1,
                  'Yale': 1, 'Louisville': 1, 'Michigan State': 1, 'Bradley': 1, 'Fairleigh Dickinson': 1,
                  'Syracuse': 1, 'Marquette': 1, ' Vermont': 1, 'Arizona State': 1, 'Northern Kentucky': 1, 'Nevada': 1,
                  'Montana': 1}
year_data_2021 = {'Baylor': 7, 'Gonzaga': 6, 'Houston': 5, 'UCLA': 5, 'Arkansas': 4, 'Michigan': 4, 'Oregon State': 4,
                  'USC': 4, 'Alabama': 3, 'Creighton': 3, 'Missouri': 1, 'Florida State': 3, 'Loyola (IL)': 3,
                  'Maryland': 2, 'Oklahoma': 2, 'Oral Roberts': 3, 'Syracuse': 3, 'Colorado': 2, 'Drake': 1, 'LSU': 2,
                  'North Carolina': 1, 'Ohio State': 1, 'Ole Miss': 2, 'Rutgers': 2, 'Texas': 1, 'UC Santa Barbara': 2,
                  'Villanova': 3, 'Virginia Tech': 1, 'Wisconsin': 2, 'Abilene Christian': 2, 'Cleveland State': 1,
                  'Drexel': 1, 'Eastern Washington': 1, 'Florida': 2, 'Georgia Tech': 1, 'Grand Canyon': 1,
                  'Hartford': 1, 'Iona': 1, 'Morehead State': 1, "Mount St. Mary’s": 1, 'North Texas': 2,
                  'Texas Tech': 2, 'Ohio': 2, 'St. Bonaventure': 1, 'Texas Southern': 1, 'Connecticut': 1,
                  'Utah State': 1, 'VCU': 1, 'Western Kentucky': 1, 'Wichita State': 1, 'BYU': 1, 'Colgate': 1,
                  'Georgetown': 1, 'Georgia Tech': 1, 'Iowa': 2, 'Kansas': 2, 'Liberty': 1, 'UNC Greensboro': 1,
                  'Oklahoma State': 1, 'Oregon': 3, 'Purdue': 1, 'San Diego State': 1, 'Tennessee': 1, 'Texas Tech': 2,
                  'Virginia': 1, 'Illinois': 1, 'Rutgers': 2,  '   Winthrop': 1, 'Clemson': 1, 'Norfolk State': 1,
                  'USBC': 1, 'Texas Southern': 1, 'St. Bonaventure': 1, 'Colorado': 2}
year_data_2022 = {'Kansas': 7, 'North Carolina': 6, 'Duke': 5, 'Villanova': 5, 'Arkansas': 4, 'Saint Peter\'s': 4,
                  'Houston': 4, 'Miami (FL)': 4, 'Arizona': 3, 'Michigan': 3, 'Providence': 3, 'Iowa State': 3,
                  'Gonzaga': 3, 'Texas Tech': 3, 'UCLA': 3, 'Purdue': 3, 'Texas Christian': 2, 'Illinois': 2,
                  'Tennessee': 2, 'Ohio State': 2, 'Creighton': 2, 'Wisconsin': 2, 'Auburn': 2, 'Memphis': 2,
                  'New Mexico State': 2, 'Notre Dame': 2, 'Michigan State': 2, 'Baylor': 2, 'Saint Mary\'s': 2,
                  'Milwaukee': 2, 'Murray State': 2, 'Georgia State': 1, 'Boise State': 1, 'Connecticut': 1,
                  'Vermont': 1, 'Alabama': 1, 'Montana State': 1, 'Davidson': 1, 'Cal State Fullerton': 1,
                  'Norfolk State': 1, 'Marquette': 1, 'Indiana': 1, 'Akron': 1, 'Virginia Tech': 1, 'Yale': 1,
                  'San Francisco': 1, 'Kentucky': 1, 'Wright State': 1, 'Seton Hall': 1, 'UAB': 1, 'Chattanooga': 1,
                  'Colorado State': 1, 'Longwood': 1, 'Loyola (IL)': 1, 'Delaware': 1, 'Texas Southern': 1,
                  'San Diego State': 1, 'Iowa': 1, 'South Dakota State': 1, 'LSU': 1, 'Colgate': 1, 'USC': 1,
                  'Jacksonville State': 1, 'Richmond': 2}

# Dictionary with the key being the year and the value being year_data
all_year_data = {2017: year_data_2017, 2018: year_data_2018, 2019: year_data_2019, 2021: year_data_2021, 2022:
                 year_data_2022}


# Creates a dataframe for team data
dataframe = pd.DataFrame([TEAM_DATA_FORMAT])

# Adds data to dataframe using march madness bracket data from the years 2017-2022 excluding 2020
dataframe = add_data(all_year_data, dataframe, printing=False)

# Sets which variables in dataframe are input and which are output
# Input is set to all variables but name and rank
# Output is set to just rank
x_data = dataframe[X_FORMAT]
y_data = dataframe[Y_FORMAT]

# Creates regression model
regression_model = create_model(x_data, y_data, printing=False)
# print(predict_results('Texas A&M-CC', regression_model))

# year_data for 2023 without the values being filled in to be predicted by the regression_model
currentBracket = {'Alabama': 0, 'Texas A&M-CC': 0, 'Maryland': 0, 'West Virginia': 0, 'San Diego State': 0,
                  'Charleston': 0, 'Virginia': 0, 'Furman': 0, 'Creighton': 0, 'North Carolina Central': 0, 'Baylor': 0,
                  'UC Santa Barbara': 0, 'Missouri': 0, 'Utah State': 0, 'Arizona': 0,'Princeton': 0, 'Purdue': 0,
                  'Texas Southern': 0, 'Memphis': 0, 'Florida A&M': 0, 'Duke': 0, 'Oral Roberts': 0, 'Tennessee': 0,
                  'Louisiana': 0, 'Kentucky': 0, 'Providence': 0, 'Kansas State': 0, 'Montana State': 0,
                  'Michigan State': 0, 'USC': 0, 'Marquette': 0, 'Vermont': 0, 'Houston': 0, 'Northern Kentucky': 0,
                  'Iowa': 0, 'Auburn': 0, 'Miami (FL)': 0, 'Drake': 0, 'Indiana': 0, 'Kent State': 0, 'Iowa State': 0,
                  'Mississippi State': 0, 'Xavier': 0, 'Kennesaw State': 0, 'Texas A&M': 0, 'Penn State': 0, 'Texas': 0,
                  'Colgate': 0, 'Kansas': 0, 'Howard': 0, 'Arkansas': 0, 'Illinois': 0, 'Saint Mary\'s': 0, 'VCU': 0,
                  'Connecticut': 0, 'Iona': 0, 'Texas Christian': 0, 'Arizona State': 0, 'Gonzaga': 0,
                  'Grand Canyon': 0, 'Northwestern': 0, 'Boise State': 0, 'UCLA': 0, 'UNC Asheville': 0}

# Smaller test bracket to run faster
# currentBracket = {'Alabama': 0, 'Texas A&M-CC': 0}

# ======= Run Games =======

# Give each team their prediction score
fill_teams_score(currentBracket, regression_model, printing=True)

print("Round 1 Results:")
round_of_games(currentBracket)

print("Round 2 Results:")
round_of_games(currentBracket)

print("Elite 8:")
round_of_games(currentBracket)

print("Final 4:")
round_of_games(currentBracket)

print("Championship:")
round_of_games(currentBracket)

print("Winner:")
round_of_games(currentBracket)
