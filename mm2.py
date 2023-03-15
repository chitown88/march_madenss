# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 08:11:13 2023

@author: Jason
"""

import requests
import pandas as pd
import time
import bs4
import numpy as np
from tqdm import tqdm
from school_names_map import REPLACE_SCHOOL_NAME


def run(current_year, team_stats_2019, school_links):

    seed_list = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
    
    seasons_list = [current_year]
    tourny_results_2019 = pd.DataFrame()
    events = []
    for season in seasons_list:
        for day in range(1,30):
            date = f'{season}03{day:02}'
            print(date)
            url = f'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=100&limit=200&dates={date}'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
            
            #time.sleep(5)
            #response = requests.get(url, headers=headers)
            jsonData = requests.get(url, headers=headers).json()
            if jsonData['events']:
                events += jsonData['events']
        
    rows = []
    for event in events:
        competition = event['competitions'][0]
        
        competition_dict = {}
        competition_dict['competitors'] = competition['competitors']
        _x, region, tourn_round = [x.strip().lower() for x in competition['notes'][0]['headline'].split('-')]
        
        
        
        
        team_a = competition_dict['competitors'][0]
        team_b = competition_dict['competitors'][1]

        team_a_name = team_a['team']['location']
        team_a_seed = team_a['curatedRank']['current']
        team_b_name = team_b['team']['location']
        team_b_seed = team_b['curatedRank']['current']
        
        row = {
            'region': region,
            'round': tourn_round,
            'School_Seed': team_a_seed,
            'School': team_a_name,
            'School_Opp_Seed': team_b_seed,
            'School_Opp': team_b_name,
            'Season': season}
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.loc[(df['round'] == '1st round') | (df['round'] == 'first four')]
    df['School_Opp_Seed'] = df['School_Opp_Seed'].astype(str).replace('99', None)
    df['School_Opp'] = df['School_Opp'].replace('TBD', None)
    
    # First, create a new column that combines the School and School_Opp values for rows with "first four" round
    df.loc[df['round'] == 'first four', 'combined'] = df['School'] + '/' + df['School_Opp']
     
    # Find the rows with combined values
    combined_rows = df[df['combined'].notna()]
    
    df = pd.merge(df, combined_rows, how='left', left_on = ['region'], right_on=['region'], suffixes=('', '_x'))
    df['School_Opp'] = df['School_Opp'].fillna(df['combined_x'])
    df['School_Opp_Seed'] = df['School_Opp_Seed'].fillna(df['School_Opp_Seed_x'])
    df = df.drop(df.filter(regex='_x$').columns, axis=1)
    df = df.drop('combined', axis=1)
    
    
    df = df[df['round'] == '1st round']
    df = df.drop('round', axis=1)
    df = df.drop('region', axis=1)

    tourny_results_2019 = df.copy()
    
    school_name_map = REPLACE_SCHOOL_NAME
    for col in ['School', 'School_Opp']:
        tourny_results_2019[col] = tourny_results_2019[col].map(school_name_map).fillna(tourny_results_2019[col])

    playInTeamList = list(df[df['School_Opp'].str.contains('/')]['School_Opp'])
    #playInTeamList = [team for teams in playInTeamList for team in teams.split('/')]


    # AVERAGE OUT PLAYIN TEAMS STATS
    playIn_team_stats_2019 = pd.DataFrame()
    
    processPlayInTeams = True
    if processPlayInTeams == True:
        for playInTeams_temp in playInTeamList:   #TODO
            teamList = playInTeams_temp.split('/')
            temp_df = team_stats_2019[team_stats_2019['School'].isin(teamList)]
            for col in temp_df.columns:
                try:
                    temp_df[col] = temp_df[col].astype(float)
                except:
                    pass
            averageDf = pd.DataFrame(temp_df.mean()).T
            averageDf['School'] = playInTeams_temp
            averageDf = averageDf[team_stats_2019.columns]
            
            #playIn_team_stats_2019 = playIn_team_stats_2019.append(averageDf)
            playIn_team_stats_2019 = pd.concat([playIn_team_stats_2019, averageDf])
            
        
        #team_stats_2019 = team_stats_2019.append(playIn_team_stats_2019).reset_index(drop=True)
        team_stats_2019 = pd.concat([team_stats_2019, playIn_team_stats_2019])
        team_stats_2019 = team_stats_2019.reset_index(drop=True)
    
    for col in team_stats_2019.columns:
        try:
            team_stats_2019[col] = team_stats_2019[col].astype(float)
        except:
            pass
    
    to_predict_df = tourny_results_2019.merge(team_stats_2019, how='left', left_on = ['School', 'Season'], right_on = ['School','Season'])
    to_predict_df = to_predict_df.merge(team_stats_2019, how='left', left_on = ['School_Opp', 'Season'], right_on = ['School','Season'], suffixes = ("","_Opp"))
    
    to_predict_df.rename(columns={'Points_Opp': 'Points_Allowed'}, inplace=True)
    to_predict_df = to_predict_df.loc[:, ~to_predict_df.columns.duplicated()]
    to_predict_df['Outcome'] = None
    
    to_predict_df['School_Seed'] = to_predict_df['School_Seed'].astype(int)
    to_predict_df['School_Opp_Seed'] = to_predict_df['School_Opp_Seed'].astype(int)
    
    return to_predict_df, team_stats_2019



