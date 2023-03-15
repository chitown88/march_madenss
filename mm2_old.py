# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:10:23 2019

@author: Jason
"""

import requests
import pandas as pd
import time
import bs4
import numpy as np
from tqdm import tqdm


def run(current_year, team_stats_2019, school_links):

    seed_list = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
    
    seasons_list = [current_year]
    tourny_results_2019 = pd.DataFrame()
    for season in seasons_list:
        url = 'https://www.sports-reference.com/cbb/postseason/%s-ncaa.html' %(season)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
        
        #time.sleep(5)
        response = requests.get(url, headers=headers)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        
        four_regions = soup.find('div', {'data-controls':'#brackets'}).find_all('a')
        regions = [ a.text.replace(' ','').replace('.','').lower() for a in four_regions ]
        #regions[-1] = 'national'
        regions = regions[:4]
        
        playInTeamList = []
        for region in regions:
            ##################
            #region = regions[0]
            bracket = soup.find_all('div', {'id':region})[0]
            rounds = bracket.find_all('div',{'class':'round'})
            #for r in rounds:
                #############
            
            playIn = bracket.find('p')
            playInSeed = playIn.find_all('strong')[-1].text
            playInTeamNames_alpha = playIn.find_all('a', href=True)
            playInTeamNames = []
            for x in playInTeamNames_alpha:
                if 'boxscores' not in x['href']:
                    playInTeamNames.append(school_links[x['href']])
                
            playInTeamNames = '|'.join(playInTeamNames)    
            playInTeamList.append(playInTeamNames)   
               
            r = rounds[0]
            games = r.find_all('div')
            comments = r.find_all(string=lambda text:isinstance(text,bs4.Comment))
            
            processPlayInTeams = False
            
            idx = 0
            for comment in comments:
                if 'game' in comment:
                    game = comment.parent
                    
                    teams = game.find_all('div')
                    
                    try:
                        team_a = teams[0]
                        team_a_name = school_links[team_a.find('a')['href']]
                        team_a_seed = team_a.find('span').text
                        if team_a_seed == '':
                            team_a_seed = str(seed_list[idx*2])  
                    except:
                        #team_a_name = 'N/A'
                        team_a_name = playInTeamNames
                        #team_a_seed = str(seed_list[idx*2])
                        team_a_seed = playInSeed
                        processPlayInTeams = True

                    
                    try:
                        team_b = teams[1]
                        team_b_name = school_links[team_b.find('a')['href']]
                        team_b_seed = team_b.find('span').text
                        if team_b_seed == '':
                            team_b_seed = str(seed_list[(idx*2)+1])                  
                    except:
                        #team_b_name = 'N/A'
                        team_b_name = playInTeamNames
                        #team_b_seed = str(seed_list[(idx*2)+1]) 
                        team_b_seed = playInSeed
                        processPlayInTeams = True
                    
                    
                    temp_df = pd.DataFrame([[team_a_seed, team_a_name, team_b_seed, team_b_name, season]], columns=['School_Seed','School','School_Opp_Seed','School_Opp', 'Season'])
                    #tourny_results_2019 = tourny_results_2019.append(temp_df)
                    tourny_results_2019 = pd.concat([tourny_results_2019, temp_df])
                    print ('%s: %s: %-20s       %s: %s' %(season, team_a_seed, team_a_name, team_b_seed, team_b_name))
                    idx+=1
    
    
    tourny_results_2019 = tourny_results_2019.reset_index(drop=True)  
    
    # AVERAGE OUT PLAYIN TEAMS STATS
    playIn_team_stats_2019 = pd.DataFrame()
    
    if processPlayInTeams == True:
        for playInTeams_temp in playInTeamList:
            teamList = playInTeams_temp.split('|')
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
    
    
    to_predict_df = tourny_results_2019.merge(team_stats_2019, how='left', left_on = ['School', 'Season'], right_on = ['School','Season'])
    to_predict_df = to_predict_df.merge(team_stats_2019, how='left', left_on = ['School_Opp', 'Season'], right_on = ['School','Season'], suffixes = ("","_Opp"))
    
    to_predict_df.rename(columns={'Points_Opp': 'Points_Allowed'}, inplace=True)
    to_predict_df = to_predict_df.loc[:, ~to_predict_df.columns.duplicated()]
    to_predict_df['Outcome'] = None
    
    to_predict_df['School_Seed'] = to_predict_df['School_Seed'].astype(int)
    to_predict_df['School_Opp_Seed'] = to_predict_df['School_Opp_Seed'].astype(int)
    
    return to_predict_df, team_stats_2019


