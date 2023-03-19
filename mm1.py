# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:17:43 2019

@author: Jason
"""

import requests
import pandas as pd
import time
import bs4
import numpy as np
from tqdm import tqdm
import datetime



def run(current_year):
    school_links = {}
    
    results = pd.DataFrame()
    seasons_list = list(range(1993, current_year+1))
    #seasons_list = list(range(2022, current_year+1))
    #seasons_list = list(range(1993, 1995))
    #seasons_list = [ x for x in seasons_list if x != 2020]
    
    cols = ['drop','School','G','W','L','W-L%','SRS','SOS','Conf_W','Conf_L','Home_W','Home_L','Away_W','Away_L','Points_Tm','Points_Opp','drop','MP','FG','FGA','FG%','3P','3PA','3P%','FT','FTA','FT%','ORB','TRB','AST','STL','BLK','TOV','PF', 'Season']   
    cols = [x for x in cols if x != 'drop']
    
    for season in seasons_list:
        url = 'https://www.sports-reference.com/cbb/seasons/%s-school-stats.html' %(season)
        
        #tables = pd.read_html(url)
        response = requests.get(url)
        table = pd.read_html(response.text, header=1)[0]
        table = table[table['Rk'].ne('Rk')]
        #table.columns = cols
        table['Season'] = season
        
        #results = results.append(table)
        results = pd.concat([results, table])
        time.sleep(5)
        print ('Downloaded %s Season' %season)
        
        
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        links = table.find_all('a')
        for link in links:
            teamName = link.text
            teamName = teamName.replace('NCAA','').strip()
            school_links[link['href']] = link.text
            
        
    
    
    results = results.reset_index(drop=True) 
    results = results.rename(columns={'W.1':'Conf_W', 'W.2':'Home_W', 'W.3':'Away_W', 'L.1':'Conf_L', 'L.2':'Home_L', 'L.3':'Away_L','Tm.':'Points_Tm','Opp.':'Points_Opp'})
    results = results[cols]
    results = results[results.G != 'Overall']
    results = results[results.G != 'School']
    results['School'] = results.School.str.replace('NCAA','')
    results['School'] = results['School'].map(lambda x: x.strip())
    for col in ['G', 'W', 'L', 'W-L%', 'SRS', 'SOS', 'Conf_W', 'Conf_L', 'Home_W', 'Home_L', 'Away_W', 'Away_L', 'Points_Tm', 'Points_Opp', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']:
        results[col] = results[col].fillna(results[col].astype(float).mean())
    
    
    
    #seasons_list = [2018]
    
    tourny_results = pd.DataFrame()
    seasons_list = [x for x in seasons_list if x not in [2020,current_year]]
    for season in seasons_list:
        url = 'https://www.sports-reference.com/cbb/postseason/%s-ncaa.html' %(season)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'}
        
        time.sleep(5)
        response = requests.get(url, headers=headers)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        
        four_regions = soup.find('div', {'data-controls':'#brackets'}).find_all('a')
        regions = [ a.text.replace(' ','').replace('.','').lower() for a in four_regions ]
        regions[-1] = 'national'
        
        
        for region in regions:
            ##################
            #region = regions[0]
            bracket = soup.find_all('div', {'id':region})[0]
            rounds = bracket.find_all('div',{'class':'round'})
            for r in rounds:
                #############
                #r = rounds[0]
                games = r.find_all('div')
                comments = r.find_all(string=lambda text:isinstance(text,bs4.Comment))
                
                for comment in comments:
                    if 'game' in comment:
                        game = comment.parent
                        
                        teams = game.find_all('div')
                        for team in teams:
                            
                            if team.attrs == {'class': ['winner']}:
                                w_team_seed = team.find('span').text
                                w_team = school_links[team.find('a')['href']]
                            if team.attrs == {}:
                                l_team_seed = team.find('span').text
                                l_team = school_links[team.find('a')['href']]                          
                        if w_team == l_team:
                            continue
                        temp_df = pd.DataFrame([[w_team_seed, w_team, l_team_seed, l_team, season]], columns=['Winning_Team_Seed','Winning_Team','Losing_Team_Seed','Losing_Team', 'Season'])
                        #tourny_results = tourny_results.append(temp_df)
                        tourny_results = pd.concat([tourny_results, temp_df])
                        print ('%s: %s: %-20s       %s: %s' %(season, w_team_seed, w_team, l_team_seed, l_team))
    
    
    tourny_results = tourny_results.reset_index(drop=True)   

    final_results_a = tourny_results.copy() 
    final_results_b = tourny_results.copy() 
    
    final_results_a = final_results_a.merge(results, how='left', left_on = ['Winning_Team','Season'], right_on = ['School','Season'], suffixes = ("","_Opp"))
    final_results_a = final_results_a.merge(results, how='left', left_on = ['Losing_Team','Season'], right_on = ['School','Season'], suffixes = ("","_Opp"))
    final_results_a = final_results_a[final_results_a['School'].notnull()]
    
    final_results_b = final_results_b.merge(results, how='left', left_on = ['Losing_Team','Season'], right_on = ['School','Season'], suffixes = ("","_Opp"))
    final_results_b = final_results_b.merge(results, how='left', left_on = ['Winning_Team','Season'], right_on = ['School','Season'], suffixes = ("","_Opp"))
    final_results_b = final_results_b[final_results_b['School'].notnull()]
    
    #final_results = final_results_a.append(final_results_b).reset_index(drop=True)
    final_results = pd.concat([final_results_a, final_results_b])
    final_results = final_results.reset_index(drop=True)
    
    final_results['Outcome'] = np.where((final_results['School'] == final_results['Winning_Team']), 'Win', 'Loss')
    
    
    for i, row in tqdm(final_results.iterrows()):
        school = row['School']
        if row['Winning_Team'] == school:
            school_seed = row['Winning_Team_Seed']
        else:
            school_seed = row['Losing_Team_Seed']
        
        school_opp = row['School_Opp']
        if row['Winning_Team'] == school_opp:
            school_opp_seed = row['Winning_Team_Seed']
        else:
            school_opp_seed = row['Losing_Team_Seed']
        
        final_results.loc[i, 'School_Seed'] = school_seed
        final_results.loc[i, 'School_Opp_Seed'] = school_opp_seed
    
    
    
    final_results = final_results.drop(['Winning_Team_Seed','Winning_Team','Losing_Team_Seed','Losing_Team'], 1)
    final_results.rename(columns={'Points_Opp': 'Points_Allowed'}, inplace=True)
    
    
    final_results.to_clipboard()
    team_stats_2019 = results[results['Season'] == current_year]
    
    return team_stats_2019, final_results, school_links



