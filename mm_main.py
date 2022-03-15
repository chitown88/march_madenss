# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:54:39 2022

@author: Jason
"""

import datetime
import mm1
import mm2
import mm3


def run():
    current_year = datetime.datetime.now().year
    current_year_input = int(input(f'Enter year (2000-{current_year}) to predict March Madness.\n-> '))
    
    team_stats, final_results, school_links = mm1.run(current_year_input)
    to_predict_df, team_stats = mm2.run(current_year_input, team_stats, school_links)
    all_pick_results = mm3.run(current_year_input, to_predict_df, final_results, team_stats)
    
    return all_pick_results, current_year_input

if __name__ == "__main__":
   all_pick_results, current_year_input = run()
   print(all_pick_results)
   all_pick_results.to_csv(f'{current_year_input}_results.csv', index=False)


