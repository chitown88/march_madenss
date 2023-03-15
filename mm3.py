# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 08:10:55 2019

@author: Jason
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pd.set_option('mode.chained_assignment', None)

# UPDATE NAN IN TO PREDICT
#to_predict_df_copy = to_predict_df.copy()


def run(current_year, to_predict_df, final_results, team_stats_2019):

    #to_predict_df = to_predict_df_copy.copy()
    
    seedings_2019_a = to_predict_df[['School_Seed','School']]
    seedings_2019_b = to_predict_df[['School_Opp_Seed','School_Opp']]
    seedings_2019_b.columns = ['School_Seed','School']
    
    seedings_2019 = seedings_2019_a
    #seedings_2019 = seedings_2019.append(seedings_2019_b)
    seedings_2019 = pd.concat([seedings_2019, seedings_2019_b])
    
    
    seedings_dict = dict(zip(seedings_2019['School'], seedings_2019['School_Seed']))
    
    seeds = list(range(1,17))
    seeds.sort(reverse=True)
    
    test_data = final_results.copy()
    null_cols = test_data.columns[test_data.isna().any()].tolist()
    null_cols_alpha = to_predict_df.columns[to_predict_df.isna().any()].tolist()
    null_cols += null_cols_alpha
    null_cols.remove('Outcome')
    
    for col in null_cols:
        if 'Opp' in col:
            test_data[col] = test_data[col].astype(float)
            
            for seed in seeds:
                temp_df = test_data[test_data['School_Opp_Seed'].astype(int) == seed]
                temp_df = temp_df.drop_duplicates(subset=['School_Opp_Seed','School_Opp','Season'])
                temp_mean = temp_df[temp_df['School_Opp_Seed'].astype(int) == seed][col].mean()
                if '%' not in col:
                    temp_mean = int(round(temp_mean,0))
                    mask = test_data['School_Opp_Seed'].astype(int) == seed
                    test_data.loc[mask, col] = test_data.loc[mask, col].fillna(temp_mean)
                    mask = to_predict_df['School_Opp_Seed'].astype(int) == seed
                    to_predict_df.loc[mask, col] = to_predict_df.loc[mask, col].fillna(temp_mean)
                else:
                    mask = test_data['School_Opp_Seed'].astype(int) == seed
                    test_data.loc[mask, col] = test_data.loc[mask, col].fillna(temp_mean)
                    mask = to_predict_df['School_Opp_Seed'].astype(int) == seed
                    to_predict_df.loc[mask, col] = to_predict_df.loc[mask, col].fillna(temp_mean)                
    
        elif 'Opp' not in col:
            test_data[col] = test_data[col].astype(float)
            
            for seed in seeds:
                temp_df = test_data[test_data['School_Seed'].astype(int) == seed]
                temp_df = temp_df.drop_duplicates(subset=['School_Seed','School','Season'])            
                temp_mean = temp_df[temp_df['School_Seed'].astype(int) == seed][col].mean()
                if '%' not in col:
                    temp_mean = int(round(temp_mean,0))
                    mask = test_data['School_Seed'].astype(int) == seed
                    test_data.loc[mask, col] = test_data.loc[mask, col].fillna(temp_mean)
                    mask = to_predict_df['School_Seed'].astype(int) == seed
                    to_predict_df.loc[mask, col] = to_predict_df.loc[mask, col].fillna(temp_mean)
                else:
                    mask = test_data['School_Seed'].astype(int) == seed
                    test_data.loc[mask, col] = test_data.loc[mask, col].fillna(temp_mean)
                    mask = to_predict_df['School_Seed'].astype(int) == seed
                    to_predict_df.loc[mask, col] = to_predict_df.loc[mask, col].fillna(temp_mean)
    test_data = test_data.dropna(axis=0, how="any")
    test_data = test_data.apply(pd.to_numeric,errors='ignore')
    to_predict_df = to_predict_df.apply(pd.to_numeric,errors='ignore')
    #test_data = test_data.dropna(axis=1, how="any")
    
    norm_cols = list(test_data.select_dtypes('int64').columns)
    norm_cols_alpha = list(to_predict_df.select_dtypes('int64').columns)
    norm_cols = norm_cols + norm_cols_alpha
    norm_cols = list(set(norm_cols))
    for del_col in ['Season','G','G_Opp','School_Seed','School_Opp_Seed']:
        norm_cols.remove(del_col)
    team_norm_cols = [x for x in norm_cols if 'Opp' not in x]
    opp_norm_cols = [x for x in norm_cols if 'Opp' in x]
        
    for col in team_norm_cols:
        test_data[col] = test_data[col] / test_data['G']
        to_predict_df[col] = to_predict_df[col] / to_predict_df['G']
        
    for col in opp_norm_cols:
        test_data[col] = test_data[col] / test_data['G_Opp']
        to_predict_df[col] = to_predict_df[col] / to_predict_df['G_Opp']
        
        
    
    drop_cols = ['G','G_Opp', 'W-L%_Opp','W-L%','School_Seed','School_Opp_Seed','School','School_Opp','Outcome']
    #xtra_cols = [ x for x in to_predict_df.columns if 'Home' in x ]
    #drop_cols += xtra_cols
    # creating input features and target variables
    #X = test_data.drop(['School','School_Opp','Outcome'], axis=1)
    X = test_data.drop(drop_cols, axis=1)
    y = test_data['Outcome']
    
    #standardizing the input feature
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_y, test_size=0.30)
    
    
    
    dim = X_train.shape[1]
    #Results: 84.26% (2.50%)
    
    # baseline model
    def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
        model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model
    
    epochs = 10
    batch_size = 64
    
    # evaluate model with standardized dataset
    estimator = KerasClassifier(build_fn=create_baseline, epochs=epochs, batch_size=batch_size, verbose=2)
    kfold = StratifiedKFold(n_splits=2, shuffle=True)
    
    results = cross_val_score(estimator, X, encoded_y, cv=kfold)
    print("Results: %.2f%% (%.2f)" % (results.mean()*100, results.std()*100))
    X_pred = to_predict_df.drop(drop_cols, axis=1)
    X_pred = sc.transform(X_pred)
    
    
    # create model
    model = Sequential()
    model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(round(dim/2), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, verbose=2)
    
    
    all_pick_results = pd.DataFrame()
    rnd = 0
    ##########################################################################
    y_results = model.predict(X_pred)
    y_results = pd.DataFrame(y_results, columns = ['Outcome'])
    
    y_results['Prob'] = y_results['Outcome']
    
    to_predict_temp_df = to_predict_df.drop(['Outcome'], axis=1)
    picks = pd.merge(to_predict_temp_df, y_results, left_index=True, right_index=True)
    picks['Outcome'] = np.where(picks['Outcome']>=.5, 'Win', 'Loss')
    
    pick_results = picks[['School_Seed','School','School_Opp_Seed','School_Opp', 'Outcome', 'Prob']]
    pick_results['Winner'] = np.where(picks['Outcome'] == 'Win', picks['School'], picks['School_Opp'])
    pick_results['Upset'] = np.where((picks['Outcome'] == 'Loss') & (picks['School_Opp_Seed'] > picks['School_Seed']), 'Upset', 'No')
    
    rnd+=1
    pick_results['Round'] = rnd
    #all_pick_results = all_pick_results.append(pick_results,sort=False).reset_index(drop=True)
    all_pick_results = pd.concat([all_pick_results, pick_results])
    all_pick_results = all_pick_results.reset_index(drop=True)
    
    
    print ('\nFirst Round Winners')
    first_round_winners = list(pick_results['Winner'])
    for team in first_round_winners:
        print ('%s: %s' %(seedings_dict[team], team))
    
    
    ###############################################################################
    
    second_round_df = pd.DataFrame(first_round_winners)
    second_round_df = pd.DataFrame(second_round_df.values.reshape(-1, 2), columns = ['School', 'School_Opp'])  
    second_round_df['Season'] = current_year
    
    
    second_round_df = second_round_df.merge(team_stats_2019, how='left', left_on = ['School', 'Season'], right_on = ['School', 'Season'])
    second_round_df = second_round_df.merge(team_stats_2019, how='left', left_on = ['School_Opp', 'Season'], right_on = ['School', 'Season'], suffixes = ("","_Opp"))
    second_round_df.rename(columns={'Points_Opp': 'Points_Allowed'}, inplace=True)
    second_round_df = second_round_df.loc[:, ~second_round_df.columns.duplicated()]

    for col in team_norm_cols:
        second_round_df[col] = second_round_df[col].astype(float) / second_round_df['G'].astype(int)
    
        
    for col in opp_norm_cols:
        second_round_df[col] = second_round_df[col].astype(float) / second_round_df['G_Opp'].astype(int)
    
    for x in ['School_Seed', 'School_Opp_Seed', 'Outcome']:
        drop_cols.remove(x)
    X_pred = second_round_df.drop(drop_cols, axis=1)
    X_pred = sc.transform(X_pred)
    
    y_results = model.predict(X_pred)
    y_results = pd.DataFrame(y_results, columns = ['Outcome'])
    y_results['Prob'] = y_results['Outcome']
    
    picks = pd.merge(second_round_df, y_results, left_index=True, right_index=True)
    picks['Outcome'] = np.where(picks['Outcome']>=.5, 'Win', 'Loss')
    
    pick_results = picks[['School','School_Opp', 'Outcome', 'Prob']]
    pick_results['Winner'] = np.where(picks['Outcome'] == 'Win', picks['School'], picks['School_Opp'])
    pick_results['Loser'] = np.where(picks['Outcome'] == 'Win', picks['School_Opp'], picks['School'])
    pick_results['WinnerSeed'] = pick_results['Winner'].map(seedings_dict)
    pick_results['LoserSeed'] = pick_results['Loser'].map(seedings_dict)
    pick_results['Upset'] = np.where(pick_results['WinnerSeed'] > pick_results['LoserSeed'], 'Upset', 'No')
    drop_cols_alpha = ['WinnerSeed', 'LoserSeed', 'Loser']

    pick_results = pick_results.drop(drop_cols_alpha, axis=1)    
    
    rnd+=1
    pick_results['Round'] = rnd
    #all_pick_results = all_pick_results.append(pick_results,sort=False).reset_index(drop=True)
    all_pick_results = pd.concat([all_pick_results, pick_results])
    all_pick_results = all_pick_results.reset_index(drop=True)
    all_pick_results['School_Seed'] = all_pick_results['School'].map(seedings_dict)
    
    
    print ('\nSecond Round Winners')
    second_round_winners = list(pick_results['Winner'])
    for team in second_round_winners:
        print ('%s: %s' %(seedings_dict[team], team))
    
    ###############################################################################
    
    third_round_df = pd.DataFrame(second_round_winners)
    third_round_df = pd.DataFrame(third_round_df.values.reshape(-1, 2), columns = ['School', 'School_Opp'])  
    third_round_df['Season'] = current_year
    
    
    third_round_df = third_round_df.merge(team_stats_2019, how='left', left_on = ['School', 'Season'], right_on = ['School', 'Season'])
    third_round_df = third_round_df.merge(team_stats_2019, how='left', left_on = ['School_Opp', 'Season'], right_on = ['School', 'Season'], suffixes = ("","_Opp"))
    third_round_df.rename(columns={'Points_Opp': 'Points_Allowed'}, inplace=True)
    third_round_df = third_round_df.loc[:, ~third_round_df.columns.duplicated()]
    
    for col in team_norm_cols:
        third_round_df[col] = third_round_df[col].astype(float) / third_round_df['G'].astype(int)
    
        
    for col in opp_norm_cols:
        third_round_df[col] = third_round_df[col].astype(float) / third_round_df['G_Opp'].astype(int)
    
    X_pred = third_round_df.drop(drop_cols, axis=1)
    X_pred = sc.transform(X_pred)
    
    y_results = model.predict(X_pred)
    y_results = pd.DataFrame(y_results, columns = ['Outcome'])
    y_results['Prob'] = y_results['Outcome']
    
    picks = pd.merge(third_round_df, y_results, left_index=True, right_index=True)
    picks['Outcome'] = np.where(picks['Outcome']>=.5, 'Win', 'Loss')
    
    pick_results = picks[['School','School_Opp', 'Outcome', 'Prob']]
    pick_results['Winner'] = np.where(picks['Outcome'] == 'Win', picks['School'], picks['School_Opp'])
    pick_results['Loser'] = np.where(picks['Outcome'] == 'Win', picks['School_Opp'], picks['School'])
    pick_results['WinnerSeed'] = pick_results['Winner'].map(seedings_dict)
    pick_results['LoserSeed'] = pick_results['Loser'].map(seedings_dict)
    pick_results['Upset'] = np.where(pick_results['WinnerSeed'] > pick_results['LoserSeed'], 'Upset', 'No')
    drop_cols_alpha = ['WinnerSeed', 'LoserSeed', 'Loser']

    pick_results = pick_results.drop(drop_cols_alpha, axis=1)    
    
    rnd+=1
    pick_results['Round'] = rnd
    #all_pick_results = all_pick_results.append(pick_results,sort=False).reset_index(drop=True)
    all_pick_results = pd.concat([all_pick_results, pick_results])
    all_pick_results = all_pick_results.reset_index(drop=True)
    
    print ('\nSweet Sixteen Winners')
    third_round_winners = list(pick_results['Winner'])
    for team in third_round_winners:
        print ('%s: %s' %(seedings_dict[team], team))
    
    ###############################################################################
    
    fourth_round_df = pd.DataFrame(third_round_winners)
    fourth_round_df = pd.DataFrame(fourth_round_df.values.reshape(-1, 2), columns = ['School', 'School_Opp'])  
    fourth_round_df['Season'] = current_year
    
    
    fourth_round_df = fourth_round_df.merge(team_stats_2019, how='left', left_on = ['School', 'Season'], right_on = ['School', 'Season'])
    fourth_round_df = fourth_round_df.merge(team_stats_2019, how='left', left_on = ['School_Opp', 'Season'], right_on = ['School', 'Season'], suffixes = ("","_Opp"))
    fourth_round_df.rename(columns={'Points_Opp': 'Points_Allowed'}, inplace=True)
    fourth_round_df = fourth_round_df.loc[:, ~fourth_round_df.columns.duplicated()]
    
    for col in team_norm_cols:
        fourth_round_df[col] = fourth_round_df[col].astype(float) / fourth_round_df['G'].astype(int)
    
        
    for col in opp_norm_cols:
        fourth_round_df[col] = fourth_round_df[col].astype(float) / fourth_round_df['G_Opp'].astype(int)
    
    
    X_pred = fourth_round_df.drop(drop_cols, axis=1)
    X_pred = sc.transform(X_pred)
    
    y_results = model.predict(X_pred)
    y_results = pd.DataFrame(y_results, columns = ['Outcome'])
    y_results['Prob'] = y_results['Outcome']
    
    picks = pd.merge(fourth_round_df, y_results, left_index=True, right_index=True)
    picks['Outcome'] = np.where(picks['Outcome']>=.5, 'Win', 'Loss')
    
    pick_results = picks[['School','School_Opp', 'Outcome', 'Prob']]
    pick_results['Winner'] = np.where(picks['Outcome'] == 'Win', picks['School'], picks['School_Opp'])
    pick_results['Loser'] = np.where(picks['Outcome'] == 'Win', picks['School_Opp'], picks['School'])
    pick_results['WinnerSeed'] = pick_results['Winner'].map(seedings_dict)
    pick_results['LoserSeed'] = pick_results['Loser'].map(seedings_dict)
    pick_results['Upset'] = np.where(pick_results['WinnerSeed'] > pick_results['LoserSeed'], 'Upset', 'No')
    drop_cols_alpha = ['WinnerSeed', 'LoserSeed', 'Loser']

    pick_results = pick_results.drop(drop_cols_alpha, axis=1)    
    
    rnd+=1
    pick_results['Round'] = rnd
    #all_pick_results = all_pick_results.append(pick_results,sort=False).reset_index(drop=True)
    all_pick_results = pd.concat([all_pick_results, pick_results])
    all_pick_results = all_pick_results.reset_index(drop=True)
    
    print ('\nElite Eight Winners')
    fourth_round_winners = list(pick_results['Winner'])
    for team in fourth_round_winners:
        print ('%s: %s' %(seedings_dict[team], team))
    
    ###############################################################################
    
    fifth_round_df = pd.DataFrame(fourth_round_winners)
    
    # Realign correct matchup
    fifth_round_df = fifth_round_df.reindex([3,0,1,2]).reset_index(drop=True)
    
    fifth_round_df = pd.DataFrame(fifth_round_df.values.reshape(-1, 2), columns = ['School', 'School_Opp'])  
    fifth_round_df['Season'] = current_year
    
    
    fifth_round_df = fifth_round_df.merge(team_stats_2019, how='left', left_on = ['School', 'Season'], right_on = ['School', 'Season'])
    fifth_round_df = fifth_round_df.merge(team_stats_2019, how='left', left_on = ['School_Opp', 'Season'], right_on = ['School', 'Season'], suffixes = ("","_Opp"))
    fifth_round_df.rename(columns={'Points_Opp': 'Points_Allowed'}, inplace=True)
    fifth_round_df = fifth_round_df.loc[:, ~fifth_round_df.columns.duplicated()]
    
    for col in team_norm_cols:
        fifth_round_df[col] = fifth_round_df[col].astype(float) / fifth_round_df['G'].astype(int)
    
        
    for col in opp_norm_cols:
        fifth_round_df[col] = fifth_round_df[col].astype(float) / fifth_round_df['G_Opp'].astype(int)
    
    X_pred = fifth_round_df.drop(drop_cols, axis=1)
    X_pred = sc.transform(X_pred)
    
    y_results = model.predict(X_pred)
    y_results = pd.DataFrame(y_results, columns = ['Outcome'])
    y_results['Prob'] = y_results['Outcome']
    
    picks = pd.merge(fifth_round_df, y_results, left_index=True, right_index=True)
    picks['Outcome'] = np.where(picks['Outcome']>=.5, 'Win', 'Loss')
    
    pick_results = picks[['School','School_Opp', 'Outcome', 'Prob']]
    pick_results['Winner'] = np.where(picks['Outcome'] == 'Win', picks['School'], picks['School_Opp'])
    pick_results['Loser'] = np.where(picks['Outcome'] == 'Win', picks['School_Opp'], picks['School'])
    pick_results['WinnerSeed'] = pick_results['Winner'].map(seedings_dict)
    pick_results['LoserSeed'] = pick_results['Loser'].map(seedings_dict)
    pick_results['Upset'] = np.where(pick_results['WinnerSeed'] > pick_results['LoserSeed'], 'Upset', 'No')
    drop_cols_alpha = ['WinnerSeed', 'LoserSeed', 'Loser']

    pick_results = pick_results.drop(drop_cols_alpha, axis=1)      
    
    rnd+=1
    pick_results['Round'] = rnd
    #all_pick_results = all_pick_results.append(pick_results,sort=False).reset_index(drop=True)
    all_pick_results = pd.concat([all_pick_results, pick_results])
    all_pick_results = all_pick_results.reset_index(drop=True)
    
    print ('\nFinal Four Winners')
    fifth_round_winners = list(pick_results['Winner'])
    for team in fifth_round_winners:
        print ('%s: %s' %(seedings_dict[team], team))
    
    
    ###############################################################################
    
    sixth_round_df = pd.DataFrame(fifth_round_winners)
    sixth_round_df = pd.DataFrame(sixth_round_df.values.reshape(-1, 2), columns = ['School', 'School_Opp'])  
    sixth_round_df['Season'] = current_year
    
    
    sixth_round_df = sixth_round_df.merge(team_stats_2019, how='left', left_on = ['School', 'Season'], right_on = ['School', 'Season'])
    sixth_round_df = sixth_round_df.merge(team_stats_2019, how='left', left_on = ['School_Opp', 'Season'], right_on = ['School', 'Season'], suffixes = ("","_Opp"))
    sixth_round_df.rename(columns={'Points_Opp': 'Points_Allowed'}, inplace=True)
    sixth_round_df = sixth_round_df.loc[:, ~sixth_round_df.columns.duplicated()]
    
    for col in team_norm_cols:
        sixth_round_df[col] = sixth_round_df[col].astype(float) / sixth_round_df['G'].astype(int)
    
        
    for col in opp_norm_cols:
        sixth_round_df[col] = sixth_round_df[col].astype(float) / sixth_round_df['G_Opp'].astype(int)
    
    X_pred = sixth_round_df.drop(drop_cols, axis=1)
    X_pred = sc.transform(X_pred)
    
    y_results = model.predict(X_pred)
    y_results = pd.DataFrame(y_results, columns = ['Outcome'])
    y_results['Prob'] = y_results['Outcome']
    
    picks = pd.merge(sixth_round_df, y_results, left_index=True, right_index=True)
    picks['Outcome'] = np.where(picks['Outcome']>=.5, 'Win', 'Loss')
    
    pick_results = picks[['School','School_Opp', 'Outcome', 'Prob']]
    pick_results['Winner'] = np.where(picks['Outcome'] == 'Win', picks['School'], picks['School_Opp'])
    pick_results['Loser'] = np.where(picks['Outcome'] == 'Win', picks['School_Opp'], picks['School'])
    pick_results['WinnerSeed'] = pick_results['Winner'].map(seedings_dict)
    pick_results['LoserSeed'] = pick_results['Loser'].map(seedings_dict)
    pick_results['Upset'] = np.where(pick_results['WinnerSeed'] > pick_results['LoserSeed'], 'Upset', 'No')
    drop_cols_alpha = ['WinnerSeed', 'LoserSeed', 'Loser']

    pick_results = pick_results.drop(drop_cols_alpha, axis=1)    
    
    rnd+=1
    pick_results['Round'] = rnd
    #all_pick_results = all_pick_results.append(pick_results,sort=False).reset_index(drop=True)
    all_pick_results = pd.concat([all_pick_results, pick_results])
    all_pick_results = all_pick_results.reset_index(drop=True)
    
    print ('\nNational Champions')
    sixth_round_winners = list(pick_results['Winner'])
    for team in sixth_round_winners:
        print ('%s: %s' %(seedings_dict[team], team))
    
    #all_pick_results['Winner_Prob'] = None
    all_pick_results['Winner_Prob'] = abs(all_pick_results.loc[all_pick_results['Upset'] == 'Upset']['Prob'] - 1)
    all_pick_results['Winner_Prob'] = all_pick_results['Winner_Prob'].fillna(all_pick_results['Prob'])
    all_pick_results['Winner_Prob'] = all_pick_results['Winner_Prob'].astype(float)
    all_pick_results['School_Opp_Seed'] = all_pick_results['School_Opp'].map(seedings_dict)
    all_pick_results['School_Seed'] = all_pick_results['School'].map(seedings_dict)
    
    all_pick_results['School_Opp_Seed'] = all_pick_results['School_Opp_Seed'].astype(int)
    all_pick_results['School_Seed'] = all_pick_results['School_Seed'].astype(int)
    
    all_pick_results['Loser'] = np.where(all_pick_results['Outcome'] == 'Win', all_pick_results['School_Opp'], all_pick_results['School'])
    all_pick_results['WinnerSeed'] = all_pick_results['Winner'].map(seedings_dict)
    all_pick_results['LoserSeed'] = all_pick_results['Loser'].map(seedings_dict)
    all_pick_results['Upset'] = np.where(all_pick_results['WinnerSeed'] > all_pick_results['LoserSeed'], 'Upset', 'No')
    drop_cols_alpha = ['WinnerSeed', 'LoserSeed', 'Loser']
    all_pick_results = all_pick_results.drop(drop_cols_alpha, axis=1)  
    
    print ('\nAdvanced to second round:\n',first_round_winners)
    print ('\nAdvanced to Sweet 16:\n',second_round_winners)
    print ('\nAdvanced to Elite Eight:\n',third_round_winners)
    print ('\nAdvanced to Final Four:\n',fourth_round_winners)
    print ('\nAdvanced to National Championship:\n',fifth_round_winners)
    print ('National Champions: %s' %(sixth_round_winners[0]))
    print("Results: %.2f%% (%.2f)" % (results.mean()*100, results.std()*100))
    
    return all_pick_results