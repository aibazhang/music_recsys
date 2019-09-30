
'''
- Evaluation metric
'''


import numpy as np
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
from functools import partial

class GetRanking:
    def __init__(self, clf, encoder, test_x, track_num):
        self.clf = clf
        self.encoder = encoder
        self.test_x_list = test_x.values.tolist()
        self.selected_item = test_x.track_id.tolist()
        
        self.track_num = track_num
        self.use_features = test_x.columns

    def __call__(self, ind):
        le = pd.DataFrame([self.test_x_list[ind]] * self.track_num, columns=self.use_features)
        le['track_id'] = range(self.track_num)

        proba = self.clf.predict(self.encoder.transform(le))

        recommend_list = list(np.argsort(proba)[::-1])
        return ind, recommend_list.index(self.selected_item[ind])


def calc_MPR_MRR(clf, test_x, encoder, track_num):
    '''
    Function of calculate MPR and MRR

    Args:
        - clf (fastFM.als or other) : trained model
        - test_x (pandas.DataFrame) : testing data
        - encoder (sklearn.preprocessing.OneHotEncoder) : encoder
        - track_num (int) : number of unique track
    
    return
        - MPR (float) : Mean percentile ranking
        - MRR (float) : Mean ranking
        - ranking / track_num (float) : percentile ranking
    '''
    print("generate recommend list")
    gr = GetRanking(clf=clf, encoder=encoder, test_x=test_x, track_num=track_num)

    with Pool() as p:
        ranking = list(tqdm(p.imap(gr, range(len(test_x)), len(test_x) // cpu_count()), total=len(test_x)))

    rec = np.array(ranking)
    rec_dct = dict(zip(rec[:,0], rec[:,1]))
    ranking = np.array(list(dict(sorted(rec_dct.items())).values()))
    
    MPR = np.mean(ranking / track_num)
    MRR = np.mean(1 / (ranking + 1))
    return MPR, MRR, ranking / track_num


def calc_mainstream(data_df, approach='frac', dist_thres=10):
    '''
    Function of calculate mainstramness
    cite: https://homepage.univie.ac.at/chris.bauer/.chb_eu/wp-content/uploads/2017/10/Schedl-2017-Distance-and-Rank-based-Music-Mai.pdf

    Args:
        - data_df (pd.Dataframe) : input dataset
        - approach (str) : approach of calculating mainstreamness
        - dist_thres (int) : threshold of number when calculating the difference of 2 distributions

    Return:
        - (pd.Series) : mainstreamness of each user
    '''
    assert approach in ['frac', 'KL', 'tau']
    play_count_g = data_df.groupby(['track_id'])['track_id'].count().values
    track_id = data_df.track_id.unique()
    
    rank_df = pd.DataFrame(index=track_id)
    rank_df['global'] = play_count_g
    
    all_users = data_df.user_id.unique()
    mainstream_list = list()
    
    for u in tqdm(all_users):
        rank_df['user'] = 0
        play_count_u = data_df[data_df.user_id==u].groupby(['track_id'])['track_id'].count().values
        track_id_u = data_df[data_df.user_id==u].track_id.unique()
        rank_df.loc[track_id_u, 'user'] = play_count_u

        rank_non_zero_df = rank_df[rank_df.user != 0]
        
        if approach in ['KL', 'frac']:
            rank_norm = rank_non_zero_df.divide(rank_non_zero_df.sum(axis=0))
        
        if approach == 'frac':
            mainstream_u = 1 - np.mean((abs(rank_norm['global'] - rank_norm['user'])) 
                                       / rank_norm.max(axis=1))
            mainstream_list.append(mainstream_u)
        else:
            if rank_non_zero_df.shape[0] < dist_thres:
                mainstream_u = np.nan
            if approach == 'KL':
                entropy1 = stats.entropy(rank_norm['global'], rank_norm['user'])
                entropy2 = stats.entropy(rank_norm['user'], rank_norm['global'])
                mainstream_u = 1 / (entropy1 + entropy2) * 2
            elif approach == 'tau':
                mainstream_u = stats.kendalltau(rank_non_zero_df['global'], rank_non_zero_df['user'])[0]
        
            mainstream_list.append(mainstream_u)
    
    return pd.Series(index=all_users, data=mainstream_list)