
'''
construct the dataset

Reading raw dataset -> Sample negative -> train_x, train_y, test_x, test_y

'''


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

import sampling
from features import non_categorical_features, music_content_features

def read_data(dataset='nowplaying-rs', test=100, 
              usecols=None, n_clusters=None):
    '''
    Args
        - dataset (str) : name of dataset
        - test (int) : test or not 
        - usecols (str) : using features
        - n_clusters (int) : use ucp_cluster / number of clusters
    Return
        data_df (pandas.DataFrame) using dataset
    '''

    if dataset == 'nowplaying-rs':
        data_dir = "nowplaying-RS-Music-Reco-FM/nowplaying_cleaned.csv"
    elif dataset == 'mmtd':
        data_dir = "mmtd/mmtd_nowplaying.csv"
    elif dataset == 'LFM-1b': 
        data_dir = "LFM-1b/LFM-1b/LFM-1b_LEs_2013_UGP_MS.csv"    
    
    if usecols == None or 'ucp_cluster' not in usecols:
        if not test:
            data_df = pd.read_csv('./../../' + data_dir, usecols=usecols)
        else:
            data_df = pd.read_csv('./../../' + data_dir, nrows=test, usecols=usecols)
    
    else:
        usecols.remove('ucp_cluster')
        usecols += music_content_features 
        if not test:
            data_df = pd.read_csv('./../../' + data_dir, usecols=usecols)
        else:
            data_df = pd.read_csv('./../../' + data_dir, nrows=test, usecols=usecols)
        data_df = calc_ucp_cluster(data_df, n_clusters)
    print(data_df.head())
    return data_df

def calc_ucp_cluster(data_df, n_clusters):
    print("------user clustering------")
    user_mcp_df = data_df.groupby('user_id')[music_content_features].mean()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(user_mcp_df)
    user_mcp_df['ucp_cluster'] = kmeans.labels_
    data_df['ucp_cluster'] = user_mcp_df.loc[data_df.user_id].ucp_cluster.tolist()
    
    return data_df.drop(music_content_features, axis=1)


class ConstructData:
    def __init__(self, dataset='nowplaying-rs', features=None, test=100,
                 sampling_approach='random', negative_ratio=1, n_clusters=None):
        '''
        Args:
            - dataset (str) : name of dataset
            - features (str) : using features
            - test (int) : test or not 
            - sampling_approach (str) : use sampling approach
            - negative_ratio (int) : ratio of negative samples
        '''
        self.k = negative_ratio
        self.sampling_approach = sampling_approach
        self.dataset = dataset

        approach_name = sampling_approach['name']
        if approach_name == 'pri_pop_lang':
            if dataset == 'LFM-1b':
                essential_features = ['user_id', 'track_id', 'id', 'dayofyear', 'country']
            else:
                essential_features = ['user_id', 'track_id', 'id', 'dayofyear', 'lang']
        else:
            essential_features = ['user_id', 'track_id', 'id', 'dayofyear']

        if approach_name == 'top_dis_cont':
            essential_features.append(sampling_approach['msc_cont'])

        self.features = list(set(features).union(set(essential_features)))
        if 'ucp_cluster' not in self.features:  
            self.data_df = read_data(dataset=self.dataset, 
                                     usecols=self.features,
                                     test=test)
        else:
            self.data_df = read_data(dataset=self.dataset, 
                                     usecols=self.features, 
                                     test=test, n_clusters=n_clusters)
        self._label_encoder()
        
        if approach_name == 'random':
            self.sampling_model = sampling.RandomSampling(k=negative_ratio)
        if approach_name == 'pop':
            self.sampling_model = sampling.PopSampling(k=negative_ratio, score_lim=10)
        if approach_name == 'top_dis_pop':
            self.sampling_model = sampling.TopDiscountPopSampling(k=negative_ratio, 
                                                                  topoff=sampling_approach['topoff'])
        if approach_name == 'pri_pop':
            self.sampling_model = sampling.PriorityPopSampling(k=negative_ratio,
                                                               alpha=sampling_approach['alpha'])
        if approach_name == 'pri_pop_lang':
            self.sampling_model = sampling.LangPriorityPopSampling(k=negative_ratio,
                                                                   alpha=sampling_approach['alpha'])
        if approach_name == 'top_dis_cont':
            self.sampling_model = sampling.TopDiscountContentSampling(k=negative_ratio,
                                                                      topoff=sampling_approach['topoff'],
                                                                      content_feature=sampling_approach['msc_cont'])
        
        # non_categorical_feature
        self.cont_fea = list(set(self.features) & set(music_content_features))

    def _label_encoder(self):
        '''
        Function of label encoder
        string -> int
        '''
        categorical_features = list(set(self.features) - set(non_categorical_features) - set(['rating']))
        print(categorical_features)
        self.data_df.loc[:, categorical_features] = self.data_df[categorical_features].apply(LabelEncoder().fit_transform)

    def make_negative(self, time_window=None):
        '''
        Function of sampling negative samples

        Args:
            - time_window time windows when sampling
        '''
        assert self.k >= 1

        # dayofyear>0: except the first day due to lacking play count list of the first day
        self.negative = pd.DataFrame(list(self.data_df[self.data_df.dayofyear>0].values) * self.k, columns=self.data_df.columns)
        reviewed_items = self.data_df.track_id
        day_of_year_items = self.data_df.dayofyear

        if self.sampling_approach['name'] != 'pri_pop_lang':
            if self.sampling_approach['name'] != 'top_dis_cont':
                playing_count_daily = calc_popularity(self.data_df, time_window)
            else:
                playing_count_daily = calc_popularity(self.data_df, 
                                                      time_window, 
                                                      content_feature=self.sampling_approach['msc_cont'])
        else:
            if self.dataset == 'LFM-1b':
                user_langs = self.data_df.country
            else:
                user_langs = self.data_df.lang
            
            top_langs_index = self.data_df.lang.value_counts()[:13].index
            
            playing_count_daily_dict = dict()
            
            print('lang: all')
            playing_count_daily_dict['all'] = calc_popularity(self.data_df, time_window)
            for tl in top_langs_index:
                print('lang: ', tl)
                if self.dataset == 'LFM-1b':
                    playing_count_daily_dict[tl] = calc_popularity(self.data_df[self.data_df.country != tl], time_window)
                else:
                    playing_count_daily_dict[tl] = calc_popularity(self.data_df[self.data_df.lang != tl], time_window)

        # Negative sampling
        neg_track = list()
        if self.sampling_approach['name'] == 'random':
            for d, t in zip(tqdm(day_of_year_items), reviewed_items):
                if d == 0:
                    continue
                neg_track.extend(self.sampling_model.generate_record(item_id=t, sample_space=playing_count_daily[d-1]))
                    
    
        if self.sampling_approach['name'] in ["pop", "top_dis_pop", "pri_pop", "top_dis_cont"]:
            self.sampling_model.make_score_list(playing_count_daily)
            for d, t in zip(tqdm(day_of_year_items), reviewed_items):
                if d == 0:
                    continue
                neg_track.extend(self.sampling_model.generate_record(item_id=t, score=self.sampling_model.score_list[d-1]))


        if self.sampling_approach['name'] == 'pri_pop_lang':
            self.sampling_model.make_score_list(playing_count_daily_dict)
            for d, t, l in zip(tqdm(day_of_year_items), reviewed_items, user_langs):
                if d == 0:
                    continue
                if l not in top_langs_index:
                    neg_track.extend(self.sampling_model.generate_record(item_id=t, score=self.sampling_model.score_dict['all'][d-1]))                 
                else:
                    neg_track.extend(self.sampling_model.generate_record(item_id=t, score=self.sampling_model.score_dict[l][d-1]))                 

        self.negative.loc[:, 'track_id'] = neg_track
        self.negative.set_index('id', inplace=True)
        self.data_df.set_index('id', inplace=True)


def calc_popularity(data_df, time_window, content_feature=None):
    '''
    Function of calculating popularity of each day

    Args:
        - data_df (pd.Dataframe) : input dateset
        - time_window time windows when sampling

    Return:
        - popularity_daily (list) : popularity of each day
    '''
    popularity_daily = list()
    dayofyear_range = range(data_df.tail(1).dayofyear.tolist()[0])
    for i in tqdm(dayofyear_range):
        if time_window is not None:
            if i+1 < time_window:
                one_day_data_df = data_df[data_df.dayofyear <= (i+1)]
            else:
                one_day_data_df = data_df[(data_df.dayofyear <= (i+1)) & (data_df.dayofyear > i+1-time_window)]
        else:
            one_day_data_df = data_df[data_df.dayofyear <= (i+1)]

        if content_feature == None:
            popularity_daily.append(one_day_data_df.track_id.value_counts())
        else:
            track_df = one_day_data_df.drop_duplicates(subset=['track_id'])
            track_series = pd.Series(index=track_df.track_id, 
                                     data=track_df[content_feature].tolist())
            popularity_daily.append(track_series)
    return popularity_daily


def split_train_test(train_positive, test_positive, negative, 
                     negative_ratio, balance_sample, use_features):
    '''
    split positive & negative samples to train and testing set
    testing set did not negative samples 

    Args:
        - train_positive (pandas.DataFrame) : train set of positive samples (not including rating)
        - test_positive (pandas.DataFrame) : testing set of positive samples (not including rating)
        - negative (pandas.DataFrame) : all negative samples
        - negative_ratio (int) : ratio of negative samples
        - balance_sample (bool) : whether balance positive and negative samples or not 
        - use_features (list) : use features

    Return
        - train_x (pandas.DataFrame) : training data
        - train_y (pandas.DataFrame) : training label
        - test_x (pandas.DataFrame) : testing data
        - test_y (pandas.DataFrame) : testing label
    '''
    
    use_index = set(train_positive.index) & set(negative.index)
    train_negative = negative[use_features].loc[tuple(use_index), ]

    if negative_ratio > 1 and balance_sample:
        train_positive_copy = train_positive.copy()
        for _ in range(negative_ratio - 1):
            train_positive = pd.concat([train_positive, train_positive_copy], 
                                       ignore_index=True)

    train_x = pd.concat([train_positive, train_negative], ignore_index=True)
    train_y = np.array([1] * len(train_positive) + [0] * len(train_negative))
    test_x = test_positive
    test_y = np.array([1] * len(test_positive))

    return train_x, train_y, test_x, test_y

            