
'''
construct the dataset

Reading raw dataset -> Sample negative -> train_x, train_y, test_x, test_y

'''


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import sampling
from features import non_categorical_features, music_content_features

def read_data(dataset='nowplaying-rs', test=100, usecols=None):
    '''
    Args
        - dataset (str) : name of dataset
        - test (int) : test or not 
        - usecols (str) : using features
    Return
        data_df (pandas.DataFrame) using dataset
    '''

    if dataset == 'nowplaying-rs':
        data_dir = "nowplaying-RS-Music-Reco-FM/nowplaying_cleaned_ms.csv"
    elif dataset == 'mmtd':
        data_dir = "mmtd/mmtd_nowplaying.csv"
    elif dataset == 'LFM-1b': 
        data_dir = "LFM-1b/LFM-1b/LFM-1b_LEs_2013_UGP_MS.csv"    
    
    if not test:
        data_df = pd.read_csv('./../../' + data_dir, usecols=usecols)
    else:
        data_df = pd.read_csv('./../../' + data_dir, nrows=test, usecols=usecols)
    print(data_df.head())
    return data_df


class ConstructData:
    def __init__(self, dataset='nowplaying-rs', features=None, test=100,
                 sampling_approach='random', negative_ratio=1):
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

        essential_features = ['user_id', 'track_id', 'id', 'dayofyear']
        if features is None:
            self.features = essential_features
        else:
            self.features = list(set(features) | set(essential_features))

        self.data_df = read_data(dataset=dataset, usecols=self.features.copy(), test=test)
        self._label_encoder()

        approach_name = sampling_approach['name']
        if approach_name == 'random':
            self.sampling_model = sampling.RandomSampling(k=negative_ratio)
        if approach_name == 'pop':
            self.sampling_model = sampling.PopSampling(k=negative_ratio, score_lim=10)
        if approach_name == 'top_dis_pop':
            self.sampling_model = sampling.TopDiscountPopSampling(k=negative_ratio, 
                                                                  score_lim=5, topoff=1)
        if approach_name == 'pri_pop':
            self.sampling_model = sampling.PriorityPopSampling(k=negative_ratio, score_lim=5,
                                                               alpha=sampling_approach['alpha'])

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
        neg_track = list()

        # dayofyear>0: except the first day due to lacking play count list of the first day
        self.negative = pd.DataFrame(list(self.data_df[self.data_df.dayofyear>0].values) * self.k, columns=self.data_df.columns)
        reviewed_items = self.data_df.track_id
        day_of_year_items = self.data_df.dayofyear

        playing_count_daily = list()
        for i in tqdm(range(self.data_df.tail(1).dayofyear.tolist()[0])):
            if time_window is not None:
                if i+1 < time_window:
                    nowplaying_filter = self.data_df[self.data_df.dayofyear <= (i+1)]
                else:
                    nowplaying_filter = self.data_df[(self.data_df.dayofyear <= (i+1)) & (self.data_df.dayofyear > i+1-time_window)]
            else:
                nowplaying_filter = self.data_df[self.data_df.dayofyear <= (i+1)]
            playing_count_daily.append(nowplaying_filter.track_id.value_counts())

        
        if self.sampling_approach['name'] == 'random':
            for d, t in zip(tqdm(day_of_year_items), reviewed_items):
                if d == 0:
                    continue
                neg_track.extend(self.sampling_model.generate_record(item_id=t, sample_space=playing_count_daily[d-1]))

        if self.sampling_approach['name'] in ["pop", "top_dis_pop", "pri_pop"]:
            self.sampling_model.make_score_list(playing_count_daily)
            for d, t in zip(tqdm(day_of_year_items), reviewed_items):
                if d == 0:
                    continue
                neg_track.extend(self.sampling_model.generate_record(item_id=t, 
                                 sample_space=playing_count_daily[d-1], score=self.sampling_model.score_list[d-1]))

        self.negative.loc[:, 'track_id'] = neg_track
        self.negative.set_index('id', inplace=True)
        self.data_df.set_index('id', inplace=True)


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
            