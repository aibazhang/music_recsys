
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



class FFMFormatPandas:
    '''
    invert pd.DataFrame to FFMLIB format
    
    cite: https://www.kaggle.com/mpearmain/pandas-to-libffm
    '''
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            name = '{}_{}'.format(col, val)
            if col_type.kind ==  'O':
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col_type.kind == 'i':
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, t) for idx, row in df.iterrows()})