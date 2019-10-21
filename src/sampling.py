'''
Sampling Approach

'''

import numpy as np
from pandas import qcut, read_csv
from tqdm import tqdm


class RandomSampling:
    def __init__(self, k):
        '''
        Select negative sample on random

        Args:
            - k (int) : negative sample ratio
        '''
        self.k = 1 if k < 1 else k

    def generate_record(self, item_id, sample_space):
        '''
        Function of generating negative samples one by one 

        Args:
            - item_id (int) : item id of a positive sample
            - sample_space (int) : sample sapce of negative samples
        Return:
            - record (int) : track id of selected negative sample(s)
        '''
        sample_index = np.random.randint(0, len(sample_space), self.k)
        record = sample_space.index[sample_index]
        while item_id in record:
            record = np.random.randint(0, len(sample_space), self.k)
        return record


# Popularity
class PopSampling(RandomSampling):
    def __init__(self, k, score_lim=5):
        '''
        Select negative sample based the popularity of track
        Track that has low popularity -> 
        (1,2,3,4,5)
        Args:
            - k (int) : negative sample ratio
            - score_lim (int) : number of segment
        '''
        RandomSampling.__init__(self, k=k)
        self.score_lim = score_lim

    def make_score_list(self, playing_count_daily):
        '''
        Function of calculating score list for sampling

        Args:
            - playing_count_daily (list) : stastic of play count in each day
        '''
        self.score_list = list()
        for pct in tqdm(playing_count_daily):
            score = qcut((1 / pct).rank(method='first'), self.score_lim, 
                         labels=range(self.score_lim, 0, -1))
            self.score_list.append(score / score.sum())

    def generate_record(self, item_id=None, score=None):
        '''
        Function of generating negative samples one by one 

        Args:
            - item_id (int) : item id of a positive sample
            - sample_space (int) : sample sapce of negative samples
            - score (list) : probability distribution of all tracks
        Return:
            - record (int) : track id of selected negative sample(s)
        '''
        record = np.random.choice(score.index, self.k, p=score)
        while item_id in record:
            record = np.random.choice(score.index, self.k, p=score)
        return record


class TopDiscountPopSampling(PopSampling):
    def __init__(self, k, score_lim=5, topoff=1):
        '''
        Select negative sample based the popularity of track
        Track that has low popularity -> 
        (1,2,2,2,2)
        Only cur off the selected probability of the tracks that have high popularity
        
        Args:
            - k (int) : negative sample ratio
            - score_lim (int) : number of segment
        '''
        PopSampling.__init__(self, k=k, score_lim=score_lim)
        self.topoff = topoff
    
    def make_score_list(self, playing_count_daily):
        '''
        Function of calculating score list for sampling

        Args:
            - playing_count_daily (list) : stastic of play count in each day
        '''

        self.score_list = list()
        for pct in tqdm(playing_count_daily):
            score = qcut((1 / pct).rank(method='first'), self.score_lim, 
                          labels=range(self.score_lim, 0, -1))
            score[score > self.topoff] = 2
            self.score_list.append(score / score.sum()) 


class PriorityPopSampling(PopSampling):
    def __init__(self, k, alpha=0.5):
        '''
        Select negative sample based the popularity of track
        The probability of a sample is selected $P(i)=p_i^{\alpha}/\sum_k p_k^{\alpha}$ 
        Priority: $p_i=1/\mathrm{rank}(i)$

        Args:
            - k (int) : negative sample ratio
            - score_lim (int) : number of segment
            - alpha (float) : control the difference probability of postive samples and negative ones
        ''' 
        PopSampling.__init__(self, k=k)
        self.alpha = alpha
    
    def make_score_list(self, playing_count_daily):
        '''
        Function of calculating score list for sampling

        Args:
            - playing_count_daily (list) : stastic of play count in each day
        '''

        self.score_list = list()
        for pct in tqdm(playing_count_daily):
            score = (1 / pct).rank(method='dense')
            rec_score_pow_alpha = np.power(1 / score, self.alpha)
            self.score_list.append(rec_score_pow_alpha / rec_score_pow_alpha.sum())


class LangPriorityPopSampling(PriorityPopSampling):
    def __init__(self, k, alpha=0.5):
        '''
        Select negative sample based the popularity of track
        The probability of a sample is selected $P(i)=p_i^{\alpha}/\sum_k p_k^{\alpha}$ 
        Priority: $p_i=1/\mathrm{rank}(i)$

        Args:
            - k (int) : negative sample ratio
            - score_lim (int) : number of segment
            - alpha (float) : control the difference probability of postive samples and negative ones
        ''' 
        super().__init__(k, alpha=alpha)

    def make_score_list(self, playing_count_daily_dict):
        '''
        Function of calculating score list for sampling

        Args:
            - playing_count_daily_dict (dict) : stastic of play count in each day group by language
        '''
        self.score_dict = dict()
        for lang in playing_count_daily_dict.keys():
            print('lang: ', lang)
            self.score_dict[lang] = list()
            for pct in tqdm(playing_count_daily_dict[lang]):
                score = (1 / pct).rank(method='dense')
                rec_score_pow_alpha = np.power(1 / score, self.alpha)
                self.score_dict[lang].append(rec_score_pow_alpha / rec_score_pow_alpha.sum())

