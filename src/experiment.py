'''
Script for experiment
'''

import argparse
import json

from copy import deepcopy
from datetime import datetime
from pandas import DataFrame, concat

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from construct_dataset import ConstructData, split_train_test, read_data
from utils import calc_MPR_MRR, FFMFormatPandas

import xlearn as xl
from fastFM import als as factorization_machine


def analysis(positive, negative, algo, use_features, 
             balance_sample, negative_ratio, split_ratio=0.8):
    '''
    Fuction of analysising the result of recommendation
    Args:
        - positive (pandas.DataFrame) : positive samples 
        - negative (pandas.DataFrame) : negative samples
        - algo (fastFM.als) : algorithm for recommedation
        - use_features (list) : use feature
        - balance_sample (bool) : whether balance positive and negative samples or not

    Return
        - split_index (int) : value of index to split train and testing set
        - perc_reaking (list) : percentile ranking of each sample in testing set
        - pred (list) : prediction value of each sample in testing set
    '''
    positive = positive[use_features]
    n_track = len(positive.track_id.unique())
    n_sample = len(positive)
    
    split_index = int(positive.shape[0] * split_ratio)
    train = positive.iloc[:split_index, ]
    testing = positive.iloc[split_index:, ]
    model = algo
    
    train_x, train_y, test_x, test_y = split_train_test(train_positive=train, test_positive=testing, 
                                                        negative=negative, negative_ratio=negative_ratio, 
                                                        balance_sample=balance_sample, use_features=use_features)
    # one hot encoder
    encoder = OneHotEncoder(handle_unknown='ignore').fit(train_x)
    train_x_en = encoder.transform(train_x)
    test_x_en = encoder.transform(test_x)

    model.fit(train_x_en, train_y)
    pred = model.predict(test_x_en)

    MPR, MRR, perc_ranking = calc_MPR_MRR(model, test_x, encoder, n_track)

    return split_index, perc_ranking, pred


def cross_validation_time_series(positive, negative, algo, algo_name, use_features, 
                                 nfold, accumulate, balance_sample, negative_ratio):
    '''
    Fuction of cross validation based on time series
    Args:
        - positive (pandas.DataFrame) : positive samples 
        - negative (pandas.DataFrame) : negative samples
        - algo (fastFM.als) : algorithm for recommedation
        - algo_name (str) : name of algorithm
        - use_features (list) : use feature
        - nfold (int) : flod size of cross validation
        - accumulate (bool) : accumulate or not when using CV
        - balance_sample (bool) : whether balance positive and negative samples or not

    Return
        - evaluate_result (dict) : out put result
    '''
    print('------cross validation------')

    evaluate_result = dict(MPR=list(), MRR=list(), RMSE=list(), TIME=list(), Accuracy=list())
    positive = positive[use_features]
    n_track = len(positive.track_id.unique())
    n_sample = len(positive)

    print('use_features:', use_features)
    print(positive.columns)

    for i in range(nfold):
        
        if not accumulate:
            train_index = [int(i / (nfold + nfold - 1) * n_sample),
                           int((i + nfold - 1) / (nfold + nfold - 1) * n_sample)]
        else:
            train_index = [0, int((i + nfold - 1) / (nfold + nfold - 1) * n_sample)]

        test_index = [int((i + nfold - 1) / (nfold + nfold - 1) * n_sample),
                    int((i + nfold) / (nfold + nfold - 1) * n_sample)]   
        print(train_index, test_index)
        train = positive.iloc[train_index[0]:train_index[1], ]
        testing = positive.iloc[test_index[0]:test_index[1], ]

        train_x, train_y, test_x, test_y = split_train_test(train_positive=train, test_positive=testing,
                                                            negative=negative, negative_ratio=negative_ratio, 
                                                            balance_sample=balance_sample, use_features=use_features)
        

        start_time = datetime.now()
        encoder = None  
        if algo_name == 'FM':
            # one hot encoder   
            encoder = OneHotEncoder(handle_unknown='ignore').fit(train_x)
            train_x_en = encoder.transform(train_x)
            test_x_en = encoder.transform(test_x)

            # training model
            model = deepcopy(algo)
            model.fit(train_x_en, train_y)
            pred = model.predict(test_x_en)

        if algo_name == 'xlearn_FM':
            '''
            TODO:
                https://xlearn-doc.readthedocs.io/en/latest/python_api/index.html#online-learning
            '''            
            # Pandas Dataframe to FFMFormat
            for f in use_features:
                train_x[f] = train_x[f].map(str)
                test_x[f] = test_x[f].map(str)
            train_x['rating'] = train_y
            test_x['rating'] = test_y

            train_test = concat([train_x, test_x], ignore_index=True)
            encoder = FFMFormatPandas()
            encoder.fit(train_test, y='rating')
            ffm_train_test = encoder.transform(train_test)
            
            ffm_train = ffm_train_test.iloc[:train_x.shape[0]]
            ffm_test = ffm_train_test.iloc[train_x.shape[0]:]

            ffm_train.to_csv(path='ffm_train.csv', index=False)
            ffm_test.to_csv(path='ffm_test.csv', index=False)
            
            # training model
            model = algo
            model.fit('./ffm_train.csv')
            pred = model.predict('./ffm_test.csv')

        evaluate_result['TIME'].append((datetime.now() - start_time).total_seconds())
        # calculating MPR and MRR
        MPR, MRR, perc_ranking = calc_MPR_MRR(model, test_x, encoder, n_track)
        evaluate_result['MPR'].append(MPR)
        evaluate_result['MRR'].append(MRR)       
        evaluate_result['Accuracy'].append(accuracy_score(test_y, pred>0.5))
        evaluate_result['RMSE'].append(mean_squared_error(test_y, pred))

    return evaluate_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # arguments about selecting model
    parser.add_argument("-algo", help="name and hyper-prameter use model", type=json.loads, 
                        default={"name":"FM"})
    parser.add_argument("-sampling_approach", help="name of sampling approach", 
                        type=json.loads, default={"name":"random"})
    parser.add_argument("-features", help="list of using feature(s)", type=str, nargs='+', default=['user_id', 'track_id'])
    parser.add_argument("-n_clusters", help="number of clusters", type=int, default=5)

    # arguments about sampling
    parser.add_argument("-negative_ratio", help="ratio of negative samples", 
                        type=int, default=3)
    parser.add_argument("-balance_sample", help="whether balance positive and negative samples or not", 
                        type=bool, default=True)
    parser.add_argument("-time_windows", help="time windows when sampling", type=int, default=None)
    parser.add_argument("-weighted_negative", help="rating of negative sampling", type=float, default=0)

    # arguments about evalution
    parser.add_argument("-nfold", help="fold size of cross validation", type=int, default=5)
    parser.add_argument("-accumulate", help="accumulate or not when using CV", type=bool, default=True)
    parser.add_argument("-analysis", help="analysis or not", type=bool, default=False)
    
    # arguments about read & write file
    parser.add_argument("-test_flag", help="test or not", type=int, default=20000)
    parser.add_argument("-dataset", help="name of dataset", type=str, 
                        choices=['nowplaying-rs', 'mmtd', 'LFM-1b'], default="nowplaying-rs")
    parser.add_argument("-out_flag", help="output or not", type=bool, default=True)
    parser.add_argument("-out_dir", help="the dir of output", type=str, default="../experiemnt_result/")

    args = parser.parse_args()

    sam_apr_name = args.sampling_approach['name']
    assert sam_apr_name in ["random", "pop", "top_dis_pop", "pri_pop", "pri_pop_lang", "top_dis_cont"]
    if sam_apr_name == "pri_pop":
        assert {'alpha'}.issubset(set(args.sampling_approach.keys()))
    elif sam_apr_name == "pop":
        pass
    elif sam_apr_name == 'top_dis_pop':
        pass
    
    assert args.algo['name'] in ["FM", "xlearn_FM"]
    if args.algo['name'] == "FM":
        algo = factorization_machine.FMRegression(rank=8, n_iter=100, l2_reg_w=0.1, l2_reg_V=0.1)
    if args.algo['name'] == "xlearn_FM":
        algo = xl.FMModel(task='reg', init=0.1, 
                          epoch=10, k=4, lr=0.2, 
                          reg_lambda=0.01, opt='sgd', 
                          metric='mae')

    if '\r' in args.features:
        args.features.remove('\r')
    features = args.features

    print('------loading data------')
    cd = ConstructData(dataset=args.dataset, features=features, test=args.test_flag,
                       sampling_approach=args.sampling_approach, negative_ratio=args.negative_ratio,
                       n_clusters=args.n_clusters)
    print('------negative sampling------')
    cd.make_negative(time_window=args.time_windows)
    

    if not args.analysis:
        result = cross_validation_time_series(positive=cd.data_df, negative=cd.negative, 
                                              algo=algo, algo_name=args.algo['name'],
                                              use_features=args.features, nfold=args.nfold, 
                                              accumulate=args.accumulate, balance_sample=args.balance_sample,
                                              negative_ratio=args.negative_ratio)

        result = DataFrame(result)
        print(result)
        nowtime = datetime.now().strftime("%Y%m%d%H%M%S")
        if "ucp_cluster" not in features:
            output_filename = 'ds_{}_m_{}_sa_{}_f_{}_k_{}_t_{}.csv'.format(args.dataset, args.algo, 
                                                                        str(args.sampling_approach).replace(':', '_'),
                                                                        args.features, args.negative_ratio, nowtime)
        else:
            output_filename = 'ds_{}_m_{}_sa_{}_f_{}_ucp_{}_k_{}_t_{}.csv'.format(args.dataset, args.algo, 
                                                                            str(args.sampling_approach).replace(':', '_'),
                                                                            args.features, args.n_clusters,
                                                                            args.negative_ratio, nowtime)
                             
    
    else:
        dataset_df = read_data(dataset=args.dataset, test=args.test_flag)
        split_index, PR, pred = analysis(positive=cd.data_df, negative=cd.negative, algo=algo, 
                                        use_features=args.features, balance_sample=args.balance_sample,
                                        negative_ratio=args.negative_ratio)
        result = dataset_df.iloc[split_index:, ]
        result['pre_rating'] = pred
        result['percentage_ranking'] = PR
        nowtime = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = 'analysis_ds_{}_m_{}_sa_{}_f_{}_k_{}_t_{}.csv'.format(args.dataset, str(args.algo).replace(':', '_'), 
                                                                                str(args.sampling_approach).replace(':', '_'),
                                                                                args.features, args.negative_ratio, nowtime)
    print("output filename:", output_filename)
    if args.test_flag == 0:
        result.to_csv(args.out_dir + output_filename)
    
    
    print('------Finished------')
    
    '''
    TODO
    '''
    

