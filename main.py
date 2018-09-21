from some_libs import *
from config import *
import glob
import pickle
import re

config = {
    'maps': None
}
nMost = 500


def read_f_data(path):
    if not os.path.isfile(path):
        print("read_f_data @{} FAILed!!!".format(path))
        assert 0
    file = os.path.basename(path)
    hz = re.findall(r"[-+]?\d*\.\d+|\d+", file)
    assert len(hz)==1
    hz=float(hz[0])
    assert hz==4 or hz==4.5 or hz==5 or hz==5.5 or hz==6
    df = pd.read_csv(path, skiprows =2, engine='python')
    df.drop(['x(microns)'], axis=1, inplace=True)
    assert df.shape[0]==301 and df.shape[1]==1
    df.loc[len(df)] = [hz]
    print("\rread_f_data @{} ... df={} hz={}OK".format(path,df.shape,hz),end="")
    return df

def GetBackGround(dirs):
    files = glob.glob(dirs + "*")
    f_datas = []
    assert len(files) == 5
    for file in files:
        f_dat = read_f_data(file)
        f_datas.append(f_dat)
    # assert len(f_datas) == 5
    df = pd.concat(f_datas).reset_index(drop=True)
    # assert df.shape[0]==302*5 and df.shape[1]==1
    df = df.transpose()
    return df

def prep_data(config, file_sets, save_to_path, nMost=5,dfBack=None):
    isNPZ = False
    if False and os.path.isfile(save_to_path):
        print('====== Load pickle={}...'.format(save_to_path))
        if isNPZ:
            with np.load(save_to_path) as data:
                df = data['df']
        else:
            with open(save_to_path, "rb") as fp:  # Pickling
                df = pickle.load(fp)
                return df

    start = time.time()
    samp_dirs = []
    list_dfs, list_day = [], []
    for path in file_sets:
        samp_dirs += glob.glob(path+"*/")
    for dirs in samp_dirs:
        files = glob.glob(dirs+"*")
        f_datas,no = [],0
        assert len(files)==5
        for file in files:
            if no==2:
                f_dat = read_f_data(file)
                f_datas.append(f_dat)
            no=no+1
        # assert len(f_datas) == 5
        df = pd.concat(f_datas).reset_index(drop=True)
        #assert df.shape[0]==302*5 and df.shape[1]==1
        df = df.transpose()
        # df -= dfBack
        list_dfs.append(df)
        if len(list_dfs)>nMost: break
    gc.collect()
    # print('====== MEMORY={:.2f}GB\tTIME={:.2f}'.format(mem_G(), time.time() - start))

    # https://tomaugspurger.github.io/modern-4-performance
    # list_dfs = [pd.read_csv(fp) for fp in files]
    # start = time.time()
    df_all = pd.concat(list_dfs).reset_index(drop=True)
    del list_dfs
    # list_dfs.clear()
    gc.collect()
    print('\n======df_all={} MEMORY={:.2f}GB \tTIME={:.2f}\n\tDump to{}......'.
          format(df_all.shape, mem_G(), time.time() - start, save_to_path))
    if isNPZ:
        np.savez_compressed(save_to_path, df=df_all.values)
        # np.savez(save_to_path, df=df_all.values)       #太耗内存
    else:
        with open(save_to_path, "wb") as fp:  # Pickling
            pickle.dump(df_all, fp, protocol=2)
    df_all.to_csv("{}___.csv".format(save_to_path))
    print('====== Save to pickle={}'.format(save_to_path))
    gc.collect()
    return df_all

def lgb_test_df(X_train_0, X_test, target, params,user_split=None):
    if user_split is not None:
        X_train, X_eval = user_split(X_train_0, random_state=42, shuffle=False)
        y_train, y_eval = X_train[target], X_eval[target]
        X_train.drop([target], 1, inplace=True)
        X_eval.drop([target], 1, inplace=True)
        gc.collect()
    else:
        y_train_0 = X_train_0[target]
        X_train_0.drop([target], 1, inplace=True)
        X_train, X_eval, y_train, y_eval = train_test_split(X_train_0, y_train_0, random_state=42, test_size=0.05,
                                                            shuffle=True)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_eval, y_eval)
    print("train[{}] head=\n{}\neval[{}] head=\n{} ".
          format(X_train.shape, X_train.head(), X_eval.shape, X_eval.head()))
    print("y_train=\n{}\ny_eval=\n{} ".format(y_train.head(), y_eval.head()))
    # specify your configurations as a dict


    print('Start training...')
    # train
    evals_result = {}  # to record eval results for plotting
    t0 = time.time()
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    verbose_eval=100,
                    valid_sets=lgb_eval,
                    evals_result=evals_result,
                    early_stopping_rounds=100)

    print('best_iter={}, time={} Save model...'.format(gbm.best_iteration, time.time() - t0))
    # save model to file
    gbm.save_model('model.txt')
    if False:
        print('Plot metrics recorded during training...')
        ax = lgb.plot_metric(evals_result)
        plt.show()
    print('Plot feature importances...')
    ax = lgb.plot_importance(gbm, max_num_features=30)
    plt.show()

    print('Start predicting...')
    # predict
    y_pred = None
    if X_test is not None:
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    if False:
        print('Plot 84th tree...')  # one tree use categorical feature to split
        ax = lgb.plot_tree(gbm, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
        plt.show()
        print('Plot 84th tree with graphviz...')
        graph = lgb.create_tree_digraph(gbm, tree_index=83, name='Tree84')
        graph.render(view=True)

    return y_pred

dfBack = GetBackGround('G:/xmu/yuan_guo/空白组数据/case0/')
Y = pd.read_excel('G:/xmu/yuan_guo/方法1训练输出数据.xls', sheet_name=None,header=None)
Y = Y['Sheet1']
file_set = ['G:/xmu/yuan_guo/新正演数据样本2/', 'G:/xmu/yuan_guo/新正演数据样本1/']
X_df = prep_data(config, file_set, './data/feo_detect_{}.pickle'.format(nMost), nMost,dfBack=dfBack)
nSamp = X_df.shape[0]
target = 'depth'
X_df[target] = Y[0]

assert X_df.shape[0]<=Y.shape[0]
params = {  # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': {'l2', 'auc'},
    'metric': 'l2',
    'metric': 'rmse',  # 'l2_root'
    'min_child_samples':20,
    #'min_data':1,
    'num_leaves': 128,
    # 'max_depth': 1,
    'learning_rate': 0.005,
    # 'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'nthread': 5,
}
lgb_test_df(X_df, None, target, params)