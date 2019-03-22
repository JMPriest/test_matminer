from matminer.data_retrieval import retrieve_MDF
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, ShuffleSplit, KFold
import  multiprocessing

quick_demo = True

mdf = retrieve_MDF.MDFDataRetrieval(anonymous=True)
if __name__ == '__main__':
    # 得到数据
    query_string = 'mdf.source_name:oqmd AND (oqmd.configuration:static OR '\
        'oqmd.configuration:standard) AND dft.converged:True'
    if quick_demo:
        query_string += " AND mdf.scroll_id:<10000"

    data = mdf.get_data(query_string, unwind_arrays=False)
    print(data.head())
    # 重命名、预处理和筛选
    data = data[['oqmd.delta_e.value', 'material.composition']]
    data = data.rename(columns={'oqmd.delta_e.value': 'delta_e', 'material.composition':'composition'})
    data = StrToComposition(target_col_id='composition_obj').featurize_dataframe(data, 'composition')
    for k in ['delta_e']:
        data[k] = pd.to_numeric(data[k])

    original_count = len(data)
    data = data[~ data['delta_e'].isnull()]
    print('Removed %d/%d entries'%(original_count - len(data), original_count))

    original_count = len(data)
    data['composition'] = data['composition_obj'].apply(lambda x: x.reduced_formula)
    data.sort_values('delta_e', ascending=True, inplace=True)
    data.drop_duplicates('composition', keep='first', inplace=True)
    print('Removed %d/%d entries'%(original_count - len(data), original_count))

    original_count = len(data)
    data = data[np.logical_and(data['delta_e'] >= -20, data['delta_e'] <= 5)]
    print('Removed %d/%d entries'%(original_count - len(data), original_count))

    #建立机器学习模型
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    #获得特征名
    feature_labels = feature_calculators.feature_labels()
    #计算特征量
    data = feature_calculators.featurize_dataframe(data, col_id='composition_obj');
    print('Generated %d features'%len(feature_labels))
    print('Training set size:', 'x'.join([str(x) for x in data[feature_labels].shape]))
    #去除空值缺省值
    original_count = len(data)
    data = data[~ data[feature_labels].isnull().any(axis=1)]
    print('Removed %d/%d entries'%(original_count - len(data), original_count))
    # 调用随机森林
    # “随机森林”算法通过训练许多不同的决策树模型来工作，
    # 其中每个模型都在数据集的不同子集上进行训练。
    # 此处，调整的主要参数是每棵树考虑的特征个数
    model = GridSearchCV(RandomForestRegressor(n_estimators=20 if quick_demo else 150, n_jobs=-1),
                         param_grid=dict(max_features=range(8,15)),
                         scoring='neg_mean_squared_error',cv=ShuffleSplit(n_splits=1, test_size=0.1))
    model.fit(data[feature_labels], data['delta_e'])
    # 找到的最佳特征集
    print(model.best_score_)
    #画出最大特征量与准确度的关系
    fig, ax = plt.subplots()
    ax.scatter(model.cv_results_['param_max_features'].data,
              np.sqrt(-1 * model.cv_results_['mean_test_score']))
    ax.scatter([model.best_params_['max_features']], np.sqrt([-1*model.best_score_]), marker='o', color='r', s=40)
    ax.set_xlabel('Max. Features')
    ax.set_ylabel('RMSE (eV/atom)')
    #保存最佳模型
    model = model.best_estimator_
    #10次交叉验证
    cv_prediction = cross_val_predict(model, data[feature_labels], data['delta_e'], cv=KFold(10, shuffle=True))
    #计算汇总统计数据
    for scorer in ['r2_score', 'mean_absolute_error', 'mean_squared_error']:
        score = getattr(metrics,scorer)(data['delta_e'], cv_prediction)
        print(scorer, score)

    print(model)
    #绘制每个预测
    fig, ax = plt.subplots()

    ax.hist2d(pd.to_numeric(data['delta_e']), cv_prediction, norm=LogNorm(), bins=64, cmap='Blues', alpha=0.9)

    ax.set_xlim(ax.get_ylim())
    ax.set_ylim(ax.get_xlim())

    mae = metrics.mean_absolute_error(data['delta_e'], cv_prediction)
    r2 = metrics.r2_score(data['delta_e'], cv_prediction)
    ax.text(0.5, 0.1, 'MAE: {:.2f} eV/atom\n$R^2$:  {:.2f}'.format(mae, r2),
            transform=ax.transAxes,
           bbox={'facecolor': 'w', 'edgecolor': 'k'})

    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')

    ax.set_xlabel('DFT $\Delta H_f$ (eV/atom)')
    ax.set_ylabel('ML $\Delta H_f$ (eV/atom)')

    fig.set_size_inches(3, 3)
    fig.tight_layout()
    fig.savefig('oqmd_cv.png', dpi=320)
