#!/usr/bin/env python
def main():
    import pathlib
    import numpy as np
    from bi1dcnn import trainUtils, utils
    from bi1dcnn import model
    import warnings
    warnings.filterwarnings("ignore")
    '''
        修改参数的位置
        Apath：Hi-C数据
        Aoutput：训练的模型输出路径
        Abedpe：loop的先验知识，已发布的loop信息
        Aresolution：分辨率
        Awidth：窗口大小
    '''
    Apath='../training-sets/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    Aoutput='../models'
    Abedpe='../training-sets/gm12878.tang.ctcf-chiapet.hg19.bedpe'
    Aresolution=10000
    Awidth=5
    Abalance=1

    np.seterr(divide='ignore', invalid='ignore')

    pathlib.Path(Aoutput).mkdir(parents=True, exist_ok=True)

    hic_info = utils.read_hic_header(Apath)
    if hic_info is None:
        hic = False
    else:
        hic = True

    coords = trainUtils.parsebed(Abedpe, lower=2, res=Aresolution)
    kde, lower, long_start, long_end = trainUtils.get_kde(coords)#返回密度概率函数
    
    if not hic:
        import cooler
        Lib = cooler.Cooler(Apath)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = utils.get_hic_chromosomes(Apath, Aresolution)

    chromosomes.pop()
    chromosomes.pop()
    # train model per chromosome
    positive_class = {}
    negative_class = {}
    for key in chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        print('读取数据：{}'.format(key))
        if not hic:
            X = Lib.matrix(balance=Abalance,
                           sparse=True).fetch(key).tocsr()
        else:
            if Abalance:
                X = utils.csr_contact_matrix(
                    'KR', Apath, key, key, 'BP', Aresolution)
            else:
                X = utils.csr_contact_matrix(
                    'NONE', Apath, key, key, 'BP', Aresolution)#大小为(NXN),value, (row, col)
        clist = coords[chromname]
        try:
            #生成正例样本的窗口
            positive_class[chromname] = np.vstack((f for f in trainUtils.build_vector(
                    X, clist, width=Awidth)))#按垂直方向（行顺序）堆叠数组构成一个新的数组
            neg_coords = trainUtils.negative_generating(
                    X, kde, clist, lower, long_start, long_end)
            stop = len(clist)
            negative_class[chromname] = np.vstack((f for f in trainUtils.build_vector(
                    X, neg_coords, width=Awidth,
                    positive=False, stop=stop)))

        except:
            print(chromname, ' 读取失败请检查数据内容。')
    
    for key in chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr'+key
        #垂直（按照行顺序）的把数组给堆叠起来。
        Xtrain_windows = np.vstack(
            (v for k, v in positive_class.items() if k != chromname))
        Xtest_windows = np.vstack(
            (v for k, v in positive_class.items() if k == chromname))
        Xtrain=np.array([Xtrain_windows,Xtest_windows])

        print('Xtrain shape: ',Xtrain.shape)
    
        Xfake = np.vstack(
            (v for k, v in negative_class.items() if k != chromname))
        Xfaketest = np.vstack(
            (v for k, v in negative_class.items() if k == chromname))
        Xfake=np.array([Xfake,Xfaketest])
        print('Xfake shape: ', Xfake.shape)
        print(chromname, 'pos/neg: ', Xtrain.shape[0], Xfake.shape[0])
        model.train(Xtrain, Xfake,chromname)
        print(chromname,'train finished.')
'''
    开启运行主函数
'''
main()
