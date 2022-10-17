#!/usr/bin/env python

def main(chr_name):
    import pathlib
    import os
    import numpy as np
    from bi1dcnn import scoreUtils, utils
    from keras.models import load_model
    np.seterr(divide='ignore', invalid='ignore')

    '''
        修改参数的位置
        Apath：Hi-C数据
        Aoutput：训练的模型输出路径
        Abedpe：loop的先验知识，已发布的loop信息
        Aresolution：分辨率
        Awidth：窗口大小
    '''

    aoutput='../result_1dcnn_62'
    amodel='../model_1dcnn_6/'
    apath='../training-sets/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    aresolution=10000
    awidth=5
    abalance=1
    alower=1
    aupper=500

    pathlib.Path(aoutput).mkdir(parents=True, exist_ok=True)

#    model = joblib.load(amodel)
    model_path=amodel+chr_name+'_1.h5'
    models=[]
    for i in range(10):
        model = load_model(amodel+chr_name+'_'+str(i)+'.h5')
        models.append(model)
    # more robust to check if a file is .hic
    hic_info = utils.read_hic_header(apath)
    if hic_info is None:
        hic = False
    else:
        hic = True

    if not hic:
        import cooler
        Lib = cooler.Cooler(apath)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = utils.get_hic_chromosomes(apath, aresolution)

    pre = utils.find_chrom_pre(chromosomes)
    tmp = os.path.split(model_path)[1]  # support full path
    ccname = pre + tmp.split('_')[0].lstrip('chr')
    cikada = 'chr' + ccname.lstrip('chr')  # cikada always has prefix "chr"
    if not hic:
        X = scoreUtils.Chromosome(Lib.matrix(balance=abalance, sparse=True).fetch(ccname).tocsr(),
                                      models=models,
                                      cname=cikada, lower=alower,
                                      upper=aupper, res=aresolution,
                                      width=awidth)
    else:
        if abalance:
            X = scoreUtils.Chromosome(utils.csr_contact_matrix('KR', apath, ccname, ccname, 'BP', aresolution),
                                          models=models,
                                          cname=cikada, lower=alower,
                                          upper=aupper, res=aresolution,
                                          width=awidth)
        else:
            X = scoreUtils.Chromosome(utils.csr_contact_matrix('NONE', apath, ccname, ccname, 'BP', aresolution),
                                          models=models,
                                          cname=cikada, lower=alower,
                                          upper=aupper, res=aresolution,
                                          width=awidth)
    result, R = X.score(thre=0.5)
    print("score finished.")
    X.writeBed(aoutput, result, R)
'''
    启动运行主函数，选择预测的目标染色体
'''
chrs_name=['chrX']
for chr_name in chrs_name:
    main(chr_name)
