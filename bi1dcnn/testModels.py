from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.utils import resample
import numpy as np
from itertools import cycle
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, ZeroPadding2D, \
    Activation, MaxPool2D, AveragePooling2D, Add, concatenate, Conv1D,MaxPool1D
from keras.models import Model
from keras.models import load_model
import random
import keras

def label_tranform(label):
    tmp = []
    for i in range(label.shape[0]):
        if (label[i][0] > label[i][1]):
            tmp.append(0)
        else:
            tmp.append(1)
    return np.array(tmp)
def test_model(test_x,test_y,chromname):
    n_split = 10
    models = []
    for i in range(n_split):
        save_path = r'../model_1dcnn_6/' + chromname+'_'+str(i+1) + '.h5'
        model = load_model(save_path)
        models.append(model)
    res = []
    acc=1
    mcc=1
    auc0=1
    auc1=1
    for model in models:
        if True: #acc>model.evaluate(test_x, test_y)[1]:
            acc+=model.evaluate(test_x, test_y)[1]# acc
        predict = model.predict(test_x)
        predict = np.argmax(predict, axis=-1)
        predict_hotlabel = model.predict(test_x)  # 二维的
        print('predict_hotlabel shape:', predict_hotlabel.shape)
        p = np.array(predict).flatten()
        label = np.array(test_y)
        print(p.shape)
        print(label.shape)
        ###MCC
        if True:# mcc>matthews_corrcoef(label_tranform(label), p):
            mcc+=matthews_corrcoef(label_tranform(label), p)
        ###AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(test_y[:, i], predict_hotlabel[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        if True:# auc0 > roc_auc[0]:
            auc0+=roc_auc[0]
            auc1+=roc_auc[1]  # AUC
        '''
        绘图
        lw = 2
        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(2), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig('./myresult_1dcnn_2/Roc.jpg')
        plt.close()  # 清图
        '''
    return [acc/10,mcc/10,auc0/10,auc1/10]  # [acc,mcc,auc0,auc1]


def main():
    import pathlib
    import numpy as np
    from peakachu import trainUtils, utils
    import warnings
    warnings.filterwarnings("ignore")

    Apath = '../training-sets/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    Aoutput = '../model_1dcnn_4'
    Abedpe = '../training-sets/gm12878.tang.ctcf-chiapet.hg19.bedpe'
    Aresolution = 10000
    Awidth = 5
    Abalance = 1

    np.seterr(divide='ignore', invalid='ignore')

    # pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    pathlib.Path(Aoutput).mkdir(parents=True, exist_ok=True)

    # more robust to check if a file is .hic
    # hic_info = utils.read_hic_header(args.path)
    hic_info = utils.read_hic_header(Apath)
    if hic_info is None:
        hic = False
    else:
        hic = True

    # coords = trainUtils.parsebed(args.bedpe, lower=2, res=args.resolution)#返回一个排列后的字典{(key):(a,b)}
    coords = trainUtils.parsebed(Abedpe, lower=2, res=Aresolution)
    kde, lower, long_start, long_end = trainUtils.learn_distri_kde(coords)  # 返回密度概率函数

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
    for key in chromosomes:  # len=25
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr' + key
        print('collecting from {}'.format(key))
        if not hic:
            X = Lib.matrix(balance=Abalance,
                           sparse=True).fetch(key).tocsr()
        else:
            if Abalance:
                X = utils.csr_contact_matrix(
                    'KR', Apath, key, key, 'BP', Aresolution)
            else:
                X = utils.csr_contact_matrix(
                    'NONE', Apath, key, key, 'BP', Aresolution)  # 大小为(NXN),value, (row, col)
        clist = coords[chromname]
        print('X shape:', X[15:26, 15:26].toarray().shape)
        # try:
        # 生成正例样本的窗口
        positive_class[chromname] = np.vstack((f for f in trainUtils.buildmatrix(
            X, clist, width=Awidth)))  # 按垂直方向（行顺序）堆叠数组构成一个新的数组
        #             positive_class[chromname] = trainUtils.buildmatrix(X, clist, width=Awidth)#按垂直方向（行顺序）堆叠数组构成一个新的数组
        print('positive shape:', positive_class[chromname][0].shape)
        neg_coords = trainUtils.negative_generating(
            X, kde, clist, lower, long_start, long_end)
        stop = len(clist)
        #             n_f=trainUtils.buildmatrix(X, neg_coords, width=Awidth,
        #                 positive=False, stop=stop)
        #             for f in n_f:
        #                 negative_class[chromname] =np.vstack(f)
        negative_class[chromname] = np.vstack((f for f in trainUtils.buildmatrix(
            X, neg_coords, width=Awidth,
            positive=False, stop=stop)))
        #             negative_class[chromname] = trainUtils.buildmatrix(
        #                 X, neg_coords, width=Awidth,
        #                 positive=False, stop=stop)
        print('negative shape:', negative_class[chromname][0].shape)

        # except:
        #     print(chromname, ' failed to gather fts')

    for key in chromosomes:
        if key.startswith('chr'):
            chromname = key
        else:
            chromname = 'chr' + key
        # 它是垂直（按照行顺序）的把数组给堆叠起来。
        #         Xtrain = np.vstack(
        #             (v for k, v in positive_class.items() if k != chromname))#除了目标染色体之外的22条染色体

        Xtrain_windows = np.vstack(
            (v for k, v in positive_class.items() if k != chromname))
        Xtest_windows = np.vstack(
            (v for k, v in positive_class.items() if k == chromname))
        #         Xtrain_ranks = np.vstack(
        #             (v[1] for k, v in positive_class.items() if k != chromname))
        #         Xtest_ranks = np.vstack(
        #             (v[1] for k, v in positive_class.items() if k == chromname))
        Xtrain = np.array([Xtrain_windows, Xtest_windows])

        print('Xtrain shape: ', Xtrain.shape)

        Xfake = np.vstack(
            (v for k, v in negative_class.items() if k != chromname))

        #         Xfake_windows = np.vstack(
        #             (v for k, v in negative_class.items() if k != chromname))
        Xfaketest = np.vstack(
            (v for k, v in negative_class.items() if k == chromname))
        #         Xfake_ranks = np.vstack(
        #             (v[1] for k, v in negative_class.items() if k != chromname))
        #         Xfaketest_ranks = np.vstack(
        #             (v[1] for k, v in negative_class.items() if k == chromname))
        Xfake = np.array([Xfake, Xfaketest])
        print('Xfake shape: ', Xfake.shape)
        #         np.save('../positive/'+chromname+'_Xtrain',Xtrain)
        #         np.save('../negative/'+chromname+'_Xfake', Xfake)
        print(chromname, 'pos/neg: ', Xtrain.shape[0], Xfake.shape[0])
        #         model = trainUtils.trainRF(Xtrain, Xfake)

        '''
        数据输入
        model.train(Xtrain, Xfake, chromname)
        '''
        # X: 正样本
        # F: 负样本
        X=Xtrain
        F=Xfake
        seed = 7
        np.random.seed(seed)
        x = np.vstack((X[0], F[0]))
        test_x = np.vstack((X[1], F[1]))
        y = np.array([1] * X[0].shape[0] + [0] * F[0].shape[0])  # 标签
        test_y = np.array([1] * X[1].shape[0] + [0] * F[1].shape[0])
        tmp = y
        tmp.reshape(y.shape[0], 1)
        label = keras.utils.to_categorical(tmp, num_classes=2)
        tmp = test_y
        tmp.reshape(test_y.shape[0], 1)
        test_label = keras.utils.to_categorical(tmp, num_classes=2)
        print('x_shape:', x.shape)
        print('y_shape:', y.shape)
        print('x_test_shape:', test_x.shape)
        print('y_test_shape:', test_y.shape)
        index = [i for i in range(len(x))]
        random.shuffle(index)
        x = x[index]
        label = label[index]
        print('load over')
        '''
        test model
        '''
        #     train_x=np.expand_dims(x, 2)
        #     train_y=label
        #     test_x=np.expand_dims(test_x, 2)#test_x[:,:,:,None]
        #     test_y=test_label
        features = 243
        train_x = np.zeros((len(x), features, 1))
        train_x[:, :, 0] = x[:, :]
        test_x_r = np.zeros((len(test_x), features, 1))
        test_x_r[:, :, 0] = test_x[:, :]
        train_y = label
        test_y = test_label
        print('train_x_shape:', train_x.shape)
        print('test_x_r_shape:', test_x_r.shape)
        result=test_model(test_x_r,test_y,chromname)
        with open('./model_result2/'+chromname+'.txt','w') as f:
            for i in range(len(result)):
                f.write(str(result[i])+"\r")
        print(chromname, 'train tested.')
#        break;
#         joblib.dump(model, Aoutput+'/'+chromname +
#                     '.pkl', compress=('xz', 3))
main()
