import pathlib

import numpy as np
from scipy import sparse
from scipy import stats


class Chromosome():
    def __init__(self, coomatrix, models, lower=1, upper=500, cname='chrm', res=10000, width=5):
        # cLen = coomatrix.shape[0] # seems useless
        R, C = coomatrix.nonzero()
        #修改了int转类型，因为list[]中index不能有小数
        validmask = np.isfinite(coomatrix.data) & (C-R+1 > lower) & (C-R < upper)
        #self.validmask=validmask
        #validmask = np
        R, C, data = R[validmask], C[validmask], coomatrix.data[validmask]
        self.M = sparse.csr_matrix((data, (R, C)), shape=coomatrix.shape)
        self.ridx, self.cidx = R, C
        self.chromname = cname
        self.r = res
        self.w = width
        self.models = models

    def getwindow(self, coords):
        """
        Generate training set
        :param Matrix: single chromosome dense array
        :param coords: List of tuples containing coord bins
        :param width: Distance added to center. width=5 makes 11x11 windows
        :return: yields paired positive/negative samples for training
        """
        fts, clist = [], []
        w2 = int(self.w//2)
        width = self.w
        for c in coords:
            x, y = c[0], c[1]
            distance = abs(y-x)
            try:
                window = self.M[x-width:x+width+1,
                                y-width:y+width+1].toarray()
            except:
                continue
            if np.count_nonzero(window) < window.size*.2:
                pass
            else:
                try:
                    center = window[width, width]
                    ls = window.shape[0]
                    p2LL = center/np.mean(window[ls-1-ls//4:ls, :1+ls//4])
                    indicatar_vars = np.array([p2LL])
                    ranks = stats.rankdata(window, method='ordinal')
                    window = np.hstack(
                        (window.flatten(), ranks, indicatar_vars))
                    window = window.flatten()
                    additional = 1

                    window = window.reshape((1, window.size))
                    if window.size == 1+2*(1+2*width)**2 and np.isfinite(window).all():
                        fts.append(window)
                        clist.append(c)
                except:
                    pass
        test_x = np.vstack((i for i in fts))
        print(len(test_x))
        features = 243
        test_x_r = np.zeros((len(test_x), features, 1))
        test_x_r[:, :, 0] = test_x[:, :]
        print('start predict.')
        # 返回的是一个 n 行 k 列的数组， 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1
        probas=self.models[0].predict(test_x_r)[:, 1]
        for i in range(9):
            probas+=self.models[i+1].predict(test_x_r)[:, 1]
        probas/=10
        print('finished predict.')
        return probas, clist

    def score(self, thre=0.5):
        print('scoring matrix {}'.format(self.chromname))
        print('num candidates {}'.format(self.M.data.size))#ok
        coords = [(r, c) for r, c in zip(self.ridx, self.cidx)]
        print('---------coord loading finished--------')
        p, clist = self.getwindow(coords)#p为预测结果
        print('---------getted windows----------------')
        clist = np.r_[clist]#是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
        pfilter = p > thre#根据阈值筛选
        ri = clist[:, 0][pfilter]
        ci = clist[:, 1][pfilter]
        result = sparse.csr_matrix((p[pfilter], (ri, ci)), shape=self.M.shape)
        data = np.array(self.M[ri, ci]).ravel()#将数组拉成一维
        self.M = sparse.csr_matrix((data, (ri, ci)), shape=self.M.shape)

        return result, self.M

    def writeBed(self, out, prob_csr, raw_csr):
        print('---------begin writing----------')
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        with open(out + '/' + self.chromname + '.bed', 'w') as output_bed:
            r, c = prob_csr.nonzero()
            for i in range(r.size):
                line = [self.chromname, r[i]*self.r, (r[i]+1)*self.r,
                        self.chromname, c[i]*self.r, (c[i]+1)*self.r,
                        prob_csr[r[i],c[i]], raw_csr[r[i],c[i]]]

                output_bed.write('\t'.join(list(map(str, line)))+'\n')
