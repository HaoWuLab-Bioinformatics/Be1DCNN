#!/usr/bin/env python
# Program to train classifier given a cooler file and
# paired bedfile containing ChIA-PET peaks
# Author: Tarik Salameh

import numpy as np

from collections import defaultdict, Counter
from scipy import stats

import random
import warnings
warnings.filterwarnings("ignore")

def build_vector(Matrix, coords, width=5, lower=1, positive=True, stop=5000):
    '''
        生成loop样本的特征向量 from Peakchu
    '''
    negcount = 0
    for c in coords:#(a,b)
        x, y = c[0], c[1]
        if y-x < lower:
            pass
        else:
            try:
                window = Matrix[x-width:x+width+1,y-width:y+width+1].toarray()#取窗口
                if np.count_nonzero(window) < window.size*.1:
                    pass
                else:
                    center = window[width, width]
                    ls = window.shape[0]#11
                    p2LL = center/np.mean(window[ls-1-ls//4:ls, :1+ls//4])#中心像素/左下象限像素的平均值
                    if positive and p2LL < 0.1:
                        pass
                    else:
                        p2lls = np.array([p2LL])
                        ranks = stats.rankdata(window, method='ordinal')#.reshape(ls,ls)计算矩阵的rank（秩）
                        window = np.hstack(
                                (window.flatten(), ranks, p2lls))
                        window = window.flatten()
                        s2 = (1+2*width)**2
                        s2 //= 2
                        if window.size == 1+2*(2*width+1)**2 and np.all(np.isfinite(window)):
                            if not positive:
                                negcount += 1
                            if negcount >= stop:
                                raise StopIteration
                            yield window#迭代器
            except:
                pass
def parsebed(chiafile, res=10000, lower=1, upper=5000000):

    coords = defaultdict(set)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, b = float(s[1]), float(s[4])
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            a //= res
            b //= res
            # all chromosomes including X and Y
            if (b-a > lower) and (b-a < upper) and 'M' not in s[0]:
                # always has prefix "chr", avoid potential bugs
                chrom = 'chr' + s[0].lstrip('chr')
                coords[chrom].add((a, b))

    for c in coords:
        coords[c] = sorted(coords[c])

    return coords


def get_kde(coords):

    dis = []
    for c in coords:
        for a, b in coords[c]:
            dis.append(b-a)

    lower = min(dis)

    # part 1: same distance distribution as the positive input
    kde = stats.gaussian_kde(dis)

    # part 2: random long-range interactions
    counts, bins = np.histogram(dis, bins=100)
    long_end = int(bins[-1])
    tp = np.where(np.diff(counts) >= 0)[0] + 2
    long_start = int(bins[tp[0]])

    return kde, lower, long_start, long_end


def negative_generating(M, kde, positives, lower, long_start, long_end):

    positives = set(positives)
    N = 3 * len(positives)
    # part 1: kde trained from positive input
    part1 = kde.resample(N).astype(int).ravel()
    part1 = part1[(part1 >= lower) & (part1 <= long_end)]

    # part 2: random long-range interactions
    part2 = []
    pool = np.arange(long_start, long_end+1)
    tmp = np.cumsum(M.shape[0]-pool)
    ref = tmp / tmp[-1]
    for i in range(N):
        r = np.random.random()
        ii = np.searchsorted(ref, r)
        part2.append(pool[ii])

    sample_dis = Counter(list(part1) + part2)

    neg_coords = []
    midx = np.arange(M.shape[0])
    for i in sorted(sample_dis):  # i cannot be zero
        n_d = sample_dis[i]
        R, C = midx[:-i], midx[i:]
        tmp = np.array(M[R, C]).ravel()
        tmp[np.isnan(tmp)] = 0
        mask = tmp > 0
        R, C = R[mask], C[mask]
        pool = set(zip(R, C)) - positives
        sub = random.sample(pool, n_d)
        neg_coords.extend(sub)

    random.shuffle(neg_coords)

    return neg_coords
