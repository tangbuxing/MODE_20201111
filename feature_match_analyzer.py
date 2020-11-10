import sys
sys.path.append(r'F:\Work\MODE\Submit')
from Spatialvx_PA1.feature_comps import feature_comps
import numpy as np
from Spatialvx_PA1.utils import get_attributes_for_feat, remove_key_from_list


def feature_match_analyzer(x, which_comps=None, sizefac=1, alpha=0.1, k=4, p=2,
                           c=float("inf"), distfun="distmapfun", y=None, matches=None, object=None):
    if which_comps is None:
        which_comps = ["cent.dist", "angle.diff", "area.ratio",
                       "int.area", "bdelta", "haus", "ph", "med", "msd", "fom",
                       "minsep", "bearing"]
    if x['match_type'] == "centmatch":
        n = x['matches'].shape[0]
        if n > 0:
            out = []
            a = x
            ## loc 没有值
            if "loc" in a.keys():
                loc = a['loc']
            else:
                loc = None
            Xfeats = x['Xlabelsfeature']
            Yfeats = x['Ylabelsfeature']
            xattribute = get_attributes_for_feat(Xfeats)
            yattribute = get_attributes_for_feat(Yfeats)
            remove_list = ['Type', 'xrange', 'yrange', 'dim', 'xstep', 'ystep', 'warnings', 'xcol', 'ycol']
            xkeys = remove_key_from_list(list(Xfeats.keys()), remove_list)
            ykeys = remove_key_from_list(list(Yfeats.keys()), remove_list)
            for i in range(n):
                j = x['matches'][i, 0]
                k = x['matches'][i, 1]
                #ymat = Yfeats[ykeys[i]]
                #xmat = Xfeats[xkeys[i]]                
                ymat = Yfeats[ykeys[j]]
                xmat = Xfeats[xkeys[k]]
                if xmat.dtype.name is not 'bool':
                    xmat = (xmat == 1)
                if ymat.dtype.name is not 'bool':
                    ymat = (ymat == 1)
                ymat = {"m": ymat}
                ymat.update(yattribute)
                xmat = {"m": xmat}
                xmat.update(xattribute)
                out.append(feature_comps(Y=ymat, X=xmat, which_comps=which_comps,
                                         sizefac=sizefac, alpha=alpha, k=k, p=p, c=c, distfun=distfun, loc=loc))
        else:
            out = "No matches found"
        return out
    elif x['match_type'] == "deltamm":
        if matches is not None:
            obj = matches
        elif y is not None:
            obj = y
        else:
            obj = x
        if "loc" in obj.keys():
            loc = obj['loc']
        else:
            loc = None
        Yfeats = obj['Ylabelsfeature']
        Xfeats = obj['Xlabelsfeature']
        xattribute = get_attributes_for_feat(Xfeats)
        yattribute = get_attributes_for_feat(Yfeats)
        #if obj['unmatched']['matches'].shape[0] == 0:
        if obj['matches']['X'].shape[0] == 0:
            out = "No matches found"
        else:
            #n = obj['unmatched']['matches'].shape[0]
            n = obj['matches']['X'].shape[0]
            out = []
            remove_list = ['Type', 'xrange', 'yrange', 'dim', 'xstep', 'ystep', 'warnings', 'xcol', 'ycol']
            xkeys = remove_key_from_list(list(Xfeats.keys()), remove_list)
            ykeys = remove_key_from_list(list(Yfeats.keys()), remove_list)
            for i in range(n):
                ymat = Yfeats[ykeys[i]]
                xmat = Xfeats[xkeys[i]]
                if xmat.dtype.name is not 'bool':
                    xmat = (xmat == 1)
                if ymat.dtype.name is not 'bool':
                    ymat = (ymat == 1)
                ymat = {"m": ymat}
                ymat.update(yattribute)
                xmat = {"m": xmat}
                xmat.update(xattribute)
                out.append(feature_comps(Y=ymat, X=xmat, which_comps=which_comps, sizefac=sizefac,
                                         alpha=alpha, k=k, p=p, c=c, distfun=distfun, loc=loc))
        return out
    else:
        print("类型错误")
        raise Exception("类型错误")

'''
if __name__ == '__main__':
    #data1 = np.load(r"F:\Work\MODE\tra_test\centmatch\centmatchResult_PA2.npy", allow_pickle=True).tolist()
    #data2 = np.load(r'F:\Work\MODE\tra_test\FeatureFinder\deltammResult_PA3.npy', allow_pickle=True).tolist()
    data1 = look_centmatch.copy()
    data2 = look_deltamm.copy()
    data1['Xlabelsfeature'] = data2['Xlabelsfeature']
    data1['Ylabelsfeature'] = data2['Ylabelsfeature']
    look_feature_match_analyzer_centmatch = feature_match_analyzer(data1)
    look_feature_match_analyzer_deltamm = feature_match_analyzer(data2)
    print("hello")
'''