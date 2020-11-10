# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
import sys
sys.path.append(r'F:\Work\MODE\Submit')
from Spatialvx_PA1.feature_axis import feature_axis
from Spatialvx_PA1 import utils


def feature_props(x, im=None, which_comps=None, areafac=1, q=None, loc=None):
    if which_comps is None:
        which_comps = ["centroid", "area", "axis", "intensity"]
    if q is None:
        q = [0.25, 0.9]
    out = {}
    if "centroid" in which_comps:
        '''
        #计算经纬度坐标上的质心坐标
        if loc is None:
            xd = x["m"].shape
            dim0 = xd[0]
            dim1 = xd[1]
            range0 = np.tile(np.arange(dim0), dim1)
            range1 = (np.arange(dim1)).repeat(dim0)
            loc = np.stack((range0, range1), axis=-1)
        xbool = np.reshape(x["m"], x["m"].size, 'F')
        xcen = np.mean(loc['lon'][xbool])
        ycen = np.mean(loc['lat'][xbool])
        '''
        #计算几何坐标上的质心坐标
        xcen = np.mean(np.argwhere(x['m' ]== 1)[:,0])
        ycen = np.mean(np.argwhere(x['m' ]== 1)[:,1])
        out['centroid'] = {"x": xcen, "y": ycen}
    if "area" in which_comps:
        out["area"] = np.sum(x["m"]) * areafac
    if "axis" in which_comps:
        out["axis"] = feature_axis(x, areafac)
    ## 没经过测试
    if "intensity" in which_comps:
        ivec = {}
        df = pd.DataFrame(np.array(im[x]), columns=q)
        for i, val in q:
            ivec[val] = df.quantile(val)
        out["intensity"] = ivec
    return out
'''
#data = np.load(r'F:\Work\MODE\tra_test\FeatureFinder\deltammResult_PA3.npy', allow_pickle = True).tolist()
data = look_deltamm.copy()
XtmpAttributes = utils.get_attributes_for_feat(data['Xlabelsfeature'])
remove_list = ['Type', 'xrange', 'yrange', 'dim', 'xstep', 'ystep', 'warnings', 'xcol', 'ycol']
xkeys = utils.remove_key_from_list(list(data['Xlabelsfeature'].keys(    )), remove_list)
Xtmp = {"m": data['Xlabelsfeature']['labels_5']}
Xtmp.update(XtmpAttributes)
look_feature_props = feature_props(x = Xtmp, which_comps = ["centroid", "area", "axis"])    #"intensity"属性输出的为空
'''
