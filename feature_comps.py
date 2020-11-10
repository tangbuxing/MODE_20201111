# -*-coding:utf-8-*-

import numpy as np
import sys
sys.path.append(r'F:\Work\MODE\Submit')
import Spatialvx_PA1.feature_props as fp
from Spatialvx_PA1 import utils
import math
import Spatialvx_PA1.bearing as bear
from Spatialvx_PA1.locperf import *
from Spatialvx_PA1.deltametric import *
from Spatialvx_PA1.intersect import *


def feature_comps(Y, X, which_comps=None, sizefac=1, alpha=0.1, k=4, p=2, c=float('inf'), distfun='distmapfun',
                  deg=True, aty='compass', loc=None):
    out = {}
    if which_comps is None:
        which_comps = ["cent.dist", "angle.diff", "area.ratio", "int.area", "bdelta", "haus", "ph", "med", "msd", "fom",
                       "minsep", "bearing"]
    id1 = []
    for i, val in enumerate(["cent.dist", "angle.diff", "area.ratio", "int.area", "bearing"]):
        id1.append(val in which_comps)
    if any(id1):
        list1 = []
        if "cent.dist" in which_comps or "bearing" in which_comps:
            list1.append("centroid")
        if "area.ratio" in which_comps or "int.area" in which_comps:
            list1.append("area")
        if "angle.diff" in which_comps:
            list1.append("axis")
    id2 = []
    for i, val in enumerate(["ph", "med", "msd", "fom", "minsep"]):
        id2.append(val in which_comps)
    list2 = []
    # 需要考虑id2为矩阵的情况
    if any(id2):
        list2 = np.array(["ph", "med", "msd", "fom", "minsep"])[id2]
    if any(id1):
        x_single_props = fp.feature_props(X, None, list1, sizefac**2, None, loc)
        y_single_props = fp.feature_props(Y, None, list1, sizefac**2, None, loc)
        if any(id2):
            out = locperf(X, Y, list2, alpha, k)
        else:
            out = {}
        if "cent.dist" in which_comps:
            x_cent_x = x_single_props["centroid"]["x"]
            x_cent_y = x_single_props["centroid"]["y"]
            y_cent_x = y_single_props["centroid"]["x"]
            y_cent_y = y_single_props["centroid"]["y"]
            out['cent.dist'] = math.sqrt((y_cent_x - x_cent_x)**2 + (y_cent_y - x_cent_y)**2) * sizefac
        if "angle.diff" in which_comps:
            phiX = x_single_props['axis']['OrientationAngle']['MajorAxis'] * math.pi/180
            phiY = y_single_props['axis']['OrientationAngle']['MajorAxis'] * math.pi/180
            out['angle.diff'] = abs(math.atan2(math.sin(phiX - phiY), math.cos(phiX - phiY)) * 180/math.pi)
        if "area.ratio"in which_comps or "int.area" in which_comps:
            Xa = x_single_props['area']
            Ya = y_single_props['area']
        if "area.ratio" in which_comps:
            out['area.ratio'] = min(Xa, Ya)/max(Xa, Ya)
        if "int.area" in which_comps:
            denom = (Xa + Ya)/2
            XY = intersect(X, Y)
            XYa = (fp.feature_props(XY, None, "area", sizefac**2, None, loc))["area"]
            out['int.area'] = XYa/denom
        if "bearing" in which_comps:
            out['bearing'] = bear.bearing(np.vstack((np.array(y_single_props['centroid']['x']), np.array(y_single_props['centroid']['y']))).transpose(),
                                          np.vstack((np.array(x_single_props['centroid']['x']), np.array(x_single_props['centroid']['y']))).transpose(), deg, aty)
        if "bdelta" in which_comps:
            out['bdelta'] = deltametric(X, Y, p, c)
        if "haus" in which_comps:
            out['haus'] = deltametric(X, Y, float('inf'), float('inf'))
        return out
'''
#data = np.load(r'F:\Work\MODE\tra_test\FeatureFinder\deltammResult_PA3.npy', allow_pickle = True).tolist()
data = look_deltamm.copy()
XtmpAttributes = utils.get_attributes_for_feat(data['Xlabelsfeature'])
remove_list = ['Type', 'xrange', 'yrange', 'dim', 'xstep', 'ystep', 'warnings', 'xcol', 'ycol']
xkeys = utils.remove_key_from_list(list(data['Xlabelsfeature'].keys(    )), remove_list)
Xtmp = {"m": data['Xlabelsfeature']['labels_5']}
Xtmp.update(XtmpAttributes)
YtmpAttributes = utils.get_attributes_for_feat(data['Ylabelsfeature'])
ykeys = utils.remove_key_from_list(list(data['Ylabelsfeature'].keys()), remove_list)
Ytmp = {"m": data['Ylabelsfeature']['labels_5']}
Ytmp.update(YtmpAttributes)
look_feture_comps = feature_comps(X = Xtmp, Y = Ytmp)
'''