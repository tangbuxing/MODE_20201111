# -*-coding:utf-8-*-
import pandas as pd
import cv2
import sys
sys.path.append(r'F:\Work\MODE\tra_test\AllFunctions_PA02')
from src.spatial_vx import utils
from src.spatial_vx.distmap import *
from src.spatial_vx.sma import sma
from src.spatial_vx.psp.angles_psp import angles_psp
from src.spatial_vx.psp.lengths_psp import lengths_psp
from src.spatial_vx.psp.midpoints_psp import midpoints_psp
from src.spatial_vx.psp.as_psp import as_psp


def feature_axis(x, fac=1, flipit=False, twixt=False):
    out = {}
    if flipit:
        x = np.transpose(x)

    out['point'] = getRedDotsCoordinatesFromLeftToRight(x['m'])
    # img = Image.fromarray(x['labels_1']).convert('RGB')
    ch = cv2.convexHull(getRedDotsCoordinatesFromLeftToRight(x['m']))
    out['chull'] = ch
    # pts = np.hstack(ch['bdry'][[1]][["x"]], ch['bdry'][[1]][["y"]])
    pts = np.zeros([len(ch), 2], dtype=int)
    for index in range(len(ch)):
        pts[index] = (ch[index][0])
    out['pts'] = pts

    axfit_frame = {'x': pts[:, 0], 'y': pts[:, 1]}
    axfit = sma(axfit_frame)
    axis_x = np.array([axfit['from'], axfit['to']])
    a = axfit['coef']['slope']
    b = axfit['coef']['intercept']
    axis_y = b + a * axis_x
    if axis_x[0] is None or axis_x[1] is None or axis_y[0] is None or axis_y[1] is None:
        return None
    # axwin = owin(xrange = range(axis_x), yrange = range(axis_y))
    axis_frame = {'x0': axis_x[0], 'y0': axis_y[0], 'x1': axis_x[1], 'y1': axis_y[1]}
    # MajorAxis = as.psp(pd.DataFrame(axis_frame), window=axwin)
    theta = angles_psp(axis_frame)
    if 0 <= theta <= math.pi / 2:
        theta2 = math.pi / 2 - theta
    else:
        theta2 = 3 * math.pi / 2 - theta
    tmp = rotate(pts, theta2)
    tmp = boundingbox(tmp, tmp)
    l = tmp['x_range'][1] - tmp['x_range'][0]
    theta = theta * 180 / math.pi
    if twixt:
        if 90 < theta <= 270:
            theta = theta - 180
        elif 270 < theta <= 360:
            theta = theta - 360
    MidPoint = midpoints_psp(axis_frame)
    r = lengths_psp(axis_frame) * fac
    phi = angles_psp(rotate(pts, math.pi / 2))
    phi = phi * 180 / math.pi
    minor_frame = {'xmid': MidPoint['x'], 'ymid': MidPoint['y'], 'length': l/fac, 'angle': phi}
    MinorAxis = as_psp(minor_frame)
    out['phi'] = phi
    out['MajorAxis'] = axis_frame
    out['MinorAxis'] = MinorAxis
    out['OrientationAngle'] = {'MajorAxis': theta, 'MinorAxis': phi}
    out['aspect_ratio'] = l / r
    out['MidPoint'] = MidPoint
    out['lengths'] = {'MajorAxis': r, 'MinorAxis': l}
    out['sma_fit'] = axfit
    return out

'''
#data = np.load(r'F:\Work\MODE\tra_test\FeatureFinder\deltammResult_PA3.npy', allow_pickle = True).tolist()
data = look_deltamm.copy()
XtmpAttributes = utils.get_attributes_for_feat(data['Xlabelsfeature'])
remove_list = ['Type', 'xrange', 'yrange', 'dim', 'xstep', 'ystep', 'warnings', 'xcol', 'ycol']
xkeys = utils.remove_key_from_list(list(data['Xlabelsfeature'].keys(    )), remove_list)
Xtmp = {"m": data['Xlabelsfeature']['labels_5']}
Xtmp.update(XtmpAttributes)
look_feature_axis = feature_axis(Xtmp)
'''