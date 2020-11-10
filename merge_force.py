import pandas as pd
import numpy as np

def merge_force(x, verbose=False):

    out = {}
    if x.__contains__('implicit_merges') and x.__contains__('merges'):
        print('MergeForce: both implicit_merges and merges components found.  Using merges component.')
        m = x['merges']
    elif not x.__contains__('implicit_merges') and not x.__contains__('merges'):
        if verbose:
            print('\nNo merges found.  Returning object x unaltered.\n')
        return x
    elif not x.__contains__('merges'):
        m = x['implicit_merges']
    else:
        m = x['merges']

    out['X'] = x['X']
    out['Xhat'] = x['Xhat']
    out['identifier_function'] = x['identifier_function']
    out['identifier_label'] = x['identifier_label']
    out['match_type'] = ['MergeForce', x['match_type']]
    out['match_message'] = ''.join((x['match_message'], " (merged) "))
    xdim = x['Xlabeled'].shape

    nmatches = len(m)
    ar = np.arange(1, nmatches + 1, 1)
    matches = np.vstack((ar, ar)).T
    matches = pd.DataFrame(matches, columns = ["Forecast", "Observed"])
    out['matches'] = matches
    xp = x['Xlabelsfeature']
    yp = x['Ylabelsfeature']
    xfeats = {}
    yfeats = {}
    xlabeled = np.zeros([xdim[0], xdim[1]], dtype=int)
    ylabeled = np.zeros([xdim[0], xdim[1]], dtype=int)

    if verbose:
        print("Loop through ", nmatches, " components of merges list to set up new (matched) features.\n")
    for i in range(nmatches):
        if verbose:
            print(i, " ")
        tmp = np.array(m[i])
        uX = sorted(set(tmp[:, 1]))
        uY = sorted(set(tmp[:, 0]))
        nX = len(uX)
        nY = len(uY)
        xtmp = xp['labels_' + str(uX[0] + 1)]
        ytmp = yp['labels_' + str(uY[0] + 1)]
        if nX > 1:
            for j in range(1, nX):
                xtmp = xtmp | xp['labels_' + str(uX[j] + 1)]
        if nY > 1:
            for k in range(1, nY):
                ytmp = ytmp | xp['labels_' + str(uY[k] + 1)]
        xfeats[i] = xtmp
        yfeats[i] = ytmp
        xlabeled[xtmp] = i
        ylabeled[ytmp] = i
        xtmp = ytmp = None

    if x['unmatched']['X'] == None or x['unmatched']['X'] == 'None':
        unX = x['unmatched']['X']
        nX2 = 0
    elif type(x['unmatched']['X']) == int:
        unX = x['unmatched']['X']
        nX2 = 1
    else:
        unX = sorted(x['unmatched']['X'])
        nX2 = len(unX)
    if x['unmatched']['Xhat'] == None or x['unmatched']['Xhat'] == 'None':
        unY = x['unmatched']['Xhat']
        nY2 = 0
    elif type(x['unmatched']['Xhat']) == int:
        unY = x['unmatched']['Xhat']
        nY2 = 1
    else:
        unY = sorted(x['unmatched']['Xhat'])
        nY2 = len(unY)
    if nX2 > 0:
        if verbose:
            print("\nLoop to add/re-label all unmatched observed features.\n")
        vxunmatched = list(range((nmatches + 1), (nmatches + nX2)))
        for i in range(nX2):
            xtmp = xp['labels_' + str(unX[i] + 1)]
            xfeats[nmatches + i - 1] = xtmp
            xlabeled[xtmp] = nmatches + i
    else:
        vxunmatched = 0
    if nY2 > 0:
        if verbose:
            print("\nLoop to add/re-label all unmatched forecast features.\n")
        fcunmatched = list(range((nmatches + 1), (nmatches + nY2)))
        for i in range(nY2):
            ytmp = yp['labels_' + str(unY[i] + 1)]
            yfeats[nmatches + i - 1] = ytmp
            ylabeled[ytmp] = nmatches + i
    else:
        fcunmatched = 0
    out['Xfeats'] = xfeats
    out['Yfeats'] = yfeats
    out['Xlabeled'] = xlabeled
    out['Ylabeled'] = ylabeled
    out['unmatched'] = {'X': vxunmatched, 'Xhat': fcunmatched}
    out['MergeForced'] = True
    return out
