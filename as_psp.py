import math


def as_psp(x):
    result = {}
    angle = x['angle']
    y0 = math.sin(angle) * x['length'] / 2 + x['ymid']
    x0 = math.cos(angle) * x['length'] / 2 + x['xmid']
    x1 = x['xmid'] - math.sin(angle) * x['length'] / 2
    y1 = x['ymid'] - math.cos(angle) * x['length'] / 2
    result['ends'] = {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
    return result
