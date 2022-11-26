import numpy as np
import pandas as pd

import config

def _info(s):
    print('---')
    print(s)
    print('---')


def _get_parcel(roi, net=7):
    '''
    return:
    parcel: grayordinate -> ROI map
    nw_info: subnetwork tags for each ROI
    '''
    parcel_path = (config.PARCEL_DIR +
                   '/Schaefer2018_%dParcels_%dNetworks_order.csv' % (roi, net))

    df = pd.read_csv(parcel_path)
    parcel = np.array(df['ROI'])

    info_path = parcel_path.replace('.csv', '_info_condensed.csv')
    df = pd.read_csv(info_path)
    nw_info = np.array(df['network'])

    return parcel, nw_info