import uproot
import pandas as pd
from itertools import chain, product
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter

def readRootFile(filename):
    """Import data from a simplified test beam ROOT file of chamber with MICROROC ASU
    Parameters:
    -----------
        filename :   path of ROOT file
    
    Returns:
    --------
        df :     Pandas DataFrame
    """

    file = uproot.open(filename)
    file.keys()
    tree = file['tvec']
    a = tree.array("xpos")
    df_cols = ['xpos', 'ypos', 'chbid', 'dt', 'digit', 'chipid', 'hardid']
    df = pd.DataFrame(list(chain(*[list(product([x],y)) \
                      for x, y in zip(range(len(a)), tree.array("xpos"))])),
                      columns= ['eventId',"xpos"])
    for col in df_cols[1:]:
        df[col] = tree.array(col).flatten()

    return df


def fixoffset(df, xoff, yoff):
    """Offset correction for small MM chambers.
    Parameters:
    -----------
        df :    Pandas DataFrame.
        xoff :  offset of x position.
        yoff :  offset of y position.
   
    Returns:
    --------
        df :    modified DataFrame.
    """

    df.loc[df.chbid < 4, 'ypos'] += 17
    df.loc[df.chbid < 4, 'xpos'] += 18
    
    return df


def filterNoisyChannels(df, Nchb = 8):
    """ Fintering noisy channels from RPWELL chambers
    The channels in the interface of the active area.
    Parameters:
    -----------
        df :    Pandas DataFrame.
        Nchb :       the number of chambers in the setup. Default: 8.
   
    Returns:
    --------
        df_quiet :    Pandas DataFrame without noisy channels.
    """
    # the first element in a row of MR is the chip ID followed by the channel ID

    MR = [[1, 0, 1, 4, 89, 15, 23],
          [5, 47, 53, 58, 57, 61, 63, 62],
          [6, 6, 13, 20, 27, 33, 39, 44, 49, 53, 57, 60, 62, 63],
          [7, 7, 15, 23, 31, 39, 47, 55, 63],
          [8, 0, 2, 5, 9, 13, 18, 23, 29, 35, 42, 49, 56, 63],
          [9, 0, 1, 2, 3, 8, 9, 10, 11, 20],
          [14, 51, 59, 58, 57, 56, 63, 62, 61, 60],
          [15, 0, 1, 2, 3, 7, 6, 5, 4, 12],
          [20, 63, 62, 61, 60, 55, 54, 53, 52, 43],
          [21, 0, 1, 2, 6, 5, 10, 16],
          [25, 63, 62, 59, 55, 54, 48, 40],
          [26, 63, 61, 58, 54, 50, 45, 40, 34, 28, 21, 14, 7, 0],
          [27, 0, 8, 16, 24, 32, 40, 48, 56],
          [28, 0, 1, 3, 6, 10, 14, 19, 24, 30, 36, 43, 50, 57]]
    df_quiet = df.copy()

    for channels in MR:
        df_quiet = df_quiet[~((df_quiet["chbid"].isin(range(7,Nchb+1)))
                            & (df_quiet["chipid"] == channels[:][0]) 
                            & (df_quiet["hardid"].isin(channels[1:])))]
    return df_quiet


def calo_time_wins(df, thresh = 5):
    """ Generate a list of time windows for relevant off-trigger incidents (CaloEvents).
    
    Parameters:
    ----------
        df : a pandas DataFrame of a single run. Preferable after cleaning noisy channels.
        threshold : integer threshold for minimum number of hits in the calorimeter per clock-cycle.
    
    Return:
    ------
        time_wins: a dictionary of time windows per event. Keys are the event ids.
        
    Note: the time windows are specified by the first clock-cycle of the window,
          which is the biggest dt value.
    
    """
    
    # Create a list of stable events
    stable_events = df.groupby("eventId").count().xpos < 1000
    stable_events = stable_events[stable_events].index.tolist()
    df_stable = df[df.eventId.isin(stable_events)]
    
    # Create hit counts per clock-cycle
    dt_dist = df_stable.groupby(['eventId', 'dt']).count()
    dt_dist.reset_index(inplace=True)

    time_wins = {}
    
    # Add lists of relevant time windows event-by-event
    for i in stable_events:
        dt_list = dt_dist.dt[(dt_dist.eventId == i) & (dt_dist.xpos >= thresh)].tolist()
        for t in dt_list:
            
            # Remove possible gitter of +/- clock-cycle or long signals double counts
            if (t+1 in dt_list) or (t+6 in dt_list):
                dt_list.remove(t)

        time_wins[i] = dt_list.copy()

    return time_wins


def read_nov18run(run, offset={'x': 18, 'y': 17}, Nchb=8):
    """Reading run file from Nov. '18 test beam at PS.

    Parameters:
    ----------
        run :   dictionary of run ID. Format: {"day": ddmmyyyy, "time": HHMM, "index": integer}.
        offset: dictionary of coordinate offset. Default: {'x': 18, 'y': 17}.
        Nchb :  the number of chambers in the setup. Default: 8.
    
    Return:
    ------
        df_quiet : Pandas DataFrame
    
    """
    file1 = open("path.txt","r")
    HOME = file1.read()

    day = format(run["day"], '08d')
    time = format(run["time"], '04d')
    index = run["index"]
    filename = HOME + "vecsimple_acq_MR1_{}_{}_calo_{}_MGT.root".format(day, time, index)
    runId = '{:08d}-{:04d}-{}'.format(run["day"], run["time"], run["index"])

    # reading file into dataframe
    print('reading {}'.format(filename))
    df = readRootFile(filename)
    df['eventId'] = df['eventId'].agg(lambda x: '{}_{}'.format(runId, x))

    # correct offset
    print('fixing off set.')
    fixoffset(df, offset['x'], offset['y'])
    
    # Filter noisy channels
    print('filter noisy channels.')
    df_quiet = filterNoisyChannels(df)
    
    return df_quiet


def cleaning(df, mode, time_wins=[15]):
    """Basic event selection.    
    Based on the following criteria:
        - stable events.
        - hits in trigger-time correlated time-window.
        - have either 1 or 2-adjacent hits in teh first layer
    
    Parameters:
    -----------
        df       :  pandas DataFrame
        mode     :  'dt', 'calo', or 'MC' for trigger-time correlated hits, CaloEvents, or simulation, respectively.
        timewins :  list of relevant time windows

    Returns:
    --------
        df_batch :  pandas DataFrame containing the summary of number of hits per event, per chamber
        nEvents  :  total number of stable events
    
    """
    if mode != 'MC':
        # Select only events with less then 1000 hits
        stable_events = df.groupby("eventId").count().xpos < 1000
        stable_events = stable_events[stable_events].index.tolist()

        df_stable = df[df.eventId.isin(stable_events)]
        print("run {}: {:%} stable out of {} events.".format(stable_events[0][:15],
                                                        len(stable_events)/len(df.groupby("eventId").count()),
                                                        len(df.groupby("eventId").count())))
        
        if mode == "calo":
            df_stable['id'] = df_stable[['eventId', 'dt']].apply(tuple, axis=1)

            # Creating Calo-Event ID to destinguish from the eventId
            df_stable['caloId'] = df_stable.id.agg(lambda x: ((x[1]-1 in time_wins[x[0]])  * '{}_{}'.format(x[0], x[1]-1)) or 
                                                            ((x[1] in time_wins[x[0]])  * '{}_{}'.format(x[0], x[1])) or 
                                                            ((x[1]+1 in time_wins[x[0]]) * '{}_{}'.format(x[0], x[1]+1)) or
                                                            ((x[1]+2 in time_wins[x[0]]) * '{}_{}'.format(x[0], x[1]+2)))
            # keep only data in time windows
            df_dt = df_stable[df_stable.caloId!='']
        
            dataId = 'caloId'
        
        else:
            df_dt = df_stable[df_stable.dt.isin(list(range(time_wins[0]-3,time_wins[0]+3)))]
            dataId = 'eventId'

        nEvents = len(df_dt[dataId].unique())
        print('{} events with hits in dt'.format(nEvents))
    
    else:
        df_dt = df.copy()
        dataId = 'eventId'
        nEvents = len(df_dt[dataId].unique())

    # Select only events some hits in the first layer
    nhits_layer = df_dt.groupby([dataId, 'chbid']).count()
    nhits_layer = nhits_layer.reset_index(level=1)
    print ('{:.2%} of events have some hits in the first layer'.format(nhits_layer[nhits_layer.chbid == 1].shape[0]/nEvents))
    
    # Select only events with 1 hit  in the first layer
    singleHit_layer1 = nhits_layer[(nhits_layer.chbid == 1) & (nhits_layer.xpos == 1)].index.tolist()
    print ('{:.2%} of events have a single hit in the first layer'.format(len(singleHit_layer1)/nEvents))
    
    # Array of events with 2 hits in the first layer
    doubleHit_layer1 = nhits_layer[(nhits_layer.chbid == 1) & (nhits_layer.xpos == 2)].index.tolist()
    df_doubleHit_layer1 = df_dt[(df_dt[dataId].isin(doubleHit_layer1)) &
                                (df_dt.chbid == 1)].groupby(dataId).agg(lambda x: x.tolist())
    
    df_doubleHit_layer1['pos'] = list(zip(df_doubleHit_layer1.xpos, df_doubleHit_layer1.ypos))
    adj_2hits_layer1 = df_doubleHit_layer1[df_doubleHit_layer1.pos.\
                                           apply(lambda x:np.sqrt((x[0][0]-x[0][1])**2 +
                                                                  (x[1][0]-x[1][1])**2) <= 1)].index.tolist()
    print ('{:.2%} of events have a single or 2 adjacent hits in the first layer'.format((len(singleHit_layer1)+
                                                                           len(adj_2hits_layer1))/nEvents))
    
    
    # pandas DataFrame containing the summary of number of hits per event, per chamber
    df_batch = df_dt[(df_dt[dataId].isin(doubleHit_layer1))].groupby([dataId, 'chbid']).agg(lambda x: x.tolist()).reset_index()
    # if df_batch.shape[0] == 0:
    #     return df_batch 0
    df_batch = df_batch[df_batch[dataId].isin(singleHit_layer1 + adj_2hits_layer1)]
    df_batch['nhits'] = df_batch.applymap(lambda x: len(str(x).split(',')))['xpos'] # xpos is selected just to extract the number of hits

    return df_batch, nEvents


def isMIP(df_batch, mode, Nchb=8, res=1):
    """Returns the list of MIP events using strict selection for detection efficiency estimation
    Parameters:
    -----------
        df_batch :  pandas DataFrame containing the summary of number of hits per event, per chamber
        mode     :  'dt', 'calo', or 'MC' for trigger-time correlated hits, CaloEvents, or simulation, respectively.
        Nchb     :  the number of chambers in the setup. Default: 8.
        Res      :  residual threshold for hits from the reconstructed truck

    Returns: 
    --------
        tuple of    :   - list of valid MIP events for efficiency estimation (tofit_xyz[tofit_xyz.inbound].index.tolist())
                        - histogram values of residual in x (hresx)
                        - histogram values of residual in y (hresy)
                        - arary of residual histograms edges (same for x and y)    
    """

    if mode == "calo":
        dataId = 'caloId'
    else:
        dataId = 'eventId'

    # up to two adjecent hits in each layer
    df_mip = df_batch[df_batch.nhits <= 2]
    # finding layers with non-adjecent two hits
    no_mip = df_mip[df_mip.nhits == 2]
    no_mip['pos'] = list(zip(no_mip.xpos, no_mip.ypos))
    no_mip = no_mip[dataId][no_mip.pos.apply(lambda x:np.sqrt((x[0][0]-x[0][1])**2 +
                                                    (x[1][0]-x[1][1])**2) > 1)].tolist()
    df_mip = df_mip[~df_mip[dataId].isin(no_mip)]
    
    # preparing for fit
    tofit = df_mip.groupby(dataId).agg(lambda x: x.tolist())[['chbid', 'xpos', 'ypos', 'nhits']]
    tofit_xyz = tofit[['xpos', 'ypos']].applymap(lambda x: np.concatenate(np.array(x)).ravel())
    tofit_z = pd.DataFrame({'z':list(zip(tofit.chbid, tofit.nhits)), dataId: tofit.index.tolist()})
    tofit_z.set_index(dataId, inplace=True)
    tofit_xyz['chbid'] = tofit_z['z'].agg(lambda x: np.concatenate(np.array([[x[0][i]]*x[1][i]\
                                         for i in range(len(x[0]))])).ravel())
    
    # Fitting track
    fit = pd.DataFrame({'zx': tofit_xyz[['chbid', 'xpos']].apply(tuple, axis=1),
                    'zy': tofit_xyz[['chbid', 'ypos']].apply(tuple, axis=1)})
    tofit_xyz[['zx', 'zy']] = fit.applymap(lambda p: np.polyfit(p[0], p[1], 1))
    
    # Calculating residuals
    residuals = pd.DataFrame({'x': tofit_xyz[['xpos', 'zx', 'chbid']].apply(tuple, axis=1),
                              'y': tofit_xyz[['ypos', 'zy', 'chbid']].apply(tuple, axis=1)})
    residuals['xres'] = residuals.x.agg(lambda x: np.abs(x[0] - (x[1][0]*x[2] + x[1][1])))
    residuals['yres'] = residuals.y.agg(lambda x: np.abs(x[0] - (x[1][0]*x[2] + x[1][1])))
    
    # A MIP should have residual < 1 
    tofit_xyz['inbound'] = residuals[['xres', 'yres']].apply(tuple, axis=1).\
                           agg(lambda x: sum(x[0] > res)+sum(x[1] > res) == 0 )
    
    l = []
    for i in residuals.xres.values.flatten():
        l += list(i)
    hresx, edgex = np.histogram(l,range=(0,10), bins=10)
    l = []
    for i in residuals.yres.values.flatten():
        l += list(i)
    hresy, edgey = np.histogram(l,range=(0,10), bins=10)

    del(no_mip)
    del(tofit)
    
    del(tofit_z)
    del(df_mip)
    del(fit)
    
    return (tofit_xyz[tofit_xyz.inbound].index.tolist(), hresx, hresy, edgey)


def efficiency_estimation(df_mips, mode, Nchb=8):
    """MIP detection estimation for 
    Parameters:
    -----------
        df_mips  :  pandas DataFrame containing a list of valid MIP events for efficiency estimation
        mode     :  'dt', 'calo', or 'MC' for trigger-time correlated hits, CaloEvents, or simulation, respectively.
        Nchb     :  the number of chambers in the setup. Default: 8.

    Returns: 
    --------
        eff      :  dictionary of MIP detection efficiency of each chamber (key)
        pool     :  dictionary of reference MIP tracks for detection efficiency estimation
                    of each chamber (key)
    """
    if mode == "calo":
        dataId = 'caloId'
    else:
        dataId = 'eventId'

    eff = {}
    pool = {}
    mult = {}
    mip_members = df_mips.groupby(dataId).agg(lambda x: x.tolist())['chbid']
    counter = mip_members
    if Nchb == 11:
        chbl = [1,2,3,4,5,6,7,8,9,10,11]
    else:
        chbl = [1,2,3,4,5,6,7,8]

    # Selecting the tracks were all chambers are efficient
    eff_tracks = mip_members[mip_members.agg(lambda x: set(chbl).issubset(list(x)))]
    neff_tracks = eff_tracks.shape[0]
    
    # Number of hits in each chamber
    mult_in_track = eff_tracks.agg(lambda x: Counter(x))

    for chb in range(2, Nchb+1):
        mult[chb] = mult_in_track.agg(lambda x: x[chb]).sum() / neff_tracks
    
        s = chbl[:]
        s.remove(chb)

        # Selecting the tracks were all chambers (excluding the tested chamber) are efficient
        chb_pool = mip_members[mip_members.agg(lambda x: set(s).issubset(list(x)))].shape[0]
        eff[chb] = neff_tracks/chb_pool
        pool[chb] = chb_pool

    return eff, pool, mult