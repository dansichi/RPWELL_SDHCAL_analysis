import uproot
import pandas as pd
from itertools import chain, product
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import csv 

from src.mip_eff.cluster import cluster
from scipy.spatial.distance import cdist 

import os
import errno

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
    # file1 = open("path.txt","r")
    # HOME = file1.read()
    HOME = '/Users/dansh/cernbox/TB_nov18/Rootfiles/simple/'
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
    toflip = ['12112018-0335-1',
              '12112018-0431-1',
              '12112018-0204-2',
              '12112018-0221-1',
              '12112018-0223-1',
              '12112018-0230-1',
              '12112018-0241-1',
              '12112018-0253-1',
              '12112018-0058-1',
              '12112018-0108-1',
              '12112018-0118-1',
              '11112018-2248-1',
              '11112018-2309-1',
              '12112018-0729-1',
              '12112018-0703-1',
              '12112018-0715-1',
              '12112018-0718-1']
    if runId in toflip:
        df.loc[df.chbid < 4, 'xpos'] = 48 - df.loc[df.chbid < 4, 'xpos']
    fixoffset(df, offset['x'], offset['y'])
    
    # Filter noisy channels
    print('filter noisy channels.')
    df_quiet = filterNoisyChannels(df)
    
    return df_quiet


def cleaning(df, mode, time_wins=[15], verbose=True):
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
        if verbose: print("run {}: {:%} stable out of {} events.".format(stable_events[0][:15],
                                                        len(stable_events)/len(df.groupby("eventId").count()),
                                                        len(df.groupby("eventId").count())))
        
        if mode == "calo":
            df_stable.loc[:, 'id'] = df_stable[['eventId', 'dt']].apply(tuple, axis=1)

            # Creating Calo-Event ID to destinguish from the eventId
            df_stable.loc[:, 'caloId'] = df_stable.id.agg(lambda x: ((x[1]-1 in time_wins[x[0]])  * '{}_{}'.format(x[0], x[1]-1)) or 
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
        if verbose: print('{} events with hits in dt'.format(nEvents))
    
    else:
        df_dt = df.copy()
        dataId = 'eventId'
        nEvents = len(df_dt[dataId].unique())

    # Select only events some hits in the first layer
    nhits_layer = df_dt.groupby([dataId, 'chbid']).count()
    nhits_layer = nhits_layer.reset_index(level=1)
    if verbose: print ('{:.2%} of events have some hits in the first layer'.format(nhits_layer[nhits_layer.chbid == 1].shape[0]/nEvents))
    
    # Select only events with 1 hit  in the first layer
    singleHit_layer1 = nhits_layer[(nhits_layer.chbid == 1) & (nhits_layer.xpos == 1)].index.tolist()
    if verbose: print ('{:.2%} of events have a single hit in the first layer'.format(len(singleHit_layer1)/nEvents))
    
    # Array of events with 2 hits in the first layer
    doubleHit_layer1 = nhits_layer[(nhits_layer.chbid == 1) & (nhits_layer.xpos == 2)].index.tolist()
    df_doubleHit_layer1 = df_dt[(df_dt[dataId].isin(doubleHit_layer1)) &
                                (df_dt.chbid == 1)].groupby(dataId).agg(lambda x: x.tolist())
    
    df_doubleHit_layer1['pos'] = list(zip(df_doubleHit_layer1.xpos, df_doubleHit_layer1.ypos))
    adj_2hits_layer1 = df_doubleHit_layer1[df_doubleHit_layer1.pos.\
                                           apply(lambda x:np.sqrt((x[0][0]-x[0][1])**2 +
                                                                  (x[1][0]-x[1][1])**2) <= 1)].index.tolist()
    if verbose: print ('{:.2%} of events have a single or 2 adjacent hits in the first layer'.format((len(singleHit_layer1)+
                                                                           len(adj_2hits_layer1))/nEvents))
    
    
    # pandas DataFrame containing the summary of number of hits per event, per chamber
    df_batch = df_dt[(df_dt[dataId].isin(singleHit_layer1 + adj_2hits_layer1))].groupby([dataId, 'chbid']).agg(lambda x: x.tolist()).reset_index()
    df_batch['nhits'] = df_batch.xpos.agg(lambda x: len(x)) 
    return df_batch, df_batch.shape[0]


def isMIP(df_batch, mode, Nchb=8, res=1, verbose=True, maxnhits=2, tolerance=2):
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

    # up to two adjacent hits in each layer
    # df_mip = df_batch[df_batch.nhits <= 2]
    df_mip = df_batch[df_batch.nhits <= maxnhits]

    
    # finding layers with non-adjacent two hits
    # no_mip = df_mip[df_mip.nhits == 2]
    no_mip = df_mip[df_mip.nhits > 1]

    no_mip.loc[:, 'pos'] = no_mip[['xpos', 'ypos']].apply(tuple, axis=1).apply(lambda x: np.array([np.array(x[0]),np.array(x[1])]).transpose())
    # no_mip.loc[:, 'pos'] = no_mip['pos'].agg(lambda x: np.array(x).transpose())
    # no_mip = no_mip[dataId][no_mip.pos.apply(lambda x:np.sqrt((x[0][0]-x[0][1])**2 +
    #                                                 (x[1][0]-x[1][1])**2) > 1)].tolist()

    # CORRECTION 15-09-2020 : the distance calculation was wrong
    no_mip = no_mip[[dataId, 'chbid']][no_mip.pos.apply(lambda x: cdist([x[0]], [x[1]], 'euclidean')[0][0] > 2)].agg(tuple, axis=1)
    
    df_mip = df_mip[~df_mip.index.isin(no_mip.index.tolist())]
    
    # UPDATE 06-01-2020: keep only tracks with least Nchb-1 MIP-like layers
    df_mip_evt = df_mip.groupby(dataId).agg(lambda x: len(set(x)))
    valid_trks = df_mip_evt[df_mip_evt['chbid'] > (Nchb - tolerance)].index.tolist()
    df_mip = df_mip[df_mip[dataId].isin(valid_trks)]    
    if verbose: print('{:.2%} valid tracks'.format(len(valid_trks)/len(df_batch[dataId].unique().tolist())))
    
    # preparing for fit 
    tofit = df_mip.groupby(dataId).agg(lambda x: x.tolist())[['chbid', 'xpos', 'ypos', 'nhits']]
    tofit_xyz = tofit.loc[:, ['xpos', 'ypos']].applymap(lambda x: np.concatenate(np.array(x)).ravel())
    tofit_z = pd.DataFrame({'z':list(zip(tofit.chbid, tofit.nhits)), dataId: tofit.index.tolist()})
    tofit_z.set_index(dataId, inplace=True)
    tofit_xyz.loc[:, 'chbid'] = tofit_z['z'].agg(lambda x: np.concatenate(np.array([[x[0][i]]*x[1][i]\
                                         for i in range(len(x[0]))])).ravel())
    
    # Fitting track
    fit = pd.DataFrame({'zx': tofit_xyz[['chbid', 'xpos']].apply(tuple, axis=1),
                    'zy': tofit_xyz[['chbid', 'ypos']].apply(tuple, axis=1)})
    tofit_xyz[['zx', 'zy']] = fit.applymap(lambda p: np.polyfit(p[0], p[1], 1))

    # Calculating residuals
    residuals = pd.DataFrame({'x': tofit_xyz[['xpos', 'zx', 'chbid']].apply(tuple, axis=1),
                              'y': tofit_xyz[['ypos', 'zy', 'chbid']].apply(tuple, axis=1)})
    residuals.loc[:, 'xres'] = residuals.x.agg(lambda x: np.abs(x[0] - (x[1][0]*x[2] + x[1][1])))
    residuals.loc[:, 'yres'] = residuals.y.agg(lambda x: np.abs(x[0] - (x[1][0]*x[2] + x[1][1])))
    
    # A MIP should have residual < 1 
    tofit_xyz['inbound'] = residuals[['xres', 'yres']].apply(tuple, axis=1).\
                           agg(lambda x: sum(x[0] > res)+sum(x[1] > res) == 0 )

    if verbose: print('{:.2%} inbound tracks out of clean events'.format(len(tofit_xyz[tofit_xyz.inbound].index.tolist())/len(df_batch[dataId].unique().tolist()), len(df_batch.index.tolist())))

    df_batch.loc[:, 'zx0'] = ""
    df_batch.loc[:, 'zy0'] = ""
    df_batch.loc[:, 'zx1'] = ""
    df_batch.loc[:, 'zy1'] = ""

    # saving the fit parameters
    for i in tofit_xyz.index.tolist():
        
        df_batch.loc[df_batch[dataId] == i, ['zx0']] = tofit_xyz['zx'][i][0]
        df_batch.loc[df_batch[dataId] == i, ['zx1']] = tofit_xyz['zx'][i][1]
        df_batch.loc[df_batch[dataId] == i, ['zy0']] = tofit_xyz['zy'][i][0]
        df_batch.loc[df_batch[dataId] == i, ['zy1']] = tofit_xyz['zy'][i][1]

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


def efficiency_estimation(df_mips, mode, Nchb=8, toPrint=True, verbose=True):
    """MIP detection estimation for 
    Parameters:
    -----------
        df_mips  :  pandas DataFrame containing a list of valid MIP events for efficiency estimation.
                    Note: A valid MIP event can have only one layer with non-MIP characteristics 
                    (i.e. not hits, 2 sparse hits, or more than 2 hits)
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
    df_mips['cl_members'] = ""
    # Filtering the hits in each layer to contain only relevant cluster hits
    for i in tqdm(range(df_mips.shape[0])):
        
        p = [df_mips.zx0.iloc[i]*df_mips.chbid.iloc[i] + df_mips.zx1.iloc[i],
            df_mips.zy0.iloc[i]*df_mips.chbid.iloc[i] + df_mips.zy1.iloc[i]]

        if (df_mips.zx0.iloc[i] == "") or (df_mips.zx1.iloc[i] == ""):
            df_mips['cl_members'].iloc[i] == []
            continue
        # print('chamber: {}'.format(df_mips.chbid.iloc[i]))
        # print("x: {}".format(df_mips.xpos.iloc[i]))
        # print("y: {}".format(df_mips.ypos.iloc[i]))
        cl = cluster(df_mips.xpos.iloc[i], df_mips.ypos.iloc[i]) 
        # cl.seeding(p, 5)
        
        df_mips['cl_members'].iloc[i] = cl.cluster(p, radius=2)
    # print( df_mips['cl_members'])
    
    # print('Exporting empty mip-clusters')
    # # df_mips[df_mips.cl_members == []].to_csv('empty_clusters_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M')))
    # DEBUG exporting mips with clusters information
    # df_mips.to_csv('mips_with_clusters_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M')))

    mip_members = df_mips.groupby(dataId).agg(lambda x: x.tolist())['chbid']
    chbl = list(range(1,Nchb+1))

    # Selecting the tracks were all chambers are efficient
    eff_tracks = mip_members[mip_members.agg(lambda x: set(chbl).issubset(list(x)))]

    # Tracks that has one chamber with more than 2 hits
    eff_tracks_exceptional = df_mips[df_mips.nhits > 2][[dataId,'chbid']]
    
    # ### DEBUG
    # df_mips.to_csv('debug_df_mips.csv', index=False)
    # ### End DEBUG


    # Tracks that has 2 sparse hits
    sparse_hits = df_mips[(df_mips.nhits == 2)]
    sparse_hits['pos'] = list(zip(sparse_hits.xpos, sparse_hits.ypos))
    sparse_hits = sparse_hits[sparse_hits.pos.apply(lambda x:
                                      np.sqrt((x[0][0]-x[0][1])**2 +
                                              (x[1][0]-x[1][1])**2) > 1)][[dataId,'chbid']]


    # Number of hits in each chamber
    # mult_in_track = eff_tracks.agg(lambda x: Counter(x))
    if toPrint:
        if not os.path.isdir('./figures'):
            os.mkdir("./figures")

        if not os.path.isdir('./results'):
            os.mkdir("./results")

        resultsfile = uproot.recreate('./results/multiplicity_{}layers_{}_{}.root'\
                            .format(Nchb, mode, datetime.now().strftime('%Y%m%d_%H%M')),
                            compression=uproot.ZLIB(4))
        
        file = uproot.recreate('./results/pad_multiplicity_{}layers_{}_{}.root'.format(Nchb, mode, datetime.now().strftime('%Y%m%d_%H%M')), compression=uproot.ZLIB(4))

    for chb in range(2, Nchb+1):
        s = chbl[:]
        s.remove(chb)
        # Selecting the tracks where all chambers (excluding the tested chamber) are efficient
        chb_pool = mip_members[mip_members.agg(lambda x: set(s).issubset(list(x)))].index.to_list()

        # list of relevant pool events
        pool_tracks =  df_mips[(df_mips[dataId].isin(chb_pool)) &
                               (~df_mips[dataId].isin(eff_tracks_exceptional[dataId][eff_tracks_exceptional.chbid != chb].tolist())) &
                               (~df_mips[dataId].isin(sparse_hits[dataId][sparse_hits.chbid != chb].tolist())) ]
        # df_mips = df_mips.groupby(dataId)

        # multiplicity in chamber for relevant tracks
        mult_in_track = pool_tracks['cl_members'][(pool_tracks.chbid == chb)].agg(lambda x: len(x))
        
        # DEBUG:
        pool_tracks.to_pickle('pool_trackschb{}_{}.pkl'.format(chb, datetime.now().strftime('%Y%m%d_%H%M')))
        print('saving pool_trackschb{}_{}.pkl'.format(chb, datetime.now().strftime('%Y%m%d_%H%M')))


        ###

        mult_in_track = mult_in_track[mult_in_track > 0]
        
        

        if toPrint: resultsfile['hmult_chb{}'.format(chb)] = np.histogram(mult_in_track, bins=range(max(mult_in_track)+2))
        neff_tracks = mult_in_track.shape[0]
        if toPrint:
            fig = plt.figure()
            ValList, bins, _ = plt.hist(mult_in_track)
            plt.title("chb {}: mult mean = {}; mult std={}".format(chb, mult_in_track.mean(), mult_in_track.std()))
            plt.savefig('figures/hist_mip_mult_{}layers_chb{}_{}_{}.png'.format(Nchb, chb, mode, datetime.now().strftime('%Y%m%d_%H%M')))
            del(fig)

        
        mult[chb] = mult_in_track.mean()
        if verbose: print("chb {}: mult mean = {}; mult std={}".format(chb, mult_in_track.mean(), mult_in_track.std()))
        
        eff[chb] = neff_tracks/pool_tracks[dataId].unique().shape[0]

        pool[chb] = pool_tracks[dataId].unique().shape[0]
    
        bins = mult_in_track.max()
        if toPrint: file["mult_chb{}".format(chb)] = np.histogram(mult_in_track, bins, range=(0, bins))

    return eff, pool, mult


def exporter(eff, pool, mult, mode, Nchb):
    
    # 48x48 cm^2 RPWELL chambers
    if Nchb == 11:
        dict_RPWELL = {7: 'ASU 61', 8: 'ASU 60', 9: 'ASU 51', 10: 'ASU 57', 11: 'ASU 52'}
    elif Nchb in [7, 8]:
        dict_RPWELL = {7: 'ASU 61', 8: 'ASU 51'}

    # saving dato to csv
    if not os.path.isdir('./results'):
        os.mkdir("./results")

    with open('./results/mip_performance_{}layers_{}_{}.csv'.format(Nchb, mode, datetime.now().strftime('%Y%m%d_%H%M')), mode='w') as exportfile:
        
        exporter = csv.writer(exportfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exporter.writerow(['chamber', 'type', 'efficiency', 'pool of tracks', 'average_multiplicity'])
        
        for i in range(2, Nchb+1):
            if i in range(2, 4):
                exporter.writerow([i , 'Small MM', eff[i], pool[i], mult[i]])
            elif i in range(4, 7):
                exporter.writerow([i , 'Large MM', eff[i], pool[i], mult[i]])
            else:
                exporter.writerow([i , dict_RPWELL[i], eff[i], pool[i], mult[i]])
