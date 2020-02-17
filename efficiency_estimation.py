import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm 
import src.mip_eff.tb_data_handling as dh
import src.mip_eff.simulation_analysis as mc
import argparse
from datetime import datetime
import os
import warnings
import csv

def eff(mode, Nchb=8, qc=False):
    """ Estimates the MIP detection efficiency for CaloEvents data.
    Parameters:
    -----------
        mode    :   'dt' or 'calo' for trigger-time correlated hits or CaloEvents, respectively.
        Nchb    :   the number of chambers in the setup. Default: 8.
    
    Returns:
    --------

        eff_tot :    a dictionary of the efficiency value of the different
                    chambers. The chamber numbers serves as keys.  
        pool_tot :   a dictionary of the number of MIP tracks used as reference
                    for the efficiency estimation. The chamber numbers serves as keys.
    """

    if mode == 'calo':
        print('Running efficiency estimation for hits in CaloEvents.')
        dataId = 'caloId'
        df_MIPs = pd.DataFrame(columns = ['eventId', 'chbid', 'xpos', 'ypos', 'dt', 'digit', 'chipid', 'hardid',
        'nhits', 'caloId'])
    else:
        print('Running efficiency estimation for hits which are time-correlated with the trigger.')
        dataId = 'eventId'
        df_MIPs = pd.DataFrame(columns = ['eventId', 'chbid', 'xpos', 'ypos', 'dt', 'digit', 'chipid', 'hardid',
        'nhits'])

    if Nchb == 7:
        runList = pd.read_csv('data/eScan8Layers.csv')
    else:
        runList = pd.read_csv('data/eScan{}Layers.csv'.format(Nchb))
    
    l = runList.shape[0]

    # for selection quality control purpose
    if qc:
        eff_runs = {'run': [], 'eff': [], 'pool': []}

    # Define list of excluded runs
    if Nchb == 8 or Nchb == 7:
        excluded_runs = ['0036', '0650', '0712']
    elif Nchb == 11:
        excluded_runs = ['0018', '0726']
    else:
        excluded_runs = []

    for i in tqdm(range(l)):
        run = runList.iloc[i]

        if format(run["time"], '04d') not in excluded_runs:
            runId = '{:08d}-{:04d}-{}'.format(run["day"], run["time"], run["index"])
            print(runId)
            df = dh.read_nov18run(run, Nchb=Nchb)
            df = df[df.chbid <= Nchb]

            if mode == 'calo':
                time_wins = dh.calo_time_wins(df)
            else:
                time_wins = [15]

            df_batch, nEvents = dh.cleaning(df, mode, time_wins)

            if nEvents == 0:
                print('Zero nEvents after cleaning!')
                continue
                
            eff_events = df_batch.groupby(dataId)['chbid'].agg(lambda x: len(x) > Nchb-2)
            eff_events = eff_events[eff_events].index.tolist()
            df_batch_eff = df_batch[df_batch[dataId].isin(eff_events)]
            
            # for selection quality control purpose
            if qc:
                hresx_tot = [0 for i in range(10)]
                hresy_tot = [0 for i in range(10)]
            
            if df_batch_eff.shape[0] > 0:
                (effective_MIPs, hresx, hresy, edge) = dh.isMIP(df_batch_eff, mode, Nchb=Nchb, res=2)

                print ('{:.2%} of events are valid MIPs'.format(len(effective_MIPs)/nEvents))
                df_batch = df_batch[df_batch[dataId].isin(effective_MIPs)]
                
                df_batch[dataId] = df_batch[dataId].agg(lambda x: '{}_{}'.format(runId, x))
                df_MIPs = pd.concat([df_MIPs,df_batch], axis=0)

                # for selection quality control purpose
                if qc:
                    hresx_tot += hresx
                    hresy_tot += hresy
                    eff_run, pool_run, mult_run = dh.efficiency_estimation(df_batch)
                    eff_runs['run'].append(runId)
                    eff_runs['eff'].append(eff_run)
                    eff_runs['pool'].append(pool_run)

    eff_tot, pool_tot, mult_tot = dh.efficiency_estimation(df_MIPs, mode, Nchb)

    dh.exporter(eff_tot, pool_tot, mult_tot, Nchb)

    return eff_tot, pool_tot, mult_tot


def eff_MC(energy, mipeff, particle='pions', Nchb=8):
    """ Estimates the MIP detection efficiency for CaloEvents data.
    Parameters:
    -----------
        energy  :   beam energy in GeV (1-6).
        mipeff  :   simulation mip threshold.
        particle:   beam's type of particle: 'pions' or 'electrons'. Default: 'pions'.
        mode    :   'dt' or 'calo' for trigger-time correlated hits or CaloEvents, respectively.
        Nchb    :   the number of chambers in the setup. Default: 8.
    
    Returns:
    --------

        eff_tot :    a dictionary of the efficiency value of the different
                    chambers. The chamber numbers serves as keys.  
        pool_tot :   a dictionary of the number of MIP tracks used as reference
                    for the efficiency estimation. The chamber numbers serves as keys.
    """

    filename, digit = mc.MAX_FILES[particle][energy][mipeff]
    dataId = 'eventId'
  
    # Define list of excluded runs
    df = mc.readGeantFile(filename, digit)

    df_batch, nEvents = dh.cleaning(df, 'MC')

    if nEvents == 0:
        raise ValueError('nEvent is zero!')
        return
        
    eff_events = df_batch.groupby(dataId)['chbid'].agg(lambda x: len(x) > Nchb-2)
    eff_events = eff_events[eff_events].index.tolist()
    df_batch_eff = df_batch[df_batch[dataId].isin(eff_events)]
    
    if df_batch_eff.shape[0] > 0:
        (effective_MIPs, hresx, hresy, edge) = dh.isMIP(df_batch_eff, mode, Nchb=Nchb, res=2)

        print ('{:.2%} of events are valid MIPs'.format(len(effective_MIPs)/nEvents))
        df_batch = df_batch[df_batch[dataId].isin(effective_MIPs)]
        df_MIPs = df_batch

    eff_tot, pool_tot, mult_tot = dh.efficiency_estimation(df_MIPs, 'MC', Nchb)
    return eff_tot, pool_tot, mult_tot


def plot_eff(eff_tot, pool_tot, mode, Nchb=8):
    """Generates plots for efficiency estimation and save them.
    Parameters:
    -----------
        eff_tot :    a dictionary of the efficiency value of the different
                    chambers. The chamber numbers serve as keys.  
        pool_tot :   a dictionary of the number of MIP tracks used as reference
                    for the efficiency estimation. The chamber numbers serve as keys.
        mode     :  'dt', 'calo', or 'MC' for trigger-time correlated hits, CaloEvents, or simulation, respectively.
        Nchb :      the number of chambers in the setup. Default: 8.
    
    Return:
    -------
        None
    
    """
    fig = plt.figure(figsize=(15, 20), constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=1, left=0.05, right=0.48, wspace=0.05)
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[-1, :])
    
    # 16x16 cm^2 uM chambers
    yerr = [1/np.sqrt(pool_tot[i]) for i in range(2, 4)]
    ax1.errorbar(range(2, 4), [eff_tot[i] for i in range(2, 4)], yerr=yerr, fmt='o',
                label='Small MM')
    
    # 48x48 cm^2 uM chambers
    yerr = [1/np.sqrt(pool_tot[i]) for i in range(4, 7)]
    ax1.errorbar(range(4, 7), [eff_tot[i] for i in range(4, 7)], yerr=yerr, fmt='s',
                label='large MM')
    yerr = [1/np.sqrt(pool_tot[i]) for i in range(7, Nchb+1)]
    
    # 48x48 cm^2 RPWELL chambers
    if Nchb == 11:
        dict_RPWELL = {7: 'ASU 61', 8: 'ASU 60', 9: 'ASU 51', 10: 'ASU 57', 11: 'ASU 52'}
    elif Nchb == 8:
        dict_RPWELL = {7: 'ASU 61', 8: 'ASU 51'}
    
    # saving dato to csv
    if not os.path.isdir('./results'):
        os.mkdir("./results")

    ax1.errorbar(range(7, Nchb+1), [eff_tot[i] for i in range(7, Nchb+1)], yerr=yerr, fmt='^',
                label='RPWELL')
    
    ax1.set_xlabel('layer number')
    ax1.set_ylim(0.4,1)
    ax1.legend()
    ax1.set_ylabel('MIP detection efficiency')

    # Add a table at the bottom of the axes
    # ax2.axis('tight')
    rowlabels = ['Small MM 2',
                 'Small MM 3',
                 'Large MM 1',
                 'Large MM 2',
                 'Large MM 3']
    ax2.axis('off')
    cell_text = []
    for i in range(7, Nchb+1):
        rowlabels.append('RPWELL {}'.format(dict_RPWELL[i]))

    cell_text.append(['{:.2}'.format(eff_tot[i] * 100) for i in range(2, Nchb+1)])
    cell_text.append(['{:.2}'.format(1/np.sqrt(pool_tot[i]) * 100) for i in range(2, Nchb+1)])
    cell_text.append([pool_tot[i] for i in range(2, Nchb+1)])
    table = ax2.table(cellText=np.array(cell_text).T,
                        colLabels=('efficiency[%]', 'error[%]', 'tested tracks'),
                        rowLabels= rowlabels,
                        loc='center')
    table.set_fontsize(18)
    table.auto_set_column_width([0, 1, 2])
    table.scale(10, 3)

    # saving figure
    if not os.path.isdir('./figures'):
        os.mkdir("./figures")
    
    fig.savefig('figures/mip_eff_{}layers_{}_{}.png'.format(Nchb, mode, datetime.now().strftime('%Y%m%d_%H%M')))

    # plt.show()
    del(fig)
    print('| ASU | Effieicny [%] | number of tested tracks|')
    print('|------|----------------|-----------------------|')
    for i in range(7,Nchb+1):
        print("|{} | {:.2}+\-{:.2} \t| {} |".format(dict_RPWELL[i],
                                                  eff_tot[i]*100,
                                                  yerr[i-7]*100,
                                                  pool_tot[i]))


def plot_mult(mult_tot, mode, Nchb=8):
    """Generates plots for efficiency estimation and save them.
    Parameters:
    -----------
        mult_tot :  a dictionary of the pad multiplicity value of the different
                    chambers. The chamber numbers serve as keys.  
        mode     :  'dt', 'calo', or 'MC' for trigger-time correlated hits, CaloEvents, or simulation, respectively.
        Nchb :      the number of chambers in the setup. Default: 8.
    
    Return:
    -------
        None
    
    """

    fig = plt.figure(figsize=(10, 7))
    # 16x16 cm^2 uM chambers
    plt.scatter(range(2, 4), [mult_tot[i] for i in range(2, 4)], 
                label='Small MM')
    
    # 48x48 cm^2 uM chambers
    plt.scatter(range(4, 7), [mult_tot[i] for i in range(4, 7)],
                label='large MM')
    
    # 48x48 cm^2 RPWELL chambers
    if Nchb == 11:
        dict_RPWELL = {7: 'ASU 61', 8: 'ASU 60', 9: 'ASU 51', 10: 'ASU 57', 11: 'ASU 52'}
    elif Nchb == 8:
        dict_RPWELL = {7: 'ASU 61', 8: 'ASU 51'}

    plt.scatter(range(7, Nchb+1), [mult_tot[i] for i in range(7, Nchb+1)],
                label='RPWELL')
    
    plt.xlabel('layer number')
    plt.ylim(0.4,1)
    plt.legend()
    plt.ylabel('MIP detection efficiency')

    # saving figure
    if not os.path.isdir('./figures'):
        os.mkdir("./figures")
    
    fig.savefig('figures/mip_mult_{}layers_{}_{}.png'.format(Nchb, mode, datetime.now().strftime('%Y%m%d_%H%M')))

    # plt.show()
    del(fig)
    print('| \# of Chamber | Multiplicity |')
    print('|------|----------------|-----------------------|')
    for i in range(7,Nchb+1):
        print("|{} | {:.2} \t|".format(i, mult_tot[i]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', help='analysis mode (dt, calo or MC)',
                        default='dt', choices=['dt', 'calo', 'MC'])
    parser.add_argument('-n', '--Nchb', help='number of layers in the setup',
                        default='8', choices=['8', '11', '7'])
    parser.add_argument('-e', '--energy', type=int, help='beam energy in GeV.',
                        choices=range(1,7))
    parser.add_argument('--mipeff', help='mip efficiency')
    parser.add_argument('-p', '--particle', help="The type of the beam's particle.",
                        default='pions', choices=('pions', 'electrons'))
    
    args = parser.parse_args()
    mode = args.mode
    if mode == 'MC':
        energy = '{}GeV'.format(args.energy)
        particle = args.particle
        mipeff = args.mipeff
        if mipeff not in list(mc.MAX_FILES[particle][energy].keys()):
            raise ValueError('Out of range for specific energy: mipeff.')
    Nchb = int(args.Nchb)

    warnings.simplefilter('ignore', np.RankWarning)
    

    if mode not in ['calo', 'dt', 'MC']:
        raise ValueError('Not a valid type of analysis mode.')

    if mode == 'MC':
        eff_tot, pool_tot, mult_tot = eff_MC(energy, mipeff, particle=particle, Nchb=Nchb)
        plot_eff(eff_tot, pool_tot, 'MC', Nchb)
        plot_mult(mult_tot, 'MC', Nchb)
    else:
        eff_tot, pool_tot, mult_tot = eff(mode, Nchb=Nchb)
        plot_eff(eff_tot, pool_tot, mode, Nchb)
        plot_mult(mult_tot, mode, Nchb)