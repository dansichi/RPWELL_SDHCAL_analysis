import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm 
import src.mip_eff.tb_data_handling as dh
import argparse
from datetime import datetime


def eff(mode, Nchb=8, qc=False):
    """ Estimates the MIP detection efficiency for CaloEvents data.
    Parameters:
    -----------

        Nchb :       the number of chambers in the setup. Default: 8.
    
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
    else:
        print('Running efficiency estimation for hits which are time-correlated with the trigger.')
        dataId = 'eventId'

    runList = pd.read_csv('data/eScan{}Layers.csv'.format(Nchb))
    l = runList.shape[0]

    df_MIPs = pd.DataFrame(columns = ['eventId', 'chbid', 'xpos', 'ypos', 'dt', 'digit', 'chipid', 'hardid',
        'nhits'])

    # for selection quality control purpose
    if qc:
        eff_runs = {'run': [], 'eff': [], 'pool': []}

    # Define list of excluded runs
    if Nchb == 8:
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
                    eff_run, pool_run = dh.efficiency_estimation(df_batch)
                    eff_runs['run'].append(runId)
                    eff_runs['eff'].append(eff_run)
                    eff_runs['pool'].append(pool_run)


    eff_tot, pool_tot = dh.efficiency_estimation(df_MIPs, mode, Nchb)
    return eff_tot, pool_tot


def plot_eff(eff_tot, pool_tot, mode, Nchb=8):
    """Generates plots for efficiency estimation and save them.
    Parameters:
    -----------
        eff_tot :    a dictionary of the efficiency value of the different
                    chambers. The chamber numbers serves as keys.  
        pool_tot :   a dictionary of the number of MIP tracks used as reference
                    for the efficiency estimation. The chamber numbers serves as keys.
        mode :      'dt' or 'calo' for trigger-time correlated hits or CaloEvents.
        Nchb :      the number of chambers in the setup. Default: 8.
    
    Return:
    -------
        None
    
    """
    fig = plt.figure(figsize=(15,20), constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=1, left=0.05, right=0.48, wspace=0.05)
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[-1, :])
    
    fig = plt.figure(figsize=(10,7))
    # 16x16 cm^2 uM chambers
    yerr = [1/np.sqrt(pool_tot[i]) for i in range(2,4)]
    ax1.errorbar(range(2,4), [eff_tot[i] for i in range(2,4)], yerr=yerr, fmt='o',
                label='Small MM')
    
    # 48x48 cm^2 uM chambers
    yerr = [1/np.sqrt(pool_tot[i]) for i in range(4,7)]
    ax1.errorbar(range(4,7), [eff_tot[i] for i in range(4,7)], yerr=yerr, fmt='s',
                label='large MM')
    yerr = [1/np.sqrt(pool_tot[i]) for i in range(7,12)]
    
    # 48x48 cm^2 RPWELL chambers
    if Nchb == 11:
        dict_RPWELL = {7: 'ASU 61', 8: 'ASU 60', 9: 'ASU 51', 10: 'ASU 57', 11: 'ASU 52'}
    elif Nchb == 8:
        dict_RPWELL = {7: 'ASU 61', 8: 'ASU 51'}
    
    ax1.errorbar(range(7,Nchb+1), [eff_tot[i] for i in range(7,Nchb+1)], yerr=yerr, fmt='^',
                label='RPWELL')
    
    ax1.xlabel('layer number')
    ax1.ylim(0.4,1)
    ax1.legend()
    ax1.ylabel('MIP detection efficiency')

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

    cell_text.append(['{:.2}'.format(eff_tot[i]*100) for i in range(2,Nchb+1)])
    cell_text.append(['{:.2}'.format(1/np.sqrt(pool_tot[i])*100) for i in range(2,Nchb+1)])
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
    
    plt.savefig('/figures/mip_eff_{}layers_{}_{}.png'.format(Nchb, mode, datetime.now().strftime('%Y%m%d_%H%M')))

    plt.show()

    print('| ASU | Effieicny \[%\] | number of tested tracks|')
    print('|------|----------------|-----------------------|')
    for i in range(7,Nchb+1):
        print("|{} | {:.2}+\-{:.2} \t| {} |".format(dict_RPWELL[i],
                                                  eff_tot[i]*100,
                                                  yerr[i-7]*100,
                                                  pool_tot[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', help='analysis mode (dt or calo)',
                        default='dt', choices=['dt', 'calo'])
    parser.add_argument('-n', '--Nchb', help='number of layers in the setup',
                        default='8', choices=['8', '11'])
    
    args = parser.parse_args()
    mode = args.mode
    Nchb = int(args.Nchb)

    if mode not in ['calo', 'dt']:
        raise ValueError('Not a valid type of analysis mode.')

    eff_tot, pool_tot = eff(mode, Nchb=Nchb)
    plot_eff(eff_tot, pool_tot, mode, Nchb)