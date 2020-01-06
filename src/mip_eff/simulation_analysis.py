import uproot
import pandas as pd
from itertools import chain, product
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

HOME = 'data/Max_20191008/'

# List of simulation files provided by Max. 
# Call with MAX_FILES[particle_type][energy][mip_threshold]
MAX_FILES = {'pions':
                {'5GeV':{ 
                    '0':  ('TB2018_pion_5GeV_0Tesla_Thr_26eV_065_085mip_Mip05kev.root', 1),
                    '02': ('TB2018_pion_5GeV_0Tesla_Thr_02_03_04mip_Mip05kev.root', 1),
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_5GeV_0Tesla_digitised_scream.root', 1),
                    '03': ('TB2018_pion_5GeV_0Tesla_Thr_02_03_04mip_Mip05kev.root', 2),
                    '04': ('TB2018_pion_5GeV_0Tesla_Thr_02_03_04mip_Mip05kev.root', 3),
                    '05': ('TB2018_pion_5GeV_0Tesla_Thr_05_1_2mip_Mip05kev.root', 1),
                    '065': ('TB2018_pion_5GeV_0Tesla_Thr_26eV_065_085mip_Mip05kev.root', 2),
                    '075': ('TB2018_pion_5GeV_0Tesla_Thr_075_125_175mip_Mip05kev.root', 1),
                    '085': ('TB2018_pion_5GeV_0Tesla_Thr_26eV_065_085mip_Mip05kev.root', 3),
                    '1': ('TB2018_pion_5GeV_0Tesla_Thr_05_1_2mip_Mip05kev.root', 2),
                    '125': ('TB2018_pion_5GeV_0Tesla_Thr_075_125_175mip_Mip05kev.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_5GeV_0Tesla_digitised_scream.root', 3),
                    '175': ('TB2018_pion_5GeV_0Tesla_Thr_075_125_175mip_Mip05kev.root', 3),
                    '2': ('TB2018_pion_5GeV_0Tesla_Thr_05_1_2mip_Mip05kev.root', 3)},
                '1GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_1GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_1GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_1GeV_0Tesla_digitised_scream.root', 3)},
                '2GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_2GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_2GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_2GeV_0Tesla_digitised_scream.root', 3)},
                '3GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_3GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_3GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_3GeV_0Tesla_digitised_scream.root', 3)},
                '4GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_4GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_4GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_4GeV_0Tesla_digitised_scream.root', 3)},
                '6GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_6GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_6GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_pion_6GeV_0Tesla_digitised_scream.root', 3)}},
            'electrons':{
                '1GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_1GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_1GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_1GeV_0Tesla_digitised_scream.root', 3)},
                '2GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_2GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_2GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_2GeV_0Tesla_digitised_scream.root', 3)},
                '3GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_3GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_3GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_3GeV_0Tesla_digitised_scream.root', 3)},
                '4GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_4GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_4GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_4GeV_0Tesla_digitised_scream.root', 3)},
                '5GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_5GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_5GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_5GeV_0Tesla_digitised_scream.root', 3)},
                '6GeV':{ 
                    '025': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_6GeV_0Tesla_digitised_scream.root', 1),
                    '05': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_6GeV_0Tesla_digitised_scream.root', 2),
                    '150': ('Thr_025_5_15mip_Mip05kev/TB2018_electron_6GeV_0Tesla_digitised_scream.root', 3)}}
            }


def readGeantFile(filename, digit):
    """Import data from a Geant4 simulation ROOT file 
    Parameters:
    -----------
        filename :   path of ROOT file
    
    Returns:
    --------
        df :     Pandas DataFrame
    """

    file = uproot.open(HOME + filename)
    file.keys()
    tree = file['digitree']
    a = tree.array("Xpos")
    df_cols = ['Xpos', 'Ypos', 'Layer', 'Thr']
    df = pd.DataFrame(list(chain(*[list(product([x],y)) for x, y in zip(range(len(a)), tree.array("Xpos"))])), columns= ['eventId',"Xpos"])
    for col in df_cols[1:]:
        df[col] = tree.array(col).flatten()

    # Rename columns to fit the data form
    df.rename(columns = {'Xpos': 'xpos', 'Ypos': 'ypos', 'Layer':'chbid', 'Thr':'digit'}, inplace=True)
    df = df[df.digit >= digit]
    return df

