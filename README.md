# RPWELL_SDHCAL_analysis
Analysis framework for RPWELL-based SDHCAL sampling elements.

This repository contains the analysis framework for data obtained with MICROROC active sensor unit (ASU).

The code for a single sampling element MIP detection efficiency estimation can be found in the folder ```src/mip_eff/```.

The required packages are listed in ```requirements.txt```, and can be installed using
```
pip install -r requirements.txt
```

```path.txt``` should contain the exact path where the **simplified** ROOT data files are stored.

The MIP detection efficiency analysis can run as is using:
```
python efficiency_estimation.py -m dt -n 8
```
One can add the following selections:

 | option | description |
 | --- | --- |
 | -m or --mode | the analysis mode - only trigger-time correlated hits (dt) or caloEvents (calo). Default: dt. |
 | -n or --Nchb | the number of layer in the setup 8 or 11. Default: 8. |
