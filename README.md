# Mapping Metastasis

# For Data Process
Please find Data process file for R codes used for processing raw clincial trial data

# For Data Analysis
Please find Data analysis file for R codes used for random effect cox proportional model
 
For Monolix codes:
new.txt is the tumor growth model, 265target_monolix is the demo data, mode265 is the monolix running file.
Monolix install guide: https://lixoft.com/downloads/full-documentation/
Monolix install time: usually no more than 30minutes.
Model demo run instruction: the running file has set up all the inital values and data distributions. Click "run" and the results include population and individual parameter estimates.

Expected results:
ESTIMATION OF THE POPULATION PARAMETERS ________________________________________

Fixed Effects ----------------------------     se_sa    rse(%)
kge_pop     :                      0.00473  0.000393       8.3
kkill_pop   :                       0.0292   0.00198       6.8
F_pop       :                        0.051    0.0089      17.4

Standard Deviation of the Random Effects -
omega_kge   :                         1.11    0.0553      4.99
omega_kkill :                         1.19     0.061      5.14
omega_F     :                         3.29     0.146      4.44

Error Model Parameters -------------------
b           :                        0.366   0.00593      1.62




For python codes:
organorder cluster.py is the python codes we used for k-means clustering.
predictxgb_simple.py is the python codes we used for Gradient boosting predictive model.
The model was run in Anacoda 3 and Spyder 4.2.5.
Installation guide:
Download website: https://www.anaconda.com/ 
installation time is around 20 minutes.



