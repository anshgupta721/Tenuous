import pandas as pd

cov = pd.read_csv("pysim/data/chesley_coeffs_covariance.csv", index_col=0)
gm = cov.index[0]            # "GM2101955"
cov.loc[gm, :] *= 1e9        # GM row
cov.loc[:, gm] *= 1e9        # GM column (the GM-GM entry is hit twice, so it gets 1e18)
cov.to_csv("bennu_gravity_covariance_m.csv", float_format="%.17g")