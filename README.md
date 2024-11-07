# Minimum reserve capacity calculation (year-ahead)

For the minimum reserve capacity calculation run py/minimum_reserve_capacity_calc.py. Results are generated to the
same folder report_<current_day-current_month-current_year.pdf> as a name.

Note that this is not a probabilistic calculation!

It takes the input data (constants.py/ACEOL_TABLE): at least one full year (constants.py/AT_LEAST_ONE_FULL_YEAR_PERIOD) starting not less than
6 months before the current date (constants.py/NOT_EARLIER_THAN_SIX_MONTHS). Extracts data for baltics region
(constants.py/REGIONS). Finds the 0.01 and 99.99 percentile (constants.py/PERCENTILE_0001_VALUE and constants.py/PERCENTILE_9999_VALUE)
and presents these as a result.

Because
From the methodology:

<i>
...

To determine the reserve capacity at SOR level required to respect the FRCE target parameters in Article 128 of SO 
Regulation, a probabilistic approach shall be applied additionally.

1. TSOs of the relevant SOR shall provide the RCC the FC block imbalance data time series </i>(ACEol)<i>. The sampling 
of those time series shall cover the time to restore the frequency according to Annex III of SO Regulation. The
time period considered for those historical records shall be representative and include at least one full year period
ending not earlier than six months before the calculation date. The time period considered shall be the same for all 
LFC block imbalance time series within the relevant SOR and agreed by all TSOs of the relevant SOR.

2.  The RCC shall sump up per sampling time the LFC block imbalance time series of the SOR received under point 1. 
without separating positive and negative imbalances

3. The RCC shall calculate the reserve capacity needed to cover the positive SOR imbalances for at least 99,99% of the
time based on the historical records summed up at SOR level referred to in point 2. The use of applying this level is 
decrease system operational risks which are increased by not separating positive and negative imbalances under 
point 2.

4. The RCC shall calculate the reserve capacity needed to cover the negative SOR imbalances for at least 99.99% of the 
time based on the summed up historical records referred to in point (b). The use of applying this level is to decrease
system operational risks which are increased by not separating positive and negative imbalances under point 2.

...

</i>




