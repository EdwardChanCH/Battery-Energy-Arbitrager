# CS4453B

Group Members:
* Chun Ho Chan
* Dilraj Deogan
* Doyle Blacklock
* Liam Truss


## Dependencies

* Git LFS
  * https://git-lfs.com/
  * Tracked files: *.csv , *.xlsx , *.png
  * Tip: Use "gif lfs fetch origin branch_name" to download the tracked files.


## Jupyter Notebook Dataset

The dataset was created as part of DataSci 3000A at Western University.
The dataset is permitted for use in CompSci 4453B at Western University.

Source:
* https://github.com/EdwardChanCH/Electricity-Demand-Forecaster

Authors:
* Jonathan Abado
* Chun Ho Chan
* Doyle Blacklock
* Keefe Feng
* Siddh Parikh


## Electricity Data
Sources:
* https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48
* https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/balancing_authority/FPL

1. Florida Power & Light Co. (FPL) Electricity Overview
  * Description:
    * Electricity hourly data at Florida's FPL node, from 2015 Jan 01 to 2025 Oct 25.
    * No older data is available.
    * The Excel file contains charts and metadata.
    * Hourly data.
  * Old Filename(s): FPL.xlsx
  * New Filename(s): FPL.csv

2. Florida Regional Electricity Overview
  * Description:
    * Probably not as useful as FPL.csv, just an alternative.
    * Elctricity hourly data at the entire Florida region, from 2015 Jan 01 to 2025 Oct 25.
    * No older data is available.
    * The Excel file contains charts and metadata.
    * Hourly data.
  * Old Filename(s): Region_FLA.xlsx
  * New Filename(s): Region_FLA.csv


## Weather Data
Website:
* https://www.ncei.noaa.gov/access/search/data-search/local-climatological-data

1. MIAMI INTERNATIONAL AIRPORT, FL US
  * Description:
    * Weather hourly data at Miami International Airport, from 2015 Jan 01 to 2025 Aug 25.
    * More older data is available (as far as 1940s).
    * The timestamp needs to be rounded to match FPL.csv.
  * Old Filename(s): 72202012839.csv (duplicated names)
  * New Filename(s): MIA_year