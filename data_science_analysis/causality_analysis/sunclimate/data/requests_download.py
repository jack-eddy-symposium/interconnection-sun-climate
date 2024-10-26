"""
    Module to download geophysical data from disc.gsfc.nasa.gov
"""

import os
from functools import partial
import requests
from tqdm.contrib.concurrent import process_map

# Precipitation percentiles:  https://disc.gsfc.nasa.gov/datasets/M2SMNXPCT_1/summary
COLLECTION_PATH = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_CLIM/M2SMNXPCT.1'
COLLECTION_NAME = 'M2SMNXPCT.1'
START_MONTH = 1
START_YEAR = 1980
END_YEAR = 2022
END_MONTH = 12
FILE_PREFIX = 'MERRA2.statM_2d_pct_Nx.'
FILE_SUFIX = '.nc4'


# Combined precipitation: https://disc.gsfc.nasa.gov/datasets/GPCPMON_3.2/summary
# COLLECTION_PATH = 'https://measures.gesdisc.eosdis.nasa.gov/data/GPCP/GPCPMON.3.2333'
# COLLECTION_NAME = 'GPCPMON.3.2'
# START_MONTH = 1
# START_YEAR = 1983
# END_YEAR = 2022
# END_MONTH = 9
# FILE_PREFIX = 'GPCPMON_L3_'
# FILE_SUFIX = '_V3.2.nc4'


def requests_download(path_save:str, file_url:str):
    """
        Function to download a file from NASA Gesdisc repository using Requests
        function made for parallelization

        Parameters:
            path_save:
                folder path to save the file in
            file_url: str
                path to the file to download
    """
    result = requests.get(file_url)
    filename = os.path.join(path_save, file_url.split('/')[-1])
    # try:
    result.raise_for_status()
    f = open(filename,'wb')
    f.write(result.content)
    f.close()
    # print('contents of URL written to '+filename)

    return filename


if __name__ == "__main__":

    for year in range(START_YEAR, END_YEAR+1):

        start_month = 1
        end_month = 12
        if year == START_YEAR:
            start_month = START_MONTH
        if year == END_YEAR:
            end_month = END_MONTH
        
        path = os.path.join('D:\\', 'geodata', COLLECTION_NAME, str(year))
        if not os.path.exists(path):
            os.makedirs(path)

        url = f'{COLLECTION_PATH}/{year}'
        files_year = [f'{url}/{FILE_PREFIX}{year}{str(month).zfill(2)}{FILE_SUFIX}' for month in range(start_month,end_month+1,1)]

        if not os.path.exists(path):
            os.makedirs(path)

        r = process_map(partial(requests_download, path), files_year, max_workers=4, position=0, leave=True)