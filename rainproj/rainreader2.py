import os
import sys
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import xarray as xr
from functools import partial
import os
from joblib import Parallel, delayed

class RainReader2:
    def __init__(self,root_dir,chunks=None) -> None:
        self.root_dir = root_dir
        self.dset = None
        self.chunks = chunks
        self.DASK_INSTALLED = 'dask' in sys.modules

    @staticmethod
    def read_rain(filepath,):
        return xr.open_dataset(filepath)['tp06']
    
    @staticmethod
    def slice_rain(data,negative_lon=False):
        if negative_lon:
            min_lon = -125.0011
            max_lon = -66.9326
        else:
            min_lon = 360-125.0011
            max_lon = 360-66.9326
        return data.sel(lat=slice(24.9493,49.5904),lon=slice(min_lon,max_lon),time=slice('1978-12-31','2025-01-01'))
    
    @staticmethod
    def resample_daily(data):
        data = data.resample(time='1D').sum()
        data.attrs["long_name"]="24-Hour Total Precipitation"
        return data
    
    @staticmethod
    def redim(rain_array,max_days=None):
        start_time = rain_array['time'][0]
        rain_array['time'] = pd.to_timedelta(rain_array['time']-start_time)
        rain_array = rain_array.rename({'time':'lead_time'}).assign_coords(date=start_time).drop_vars('time')
        return rain_array
    
    @staticmethod
    def limit_lead_days(rain_array,max_days):
        rain_array = rain_array.sel(lead_time=slice('0D',max_days))
        return rain_array
    
    @staticmethod
    def stack(data):
        data = pd.concat({data['date'].values:data.stack(idx=("lat","lon","lead_time")).to_series()},names=['date']).rename('precipitation tp06 [m]').sort_index().to_frame()
        return data
    
    @staticmethod
    def run_all(filepath):
        data = RainReader2.read_rain(filepath)
        data = RainReader2.slice_rain(data)
        data = RainReader2.resample_daily(data)
        data = RainReader2.redim(data)
        data = RainReader2.limit_lead_days(data,'7D')
        data = RainReader2.stack(data)
        return data
