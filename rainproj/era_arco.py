from datetime import datetime
import fsspec
import xarray as xr
from xarray.groupers import TimeResampler
from rainproj.rainreader2 import RainReader2
import dask
import joblib
import os
import gc
from pathlib import Path
import pandas as pd

import os

WORK = os.environ.get('WORK')
SCRATCH = os.environ.get('SCRATCH')

# https://codes.ecmwf.int/grib/param-db/?search=crwc
class ERA:
    FILENAMES = {
        'moisture':'model-level-moisture.zarr',
        'wind':'model-level-wind.zarr',
        'forecast':'single-level-forecast.zarr',
        'reanalysis':'single-level-reanalysis.zarr',
        'surface':'se'
    }
    @staticmethod
    def _load_zarr(filename,chunks={'time': 48}):
        reanalysis = xr.open_zarr(
            filename,
            chunks=chunks,storage_options={'token': 'anon'},
            consolidated=True
        )
        return reanalysis

    @staticmethod
    def list_files(era_type='ar'):
        # Type can also be co
        fs = fsspec.filesystem('gs',token='anon')
        return fs.ls(f'gs://gcp-public-data-arco-era5/{era_type}/')

    @staticmethod
    def load_era_zarr(dataset,v2=False):
        root = 'gs://gcp-public-data-arco-era5/co'
        filename = ERA.FILENAMES[dataset]

        if v2:
            filename+='-v2'
        filename = root+'/'+filename
        return ERA._load_zarr(filename)
    
    @staticmethod
    def load_ar_total_precipitation(chunks={'time': 48}):
        root = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
        zarr = ERA._load_zarr(root,chunks=chunks)['total_precipitation']
        return zarr
        
    @staticmethod
    def load_moisture_zarr():
        return ERA._load_zarr('gcp-public-data-arco-era5/co/model-level-moisture.zarr')

    @staticmethod
    def list_attr(dataset):
        for name,value in dataset.data_vars.items():
            print(name+' '+value.attrs['GRIB_name'])

def run_download():
    print(f"{datetime.now()}: Building graph...")
    ds_daily = ERA.load_ar_total_precipitation().sortby('latitude').rename({'latitude':'lat','longitude':'lon'}).pipe(RainReader2.slice_rain).chunk(time=TimeResampler("MS")).pipe(RainReader2.resample_daily)
    month_year_groups = ds_daily.groupby(time=TimeResampler("MS"))

    def process_month(month_year_ds):
        df = month_year_ds.stack(idx=("time","lat","lon")).to_pandas().to_frame()
        month_year = df.index.get_level_values('time')[0].strftime('%Y-%m')
        df.to_parquet(f'{SCRATCH}/hindcast/era2/{month_year}.parquet')

    print(f"{datetime.now()}: Building tasks...")
    tasks = [dask.delayed(process_month)(month_year_ds) for time, month_year_ds in month_year_groups]

    print(f"{datetime.now()}: Computing...")
    dask.compute(tasks)

    print(f"{datetime.now()}: Done.")

def run_download2():
    print(f"{datetime.now()}: Building graph...")
    month_year_ds_groups = ERA.load_ar_total_precipitation().sortby('latitude').rename({'latitude':'lat','longitude':'lon'}).pipe(RainReader2.slice_rain).chunk(time=TimeResampler("MS")).pipe(RainReader2.resample_daily).groupby(time=TimeResampler("MS"))
    gc.collect()

    def process_month(timestamp,month_year_ds):
        month_year = pd.Timestamp(timestamp).strftime('%Y-%m') #numpy datetime to datetime
        filename = f'{SCRATCH}/hindcast/era3/{month_year}.parquet'
        # month_year = df.index.get_level_values('time')[0].strftime('%Y-%m')
        if not Path(filename).is_file():
            df = month_year_ds.stack(idx=("time","lat","lon")).to_pandas().to_frame()
            df.to_parquet(filename)
            print(f"{datetime.now()}: Done with {month_year}...")
            del df
        del month_year,filename
        gc.collect()

    print(f"{datetime.now()}: Building tasks...") # From htop, roughly estimated that 20 task will consume 80 GB, note that TACC nodes have 251 GB, probably safe to use 40 workers avoid memory overload. Or maybe even stick to 20 since this is an IO bound task. Cpus will wait until all data is got,.

    tasks = joblib.Parallel(n_jobs=min(10,os.cpu_count()),verbose=100,timeout=99999)(joblib.delayed(process_month)(timestamp, month_year_ds) for timestamp, month_year_ds in month_year_ds_groups)
    # tasks = [dask.delayed(process_month)(month_year_ds) for month_year_ds in month_year_groups]

    print(f"{datetime.now()}: Computing...")
    # dask.compute(tasks)

    print(f"{datetime.now()}: Done.")

if __name__=='__main__':
    run_download2()
