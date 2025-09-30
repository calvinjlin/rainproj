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

    @staticmethod
    def xload_rain_dataset():
        return ERA.load_ar_total_precipitation().sortby('latitude').rename({'latitude':'lat','longitude':'lon'}).pipe(RainReader2.slice_rain)

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

def run_download3():
    import glob 
    ds = ERA.xload_rain_dataset()
    date_strings = sorted(set(str(date.astype('datetime64[D]')) for date in ds.time.values))

    era_stage1_dir = Path(f'{SCRATCH}/hindcast/xarray/era_stage1/')
    era_stage1_dir.mkdir(parents=True,exist_ok=True)

    date_strings = [date_string for date_string in date_strings if not Path(era_stage1_dir/f'era_{date_string}.nc').is_file()]
    def process_day(date,sub_ds):
        sub_ds.pipe(RainReader2.resample_daily).to_netcdf(era_stage1_dir/f'era_{date}.nc')
    tasks = joblib.Parallel(n_jobs=os.cpu_count(),verbose=100,timeout=99999)(joblib.delayed(process_day)(date,ds.sel(time=date)) for date in date_strings)


    year_months = set([filename.split('_')[-1].rsplit('-',maxsplit=1)[0] for filename in os.listdir(era_stage1_dir)])
    mon_year_files = {prefix:glob.glob(f'{SCRATCH}/hindcast/xarray/era_stage1/*{prefix}*.nc') for prefix in year_months}

    stage2_dir = Path(f'{SCRATCH}/hindcast/xarray/era_stage2/')
    stage2_dir.mkdir(parents=True,exist_ok=True)
    def execute1(prefix_file_batch):
        prefix,file_batch = prefix_file_batch
        out_path = f'{SCRATCH}/hindcast/xarray/era_stage2/era_rain_{prefix}.nc'
        xr.open_mfdataset(file_batch,combine='nested',concat_dim='time',engine="netcdf4").sortby('time').to_netcdf(out_path)
    parallel = joblib.Parallel(n_jobs=os.cpu_count()-1)
    silent2 = parallel(joblib.delayed(execute1)(prefix_file_batch) for prefix_file_batch in mon_year_files.items())

if __name__=='__main__':
    run_download3()
