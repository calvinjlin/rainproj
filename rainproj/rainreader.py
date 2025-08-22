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


def _check_access(file):
    if not (os.access(file,os.R_OK)):
        print(f'No read permission: {file}')
        return None
    else: 
        return file

def list_available_files(ROOT_DIR,run_parallel=True):
    nc_folder_files = sorted(list(glob.glob(f'{ROOT_DIR}/*/surface_variables_*.nc',recursive=True)))

    if not run_parallel:
        for file in nc_folder_files:
            _check_access(file)
        accessible_files = [file for file in nc_folder_files if os.access(file,os.R_OK)]
    else:
        print("Parallel enabled")
        parallel = Parallel(n_jobs=os.cpu_count()-1)
        accessible_files = parallel(delayed(_check_access)(file) for file in nc_folder_files)
            
    
    print(f'{len(accessible_files)} of {len(nc_folder_files)} {len(accessible_files)/len(nc_folder_files)*100} files accessible')
    return accessible_files

class RainReader:
    def __init__(self,root_dir,chunks=None) -> None:
        self.root_dir = root_dir
        self.dset = None
        self.chunks = chunks
        self.DASK_INSTALLED = 'dask' in sys.modules

    def list_year_folders(self):
        folders = [entry for entry in os.listdir(self.root_dir) if os.path.isdir(f'{self.root_dir}/{entry}') and (len(entry)==4)]
        return sorted(folders)
    
    def list_files_in_year(self,year,prefix=None):
        filenames = os.listdir(f"{self.root_dir}/{year}")
        if prefix:
            filenames = [entry for entry in filenames if prefix in entry]
        return filenames

    def list_rain_files_in_year(self,year):
        return self.list_files_in_year(year=year,prefix='surface_variables')
    
    def list_dates(self,year,month=None):
        dates = [entry.split('_')[-1].split('.')[0] for entry in self.list_rain_files_in_year(year=year)]
        if month:
            MM = str(month).zfill(2)
            dates = [date for date in dates if f'-{MM}-' in date]
        return dates
        
    def read_rain(self,date,prefix='surface_variables'):
        return xr.open_dataset(f"{self.root_dir}/{date.split('-',maxsplit=1)[0]}/{prefix}_{date}.nc",engine='h5netcdf',chunks=self.chunks)['tp06']

    @staticmethod
    def redim(rain_array,resample=None,max_days=None):
        start_time = rain_array['time'][0]
        rain_array['time'] = pd.to_timedelta(rain_array['time']-start_time)
        rain_array = rain_array.rename({'time':'lead_time'}).assign_coords(time=start_time)

        # For resample, provide a frequency string like '1D'
        if resample:
            rain_array = rain_array.resample(lead_time=resample).sum()
            if max_days:
                rain_array = rain_array.sel(lead_time=slice('0D',max_days))
        return rain_array
    
    @staticmethod
    def redim_dask_func(rain_array):
        return RainReader.redim(rain_array,resample='1D',max_days='7D')
    
    def process_file(self,date):
        return self.redim(self.read_rain(date),resample='1D',max_days='7D')
    
    def process_rain_month(self,year,month,force_multiprocessing=False):
        dates = self.list_dates(year=year,month=month)
        
        if self.DASK_INSTALLED and self.chunks is not None and not force_multiprocessing:
            print('Native multiprocessing module not used, dask enabled')
            days = [self.process_file(date) for date in dates]
        else:
            print('Multiprocessing module used')
            with ProcessPoolExecutor(max_workers=os.cpu_count()-1,mp_context=multiprocessing.get_context("spawn")) as p:
                days = list(p.map(self.process_file,dates))
        
        return xr.concat(days,dim='time').sortby('time')
    
    @staticmethod
    def to_netcdf(dset,*,filepath,compression=True):

        if compression:
            if isinstance(dset,xr.DataArray):
                compression_params = {dset.name:dict(compression='zlib',shuffle=True)}   # default complevel is 4
            else:
                compression_params = {var:dict(compression='zlib',shuffle=True) for var in dset.data_vars}
        else:
            compression_params = None
        dset.to_netcdf(filepath,format='NETCDF4',engine='h5netcdf',encoding=compression_params)
