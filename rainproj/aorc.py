import functools
from datetime import datetime
import numpy as np
import pandas as pd
import os
import gc
import math
from pathlib import Path
import xarray as xr
from xarray.groupers import TimeResampler
import joblib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# import logging

from rainproj.rainreader2 import RainReader2

# logger = multiprocessing.log_to_stderr()
# logger.setLevel(logging.INFO)

import os

WORK = os.environ.get("WORK")
SCRATCH = os.environ.get("SCRATCH")


class AORC:
    @staticmethod
    def download_aorc():
        years = range(1979, 2025)
        urls = [f"https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/{year}.zarr/" for year in years]
        aorc = xr.open_mfdataset(
            urls,
            engine="zarr",
            drop_variables=[
                "DLWRF_surface",
                "DSWRF_surface",
                "PRES_surface",
                "SPFH_2maboveground",
                "TMP_2maboveground",
                "UGRD_10maboveground",
                "VGRD_10maboveground",
            ],
        )["APCP_surface"]
        return aorc


def _prep_download(output_dir=f"{WORK}/hindcast/aorc"):
    dateformat = "%Y-%m-%d %H:%M:%S"
    print(f"{datetime.now().strftime(dateformat)}: Preparing")
    result = (
        AORC.download_aorc()
        .sortby("latitude")
        .rename({"latitude": "lat", "longitude": "lon"})
        .pipe(RainReader2.slice_rain, negative_lon=True)
        .chunk(time=TimeResampler("MS"))
    )  # .pipe(RainReader2.resample_daily)

    # Make positive to align with other datasets
    result["lon"] = result["lon"] + 360

    print(f"{datetime.now().strftime(dateformat)}: Interpolating")

    # TODO: don't hardcode this
    lats = np.arange(25.0, 49.5 + 0.25, 0.25)
    lons = np.arange(235.0, 293.0 + 0.25, 0.25)

    # result = result.interp(lon=lons, lat=lats,method='linear')

    print(f"{datetime.now().strftime(dateformat)}: Grouping")
    month_year_ds_groups = result.groupby(time=TimeResampler("MS"))

    print(f"{datetime.now().strftime(dateformat)}: Identifying and skipping existing files")
    month_year_ds_groups_dict = {
        f'{output_dir}/{pd.Timestamp(timestamp).strftime("%Y-%m")}_aorc.parquet': month_year_ds
        for timestamp, month_year_ds in month_year_ds_groups
    }
    month_year_ds_groups_dict = {
        filepath: ds for filepath, ds in month_year_ds_groups_dict.items() if not Path(filepath).is_file()
    }
    return month_year_ds_groups_dict


def time_with_message(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        dateformat = "%Y-%m-%d %H:%M:%S"
        start_time = datetime.now()
        message = func(*args, **kwargs)
        end_time = datetime.now()
        elapsed = end_time - start_time
        seconds = int(elapsed.total_seconds() % 60)
        minutes = int((elapsed.total_seconds() % 3600) / 60)
        hours = int(elapsed.total_seconds() / 3600)
        return f"{end_time.strftime(dateformat)}: Done with {message}...Elapsed: {hours} hrs {minutes} min {seconds} sec, Started: {start_time.strftime(dateformat)}"

    return wrapper


def run_download():
    month_year_ds_groups = _prep_download()

    @time_with_message
    def process_month(filename, month_year_ds):
        month_year_ds = month_year_ds.pipe(RainReader2.resample_daily)
        df = month_year_ds.stack(idx=("time", "lat", "lon")).to_pandas().to_frame()
        df.to_parquet(filename)
        message = filename
        return message

    print(f"{datetime.now()}: Starting Parallism with {len(month_year_ds_groups)} months to process")
    task_generator = joblib.Parallel(
        n_jobs=min(128, os.cpu_count()), timeout=99999, return_as="generator_unordered", verbose=100
    )(
        joblib.delayed(process_month)(filename, month_year_ds)
        for filename, month_year_ds in month_year_ds_groups.items()
    )

    # As generator, this loop is where Parallel actually runs
    for task in task_generator:
        print(task)

    print(f"{datetime.now()}:Done")
    # return result


@time_with_message
def _process_month(filename, month_year_ds):
    month_year_ds = month_year_ds.pipe(RainReader2.resample_daily)
    df = month_year_ds.stack(idx=("time", "lat", "lon")).to_pandas().to_frame()
    df.to_parquet(filename)
    return filename


def run_download2(workers=os.cpu_count()):
    month_year_ds_groups = _prep_download()

    print(f"{datetime.now()}: Starting Parallism with {len(month_year_ds_groups)} months to process")

    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as p:
        results = p.map(_process_month, month_year_ds_groups.keys(), month_year_ds_groups.values())
        # futures = [p.submit(_process_month,filename,month_year_ds) for filename,month_year_ds in month_year_ds_groups.items()]

        print(f"{datetime.now()}: Tasks submitted successfully")

        for result in results:
            print(f"Completed: {result}", flush=True)

    print(f"{datetime.now()}:Done")


def run_download3(workers=os.cpu_count(), output_dir=f"{WORK}/hindcast/aorc"):
    month_year_ds_groups = _prep_download(output_dir=output_dir)

    print(f"{datetime.now()}: Starting Parallism with {len(month_year_ds_groups)} months to process")

    for filename, ds in month_year_ds_groups.items():
        result = _process_month(filename, ds)
        print(f"Completed: {result}", flush=True)

    print(f"{datetime.now()}:Done")


def _process_month_batch(batch):
    result = [_process_month(month_data[0], month_data[1]) for month_data in batch]
    return result


def run_download5(workers=os.cpu_count(), nbatchs=3, output_dir=f"{WORK}/hindcast/aorc"):

    workers = min(workers, nbatchs)
    month_year_ds_groups = _prep_download(output_dir=output_dir)
    month_year_ds_groups = [(filename, ds) for filename, ds in month_year_ds_groups.items()]

    # logger.info(f'{datetime.now()}: Creating batches of months')
    batch_size = math.ceil(len(month_year_ds_groups) / nbatchs)
    month_year_ds_batchs = [
        month_year_ds_groups[batch_num * batch_size : (batch_num + 1) * batch_size] for batch_num in range(0, nbatchs)
    ]

    # logger.info(f'{datetime.now()}: Submitting tasks to process pool')
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as p:
        results = p.map(_process_month_batch, month_year_ds_batchs)

    # logger.info(f'{datetime.now()}: Tasks submitted successfully')

    for result in results:
        print(f"Completed: {result}", flush=True)


def identify_graphcast_coords():
    df = pd.read_parquet(f"{WORK}/hindcast/stage1/surface_variables_1979-01-01.parquet")
    print(df.index.min())
    print(df.index.max())
    return df


if __name__ == "__main__":
    run_download3()
