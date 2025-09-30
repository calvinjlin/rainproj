import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import brier_score_loss
from xskillscore import brier_score
import os
import glob

WORK = os.environ.get('WORK')
SCRATCH = os.environ.get('SCRATCH')

idx = pd.IndexSlice

def process_lead_day(era_df,hc_df,lead_day,xskill=False):   
    hc_df = hc_df.loc[idx[:,:,:,lead_day]]
    era_df = era_df.loc[hc_df.index]
    if not xskill:
        return {lead_day:{'BS 25ref':brier_score_loss(era_df['>25mm'],era_df['>25mm climate']),'BS 2.5ref':brier_score_loss(era_df['>2.5mm'],era_df['>2.5mm climate']),
                      'BS 25':brier_score_loss(era_df['>25mm'],hc_df['>25mm']),'BS 2.5':brier_score_loss(era_df['>2.5mm'],hc_df['>2.5mm']),
                      }}
    else:
        return {lead_day:{
                        'BS 25ref xskill':brier_score(era_df['>25mm'],era_df['>25mm climate']),'BS 2.5ref':brier_score(era_df['>2.5mm'],era_df['>2.5mm climate']),
                        'BS 25 xskill':brier_score(era_df['>25mm'],hc_df['>25mm']),'BS 2.5':brier_score(era_df['>2.5mm'],hc_df['>2.5mm']),
                    }}

def calc_prob(frame,col):
    # Both ERA and hindcast use units of meters for precipitation.
    rain_mm = frame[col]*1000
    frame['>2.5mm'] = rain_mm>2.5
    frame['>25mm'] = rain_mm>25
    return frame

def xprocess_month(month,*,era_paths,hc_paths,xskill=False):
    with Parallel(n_jobs=-1,prefer="threads",return_as='generator_unordered') as parallel:
        era_df = pd.concat(parallel(delayed(pd.read_parquet)(file) for file in era_paths[month])).pipe(calc_prob,'total_precipitation')
        hc_df = pd.concat(parallel(delayed(pd.read_parquet)(file) for file in hc_paths[month])).pipe(calc_prob,'precipitation tp06 [m]')

    print("load done")
    # Climatology
    era_df['month'] = era_df['time'].dt.month
    era_df['day'] = era_df['time'].dt.day
    era_df['>2.5mm climate'] = era_df.groupby(['month','day'])['>2.5mm'].transform('mean')
    era_df['>25mm climate'] = era_df.groupby(['month','day'])['>25mm'].transform('mean')
    era_df = era_df.drop(['month','day'],axis=1)

    print("start tasks")
    lead_days = hc_df.index.get_level_values('lead_time').unique()
    lead_days = lead_days[lead_days<=pd.Timedelta('7D')]

    results = Parallel(n_jobs=3,return_as='generator_unordered')(delayed(process_lead_day)(era_df,hc_df,lead_day,xskill=xskill) for lead_day in lead_days)
    return {month:{k:v for item in results for k,v in item.items()}}
    # return era_df,hc_df,lead_days

def process_month(month,*,era_paths,hc_paths,xskill=False):
    with Parallel(n_jobs=-1,prefer="threads",return_as='generator_unordered') as parallel:
        era_df = pd.concat(parallel(delayed(pd.read_parquet)(file) for file in era_paths[month])).pipe(calc_prob,'total_precipitation')
        hc_df = pd.concat(parallel(delayed(pd.read_parquet)(file) for file in hc_paths[month])).pipe(calc_prob,'precipitation tp06 [m]')

    print("load done")
    # Climatology
    era_df['month'] = era_df.index.get_level_values('time').month
    era_df['day'] = era_df.index.get_level_values('time').day
    era_df['>2.5mm climate'] = era_df.groupby(['month','day'])['>2.5mm'].transform('mean')
    era_df['>25mm climate'] = era_df.groupby(['month','day'])['>25mm'].transform('mean')
    era_df = era_df.drop(['month','day'],axis=1)

    print("start tasks")
    lead_days = hc_df.index.get_level_values('lead_time').unique()
    lead_days = lead_days[lead_days<=pd.Timedelta('7D')]

    results = Parallel(n_jobs=3,return_as='generator_unordered')(delayed(process_lead_day)(era_df,hc_df,lead_day,xskill=xskill) for lead_day in lead_days)
    return {month:{k:v for item in results for k,v in item.items()}}
    # return era_df,hc_df,lead_days

def process_all(*,era_paths,hc_paths,xskill=False):
    return {k:v for list_item in (process_month(month,era_paths=era_paths,hc_paths=hc_paths,xskill=xskill) for month in range(1,13)) for k,v in list_item.items()}

if __name__ == '__main__':
    era_paths = {month:sorted(list(glob.glob(f'{SCRATCH}/hindcast/era3/*-{str(month).zfill(2)}.parquet',recursive=True))) for month in range(1,13)}
    hc_paths = {month:sorted(list(glob.glob(f'{WORK}/hindcast/stage1/surface_variables_*-{str(month).zfill(2)}-*.parquet',recursive=True))) for month in range(1,13)}

    test2 = process_all(era_paths=era_paths,hc_paths=hc_paths,xskill=True)

    df = pd.DataFrame.from_dict({(month,lead):metrics for month,lead_list in test2.items() for lead,metrics in lead_list.items()},orient='index').sort_index()
    df.index.set_names(['Month','Lead Day'],inplace=True)
    df.to_parquet(f'{WORK}/rainproj/data/br_loss_score_xskill.parquet')
