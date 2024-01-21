import requests
import os
import pandas as pd
import dataretrieval.nwis as nwis


class GWIC(object):
    def __init__(self, gwicids):
        """gwicids (list): list of gwicids to process"""
        self.gwicid = gwicids

    def get_data(self, gwicid):
        """Get data from the gwic API and return it as a dict"""

        url = "https://mbmgweb1a.mtech.edu:5001/Swl/json?gwicid={}".format(gwicid) 
        try:
            response = requests.get(url)
            response.raise_for_status()  
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print("Error connecting to the API:", e)
            return None

    def process_data(self, data):
        """Process the data and return the result as pandas DataFrame"""
        
        df = pd.DataFrame(data)
        df = df[['gwicid', 'date_measured', 'swl_ground']]
        df = df.rename(columns={'date_measured': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        df = df.loc[(df.time >= '1995-01-01') &
                    (df.time < '2024-01-01')]
        df = df.resample('M', on='time').median()
        df['gwicid'] = df['gwicid'].ffill().astype(int)
        df = df.reset_index()
        return df

    def get_all_data(self, filename, check_file=True):
        """Return monthly processed data for all gwicids as a pandas DataFrame.
        
        filename: str, filename to check for data or save data to
        check_file: bool (default=True) If True, check if file exists. If False,
                    retreive data from API and process it."""
        
        if check_file and os.path.exists(filename):
            df_full = pd.read_csv(filename, index_col='time')
            df_full.index = pd.to_datetime(df_full.index)
            return df_full
        else:
            df_full = pd.DataFrame()
            for gwicid in self.gwicids:
                print('Processing data for well {}'.format(gwicid))
                data = self.get_data(gwicid)
                df = self.process_data(data)
                df.set_index('time', inplace=True)
                df.rename(columns={'swl_ground': gwicid}, inplace=True)
                df_full = pd.concat([df_full, df[gwicid]], axis=1)
            df_full.index.name = 'time'
            df_full.index = pd.to_datetime(df_full.index)
            df_full.index = df_full.index.tz_localize(None)
            df_full.to_csv(filename, index=True)
        return df_full

class USGS(object):
    def __init__(self, siteid):
        """siteid (int): siteid to process"""
        self.siteid = siteid

    def get_data(self):
        """Get data from the USGS API and return it as a dict"""

        try:
            df = nwis.get_record(sites=self.siteid, service='dv', start='1995-01-01', end='2024-01-01')
            return df['00060_Mean']
        
        except requests.exceptions.RequestException as e:
            print("Error connecting to the API:", e)
            return None

    def process_data(self):
        """Process the data and return the result as pandas DataFrame"""
       
        df = self.get_data()
        df = pd.DataFrame(df)
        df = df.reset_index()
        df = df.rename(columns={'00060_Mean': 'Q', 'datetime': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        df = df.loc[(df['time'] >= '1995-01-01') &
                    (df['time'] < '2024-01-01')]
        df = df.resample('M', on='time').median()
        return df
    
    def get_all_data(self, filename, check_file=True):
        
        if check_file and os.path.exists(filename):
            df = pd.read_csv(filename, index_col='time')
            df.index = pd.to_datetime(df.index)
            return df
        else:
            df = self.process_data()
            df.index.name = 'time'
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df.to_csv(filename, index=True)
        return df

# Get gwicids
gwic_filename = './data/missoula_valley_monitored_wells_data.csv'
metadf = pd.read_csv('./data/gwic_site_metadata.csv')
metadf = metadf[['gwicid', 'latitude', 'longitude']]
gwicids = metadf['gwicid'].values

# Process gwic data
GWIC = GWIC(gwicids)
gwic_data = GWIC.get_all_data(filename=gwic_filename, check_file=True)
gwic_data.plot(kind='line', style='o', legend=False, markersize=2)

# Process USGS data
clk_fk_avb_mso = '12340500'
usgs_filename = './data/clark_fk_above_missoula_q.csv'
USGS = USGS(clk_fk_avb_mso)
usgs_data = USGS.get_all_data(filename=usgs_filename, check_file=True)
usgs_data.plot(kind='line', legend=False)