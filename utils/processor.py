import requests
import os
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
import numpy as np
import dataretrieval.nwis as nwis
from sklearn.linear_model import LinearRegression
from .pysda import sdapoly, sdaprop, sdainterp
from shapely.geometry import Polygon


class GWIC():
    def __init__(self, gwicids):
        """gwicids (list): list of gwicids to process"""
        self.gwicids = gwicids

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
    
    def drop_outliers(self, raw_data, std=5):
        """Drop outliers from raw_data
        
        std: int, number of standard deviations to use for outlier detection
        """
        
        stl = sm.tsa.STL(raw_data['swl_ground'], period=12, robust=True)
        result = stl.fit()
        residuals = result.resid
        outliers = residuals[np.abs(residuals) > np.std(residuals)*std]
        return raw_data.drop(outliers.index)

    def process_data(self, data):
        """Process the data and return the result as pandas DataFrame"""
        
        df = pd.DataFrame(data)
        df = df[['gwicid', 'date_measured', 'swl_ground']]
        df = df.rename(columns={'date_measured': 'time'})
        df['time'] = pd.to_datetime(df['time'])
        df = self.drop_outliers(df)
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

class USGS():
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

class SSURGO():
    def __init__(self, shp_path, gdf_pts):
        self.shp_path = shp_path
        self.gdf_pts = gdf_pts
        
    def create_bbox_shp(self):
        """Create a bounding box from pts and save it as a shapefile."""
        bounds = self.gdf_pts.total_bounds
        bbox = Polygon([(bounds[0], bounds[1]), (bounds[0], bounds[3]), (bounds[2], bounds[3]), (bounds[2], bounds[1])])
        gdf_bbox = gpd.GeoDataFrame({'geometry': [bbox]}, crs=self.gdf_pts.crs)
        gdf_bbox = gdf_bbox.buffer(1000, join_style=2).to_crs(epsg=4326)
        gdf_bbox.to_file(self.shp_path)
        
    def soil_prop(self, prop='ksat_r'):
        """Get soil property from SSURGO for all points."""
        myaoi = sdapoly.shp(self.shp_path)
        # spatial join to get soil properties that intersect with pts
        units = gpd.sjoin(myaoi, 
                  self.gdf_pts.reset_index().to_crs(4326), 
                  how='inner', 
                  predicate='intersects')
        # weighted average of property
        wtdavg = sdaprop.getprop(df=units,
                                 column='mukey',
                                 method='wtd_avg',
                                 top=0,
                                 bottom=100,
                                 prop=prop,
                                 minmax=None,
                                 prnt=False,
                                 meta=False)
        # merge and change ksat None to zero
        df_prop = units[['gwicid', 'mukey']].merge(wtdavg[['mukey', prop]], 
                                                   on='mukey').fillna(0)
        return df_prop[['gwicid', prop]].merge(self.gdf_pts, on='gwicid')
    
    def get_all_data(self, prop='ksat_r'):
        """Get soil properties for all points."""
        if not os.path.exists(self.shp_path):
            self.create_bbox_shp()
        df_prop = self.soil_prop(prop=prop)
        df_prop[prop] = df_prop[prop].astype(float)
        return df_prop.set_index('gwicid')

class Imputer():
    def __init__(self, df):
        """df: pandas DataFrame with Q and swl columns"""
        self.df = df
    
    def q_and_doy_regression(self):
        """Impute missing values in the DataFrame using a linear regression model
        with day of year and discharge as predictors."""
        
        reg_imputed = self.df.copy()
        
        # Loop through columns
        for c in np.arange(1, self.df.shape[1]):

            # Drop preceding NaNs
            first_ind = self.df.iloc[:, c].first_valid_index()
            new_df = self.df.iloc[:, [0, c]][first_ind:]  
                
            # Get indices of NaNs
            imputed_indices = new_df.iloc[:, 1][new_df.iloc[:, 1].isnull()].index

            # Drop rows with missing values
            df_dropped = new_df.iloc[:, [0, 1]].dropna()    

            # Get predictors (Q and doy) and dependent variable
            X = np.column_stack((df_dropped.Q.values, df_dropped.index.dayofyear.values))
            Y = df_dropped.iloc[:, 1].values

            # Instantiate model
            model = LinearRegression()

            # Fit model
            model.fit(X, Y)

            # Predict missing values
            pred1 = new_df.loc[imputed_indices, 'Q'].values
            pred2 = new_df.loc[imputed_indices, 'Q'].index.dayofyear.values
            preds = np.column_stack((pred1, pred2))
            predicted = model.predict(preds)

            # Fill missing values with predicted values
            reg_imputed.iloc[:, c][imputed_indices] = predicted
        
        return reg_imputed
    
    def linear(self):
        """Impute missing values in the DataFrame using linear interpolation."""
        
        # Make sure dataset is clean each time
        lin_imputed = self.df.copy()
        
        # Loop through columns
        for c in np.arange(1, self.df.shape[1]):

            # Drop preceding NaNs
            first_ind = lin_imputed.iloc[:, c].first_valid_index()
            new_df = lin_imputed.iloc[:, c][first_ind:]

            # Linear interpolation
            new_df_int = new_df.interpolate(method='linear')
            new_df_int_ind = new_df_int.index

            # Update dataframe
            lin_imputed.iloc[:, c][new_df_int_ind] = new_df_int

        return lin_imputed