import requests
import pandas as pd
import geopandas as gpd

gwicid = "139851"



def get_data(gwicid):
    """Get data from the API and return it as a dict"""

    url = "https://mbmgweb1a.mtech.edu:5001/Swl/json?gwicid={}".format(gwicid) 
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request was unsuccessful
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print("Error connecting to the API:", e)
        return None

def process_data(data):
    """Process the data and return the result as pandas DataFrame"""
    
    # Assuming data is a dictionary
    df = pd.DataFrame(data)
    return df

def process_data(data):
    """Process the data and return the result as a GeoDataFrame"""
    
    # Assuming data is a dictionary
    df = pd.DataFrame(data)

    # Convert lat and lon to a geometry column
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    return gdf

metadf = pd.read_csv('./data/missoula_valley_monitored_wells.csv')
metadf = metadf[['gwicid', 'latitude', 'longitude']]



data = get_data(gwicid)
df = process_data(data)