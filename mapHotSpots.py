import numpy as np 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import sys
import datetime 
from shapely.geometry import Point
from scipy.stats import gaussian_kde
import xarray as xr
import rioxarray
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj 
from cartes.crs import EPSG_3035

indir='/data/paugam/EUBURN/Silex/Hotspots/'
#gdf = gpd.read_file(indir+'fire_archive_M-C61_527571.shp')
gdf_all = gpd.read_file(indir+'fire_archive_M-C61_528314.shp')

# Convert 'ACQ_DATE' to datetime
gdf_all['ACQ_DATE'] = pd.to_datetime(gdf_all['ACQ_DATE'])

# Convert 'ACQ_TIME' to a string and then to a time object (HHMM format)
gdf_all['ACQ_TIME'] = gdf_all['ACQ_TIME'].apply(lambda x: pd.to_timedelta( float(x[:2])*60 + float(x[2:]), unit='minutes'))

# Combine 'ACQ_DATE' and 'ACQ_TIME' into a single datetime column
gdf_all['timestamp'] = gdf_all['ACQ_DATE'] + gdf_all['ACQ_TIME']

gdf_all['timestamp'] = pd.to_datetime(gdf_all['timestamp'])  # Ensure timestamp is in datetime format

gdf_all['longitude'] = gdf_all.geometry.x
gdf_all['latitude'] = gdf_all.geometry.y

for year in np.arange(2017,2024):

    print(year)
#density of fire event on period july to august
    start_summer = datetime.datetime(int(year),7,1)
    end_summer =   datetime.datetime(int(year),8,31)

    gdf = gdf_all[(gdf_all['timestamp']>start_summer) &
                                    (gdf_all['timestamp']<end_summer)]

 
# Use DBSCAN for spatial clustering (define 1000 meters as the spatial threshold)
# DBSCAN requires the distance in degrees, so you might need to convert meters to degrees.
# Approximation: 1 degree latitude ~ 111 km.
    epsilon = 1. / 111.0  # Approx. 1 km in degrees
    db = DBSCAN(eps=epsilon, min_samples=1, metric='haversine').fit(np.array(gdf[['longitude','latitude' ]]))

# Add cluster labels to the GeoDataFrame
    gdf.loc[:,['cluster']] = db.labels_

# Optionally, sort by time and perform temporal aggregation within each spatial cluster
    gdf = gdf.sort_values(by=['cluster', 'timestamp'])

# You can define a time threshold, e.g., 24 hours
    time_threshold = pd.Timedelta('2 day')

# Create a fire event ID by combining spatial clusters with temporal proximity
    gdf.loc[:,['fire_event']] = np.array((gdf.groupby('cluster')['timestamp'].apply(lambda x: (x.diff() > time_threshold).cumsum()+1 )))

# Create a unique fire_event ID across all clusters by concatenating cluster and fire_event
    gdf.loc[:,['cluster_event_id']] = gdf['cluster'].astype(str) + '_' + gdf['fire_event'].astype(str)

# Map the unique combinations to a continuous global fire event index
    gdf.loc[:,['global_fire_event']] = gdf['cluster_event_id'].factorize()[0]

# Aggregate hotspots by fire event
    fire_events = gdf.groupby('global_fire_event').agg(
        total_hotspots=('latitude', 'count'),
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        #avg_frp=('frp', 'mean')  # Assuming FRP (Fire Radiative Power) is present
    ).reset_index()

# Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(fire_events['longitude'], fire_events['latitude'])]
    fire_events = gpd.GeoDataFrame(fire_events, geometry=geometry)
    fire_events = fire_events.set_crs('epsg:4326')
    
    fire_events = fire_events.to_crs('epsg:3035')

    '''
#density of fire event on period july to august
    start_summer = datetime.datetime(int(year),7,1)
    end_summer =   datetime.datetime(int(year),8,31)

    fireEvent_summer = fire_events[(fire_events['start_time']>start_summer) &
                                    (fire_events['end_time']<end_summer)]
    gdf_summer       = gdf[(gdf['timestamp']>start_summer) &
                                    (gdf['timestamp']<end_summer)]
    '''

    coords = np.vstack([fire_events.geometry.x, fire_events.geometry.y])

# Perform Gaussian KDE on the coordinates
    kde = gaussian_kde(coords)

    longitude=np.arange(-10,40.1,0.1)
    latitude=np.arange(33,62.1,0.1)

    # Define the transformer: from EPSG 4326 to EPSG 3035
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    # Sample 1D arrays of latitude and longitude
    # Convert coordinates from 4326 to 3035
    lats, lons = np.meshgrid(latitude, longitude)
    xx, yy = transformer.transform(lons.flatten(), lats.flatten())

    xxmin,xxmax = xx.min(), xx.max()
    yymin,yymax = yy.min(), yy.max()

    xx = np.arange(xxmin,xxmax,1.e4)
    yy = np.arange(yymin,yymax,1.e4)

# Create a grid over the area based on the bounding box of the points
    y, x = np.meshgrid(yy, xx)
    grid_coords = np.vstack([x.ravel(), y.ravel()])

# Evaluate the KDE on the grid points to get density values
    z = kde(grid_coords).reshape(x.shape)

    da = xr.DataArray(z.T ,
                      coords={
                          'y': ('y', y[0,:]),
                          'x': ('x', x[:,0])
                      },
                      dims=('y', 'x'))
    da = da.rio.write_crs("EPSG:3035", inplace=True)
    da.name = 'density probility function of fire event for the summer period'
    
    da = da.rio.reproject("EPSG:4326",nodata=-999)

    fig, ax = plt.subplots(figsize=(10, 8),subplot_kw={'projection': ccrs.PlateCarree()})
    da.where(da>1.e-14).plot(ax=ax,cbar_kwargs={'orientation': 'horizontal'},vmin=1.e-14,vmax=1e-12)
    fire_events.to_crs('epsg:4326').plot(ax=ax,markersize=1,color='r',alpha=0.2)

    ax.set_xlim(longitude.min(),longitude.max())
    ax.set_ylim(latitude.min(),latitude.max())

    ax.add_feature(cfeature.BORDERS, linestyle='-', color='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.LAND,)# facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Add gridlines
    ax.gridlines(draw_labels=True)

    plt.title('{:4d} \n number of fire events={:d}'.format(year,fire_events.shape[0]), pad=40)

    fig.savefig('./fireEventDensity_{:4d}.png'.format(year))
    plt.close(fig)
