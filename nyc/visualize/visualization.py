import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt

tractsshp = None
# tractmapping = None

# census tract shape files downloaded from: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html

import copy

import geoplot.crs as gcrs

def plot_map_of_colors(
    df,
    tract_col="census_tract",
    hue_col="lambdaval",
    shpfilename="visualize/viz_data/cb_2018_36_tract_500k.shp",
    # tractmappingfilename="data_clean/SR_to_censustract.csv",
    filter_tracts="visualize/viz_data/censustractlist",
    vmax=None,vmin = None,
    do_percentile = False
):
    global tractsshp
    if tractsshp is None:
        tractsshp = gpd.read_file(shpfilename)
    # if not tractmapping:
    #     tractmapping = pd.read_csv(tractmappingfilename)

    dfloc = copy.copy(df.replace([np.inf, -np.inf]).dropna(subset=[hue_col]))

    if vmax is None:
        vmax = dfloc[hue_col].max()
        print(vmax)
    if vmin is None:
        vmin = dfloc[hue_col].min()
        print(vmin)
    tractsshp.loc[:, tract_col] = tractsshp["GEOID"]
    dfloc.loc[:, tract_col] = dfloc[tract_col].astype(
        str)  # .apply(lambda x: str(x)[0:11])
    dflocjoin = dfloc.merge(tractsshp, on=tract_col, how="left")

    dflocjoin.loc[:, hue_col] = dflocjoin[hue_col].apply(
        lambda x: max(min(vmax, x), vmin))

    if do_percentile:
        dflocjoin.loc[:, hue_col] = dflocjoin[hue_col].rank(pct = True)

    # return dflocjoin
    # dflocjoin = dflocjoin.dropna(subset=["GEOID", hue_col])
    # return dflocjoin, tractsshp
    gplt.choropleth(gpd.GeoDataFrame(dflocjoin),
                    hue=hue_col, legend=True, linewidth=0, legend_kwargs={'shrink': 0.7}, projection = gcrs.AlbersEqualArea())


def plot_lambda_estimates_catplot(
    df, x="Category", y="EstLambda", hue="Borough", filter_categories=True, categories_to_not_plot=["Pest/Disease", "Planting Space", "Remove Stump"]
):
    if filter_categories:
        plotdf = df[~df.Category.isin(categories_to_not_plot)]
    else:
        plotdf = df
    sns.catplot(x=x, y=y, hue=hue, data=plotdf, kind="bar", legend=False)
    plt.xticks(rotation=90)
    plt.legend(loc="upper right", frameon=False, fontsize=15)
    plt.ylabel("Reporting rate", fontsize=20)
