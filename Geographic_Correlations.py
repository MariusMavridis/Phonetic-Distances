#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import numpy as np
from collections import defaultdict
from geopy import distance
import dcor
import mantel
import random
from scipy.optimize import curve_fit as cf
import plotly.express as px
from tqdm import tqdm
from cmcrameri import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature


path_to_dist = 'Alldist.txt' # path to file with all the phonetic distances between languages (file Alldist.txt in this repository)

# load wals database
languages_df = pd.read_csv('wals_languages.csv')

# List all languages
Lg_codes ={ 'af' : 'Afrikaans',
'sq' : 'Albanian',
'am' : 'Amharic',
'ar' : 'Arabic (Modern Standard)',
'eu' : 'Basque',
'bg' : 'Bulgarian',
'cs' : 'Czech',
'nl' : 'Dutch',
'et' : 'Estonian',
'en-gb' : 'English',
'fa' : 'Persian',
'fi' : 'Finnish',
'fr-fr' : 'French',
'de' : 'German',
'el' : 'Greek (Modern)',
'gu' : 'Gujarati',
'hi' : 'Hindi',
'hu' : 'Hungarian',
'id' : 'Indonesian',
'it' : 'Italian',
'kn' : 'Kannada',
'lv' : 'Latvian',
'lt' : 'Lithuanian',
'la' : 'Latin',
'ml' : 'Malayalam',
'mr' : 'Marathi',
'ne' : 'Nepali',
'nb' : 'Norwegian',
'pl' : 'Polish',
'pt' : 'Portuguese',
'ro' : 'Romanian',
'ru' : 'Russian',
'sr' : 'Serbian-Croatian',
'sk' : 'Slovak',
'sl' : 'Slovene',
'sv' : 'Swedish',
'es' : 'Spanish',
'te' : 'Telugu',
'tr' : 'Turkish',
'uk' : 'Ukrainian',
'az' : 'Azerbaijani',
'ava-Cyrl' : 'Avar',
'bxk-Latn' : 'Bukusu',
'hy' : 'Armenian (Eastern)', 
'hyw' : 'Armenian (Western)',
'ba' : 'Bashkir',
'cu' : 'Chuvash',
'be' : 'Belorussian',
'bn' : 'Bengali',
'bs' : 'Bosnian',
'ca' : 'Catalan',
'gd' : 'Gaelic (Scots)',
'ka' : 'Georgian',
'kk' : 'Kazakh', 
'ky' : 'Kirghiz',
'ltg' : 'Latgalian', # not in WALS
'nog' : 'Noghay',
'om' : 'Oromo (Boraana)',
'sd' : 'Sindhi',
'si' : 'Sinhala',
'ta' : 'Tamil',
'tk' : 'Turkmen',
'tt' : 'Tatar',
'ug' : 'Uyghur',
'cy' : 'Welsh',
'ms' : 'Malay',
'aar-Latn' : 'Qafar',
'tgk-Cyrl' : 'Tajik'}


# define colors
Families = {'Indo-European': None,
 'Afro-Asiatic': None,
 'Nakh-Daghestanian': None,
 'Altaic': None,
 'Basque': None,
 'Niger-Congo': None,
 'Austronesian': None,
 'Uralic': None,
 'Kartvelian': None,
 'Dravidian': None,
 }


# get studied languages from WALS
filtered_df = languages_df[languages_df['Name'].isin(list(Lg_codes.values()))]
Coord = {}

# Group them by family
grouped = defaultdict(list)
for _, row in filtered_df.iterrows():
    if not 'genus' in row['ID'] and not 'family' in row['ID'] and row['Name'] != 'Papiamentu':
        grouped[row["Family"]].append(row['Name'])
        Coord[row['Name']] = [row['Latitude'], row['Longitude']]
grouped["Indo-European"].append('Latgalian') # not in WALS database
Coord["Latgalian"] = [56,27]

# load phonetic distances
distances = {}

with open(path_to_dist) as f:
    d = f.readlines()
    for line in d:
        line = line.split(' ')
        lg1, lg2 = line[0], line[1]
        if not any(lg in ['af', 'la', 'pap'] for lg in [lg1, lg2]): # remove Afrikaans for geographic correlations and exclude Latin and Papiamentu because not in final sample
            distances[(Lg_codes[lg1], Lg_codes[lg2])] = float(line[3][:-2])


# compute geographic distances
Geodist = {(l1, l2) : distance.distance(Coord[l1], Coord[l2]).km for l1, l2 in distances.keys()}

Geodist_IE = {(l1,l2) : Geodist[(l1,l2)] for l1,l2 in Geodist if l1 in grouped["Indo-European"] and l2 in grouped["Indo-European"]}
distances_IE = {(l1,l2) : distances[(l1,l2)] for l1,l2 in Geodist_IE}

Coord_IE = {name : coord for name, coord in Coord.items() if name in grouped["Indo-European"]}

# #### Correlation coefficients

corr_all = dcor.distance_correlation(np.array(list(Geodist.values())), np.array(list(distances.values())))
print(f"Correlation between phonetic and geographic distance for all languages: {corr_all}")


corr_IE = dcor.distance_correlation(np.array(list(Geodist_IE.values())), np.array(list(distances_IE.values())))
print(f"Correlation between phonetic and geographic distance for IE languages: {corr_IE}")


# #### Mantel test

mantel.test(list(Geodist.values()), list(distances.values()))
mantel.test(list(Geodist_IE.values()), list(distances_IE.values()))


# #### Plots

plt.scatter(Geodist.values(), distances.values(), s = 3)
plt.xscale('log')
plt.rcParams["font.family"] = "serif"
plt.xlabel('Geographic distance (km)')
plt.ylabel('Phonetic distance')
plt.title('Phonetic (Wasserstein) vs geographic distance for all languages ')

plt.show()


plt.scatter(Geodist_IE.values(), distances_IE.values(), s = 3)
plt.xscale('log')
plt.xlabel('Geographic distance (km)')
plt.ylabel('Phonetic distance')
plt.title('Phonetic (Wasserstein) vs geographic distance for IE languages ')
plt.show()



df = pd.DataFrame({
    "Geographic distance (km)": list(Geodist_IE.values()),
    "Phonetic distance": list(distances_IE.values()),
    "label": list(Geodist_IE.keys())
})

fig = px.scatter(df, x= "Geographic distance (km)", y="Phonetic distance", hover_name="label", log_y = False, log_x = True, height = 700, width = 800)
fig.update_traces(marker=dict(size=8), textposition="top center")
fig.show()



# Logarithmic model for phonetic distance as a function of geographic distance

Lats1 = np.array([Coord[k[0]][0] for k in Geodist_IE])
Lats2 = np.array([Coord[k[1]][0] for k in Geodist_IE])
Longs1 = np.array([Coord[k[0]][1] for k in Geodist_IE])
Longs2 = np.array([Coord[k[1]][1] for k in Geodist_IE])

Lats1all = np.array([Coord[k[0]][0] for k in Geodist])
Lats2all = np.array([Coord[k[1]][0] for k in Geodist])
Longs1all = np.array([Coord[k[0]][1] for k in Geodist])
Longs2all = np.array([Coord[k[1]][1] for k in Geodist])

def Modified_Distance(coords):
    # coords should be a len 4 tuple (la1, lo1, la2, lo2)
    # where la1/la2 is the list of latitudes of every 1st/2nd language in all language pairs
    # same for lo1/lo2 (longitudes)
    la1, lo1, la2, lo2 = coords
    return [distance.distance((la1[i], lo1[i]),(la2[i], lo2[i])).km for i in range(len(la1))]
    
def Model1(coords, a, b):
    return np.log(Modified_Distance(coords)) * a + b

def logmodel(x, a, b):
    return np.log(x) * a + b

poptwsIE, pcovwsIE = cf(Model1, (Lats1, Longs1, Lats2, Longs2), list(distances_IE.values()))
poptwsall, pcovwsall = cf(Model1, (Lats1all, Longs1all, Lats2all, Longs2all), list(distances.values()))




plt.figure(figsize = (8,6))
Phdist = np.array(list(distances_IE.values()))
xdata = list(Geodist_IE.values())
yfit = np.array(logmodel(xdata, poptwsIE[0], poptwsIE[1]))

# Calculate phondist mean
phdist_mean = np.mean(Phdist)
print(f"phdist_mean: {phdist_mean:.3f}")

# Calculate total sum of squares, ss_tot
deviation_squared = (Phdist - phdist_mean)**2
ss_tot = np.sum(deviation_squared)
print(f"ss_tot: {ss_tot:.3f}")

# Calculate residual sum of squares, ss_res
error_squared = (Phdist - yfit)**2
ss_res = np.sum(error_squared)
print(f"ss_res: {ss_res:.3f}")

# Calculate R squared
r_squared = 1 - ss_res / ss_tot
print(f"R squared: {r_squared:.4f}")
plt.plot(np.linspace(np.min(xdata),np.max(xdata),1000), logmodel(np.linspace(np.min(xdata),np.max(xdata),1000), poptwsIE[0], poptwsIE[1]), color = 'black', label = f'${round(poptwsIE[0],3)}\ln(d) + {round(poptwsIE[1],3)}, R^2 = {round(r_squared,3)}  $', linestyle = '-')

plt.scatter(Geodist_IE.values(), distances_IE.values(), s = 3)
plt.legend(fontsize = 14)
plt.ylabel('Phonetic distance (Wasserstein)', fontsize = 14)
plt.xlabel('$d_{geo}$ (km)', fontsize = 14)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'in', length = 6)


plt.xscale('log')
plt.show()


plt.figure(figsize = (8,6))
Phdist = np.array(list(distances.values()))
xdata = list(Geodist.values())
yfit = np.array(logmodel(xdata, poptwsall[0], poptwsall[1]))

# Calculate phondist mean
phdist_mean = np.mean(Phdist)
print(f"phdist_mean: {phdist_mean:.3f}")

# Calculate total sum of squares, ss_tot
deviation_squared = (Phdist - phdist_mean)**2
ss_tot = np.sum(deviation_squared)
print(f"ss_tot: {ss_tot:.3f}")

# Calculate residual sum of squares, ss_res
error_squared = (Phdist - yfit)**2
ss_res = np.sum(error_squared)
print(f"ss_res: {ss_res:.3f}")

# Calculate R squared
r_squared = 1 - ss_res / ss_tot
print(f"R squared: {r_squared:.4f}")
plt.plot(np.linspace(np.min(xdata),np.max(xdata),1000), logmodel(np.linspace(np.min(xdata),np.max(xdata),1000), poptwsall[0], poptwsall[1]), color = 'black', label = f'${round(poptwsall[0],3)}\ln(d) + {round(poptwsall[1],3)}, R^2 = {round(r_squared, 3)}  $', linestyle = '-')

plt.scatter(Geodist.values(), distances.values(), s = 3)
plt.legend(fontsize = 14)
plt.ylabel('Phonetic distance (Wasserstein)', fontsize = 14)
plt.xlabel('$d_{geo}$ (km)', fontsize = 14)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'in', length = 6)


plt.xscale('log')
plt.show()


path_to_avg_distances = 'AvgdistIE.txt' # path to file with distances to avg distrib

geod = Geod(ellps="WGS84")

def geodesic_grid_to_langs(lat_grid, lon_grid, lang_lats, lang_lons):
    # Returns array of shape (H, W, N) with geodesic distances in km
    H, W = lat_grid.shape
    N = len(lang_lats)
    dist_grid = np.empty((H, W, N))

    for k in range(N):
        _, _, dist = geod.inv(lon_grid, lat_grid, np.full((H, W), lang_lons[k]), np.full((H, W), lang_lats[k]))
        dist_grid[:, :, k] = dist / 1000.0        # m to km
    return dist_grid


# Minimize khi2, plot heatmap, and compute uncertainty region

def IEOriginmap():

    popt, _ = cf(logmodel, list(Geodist_IE.values()), list(distances_IE.values()))

    def Phon_to_geo(d_phon, a, b):
        return np.exp((d_phon - b) / a)

    with open('C:/Users/mariu/Downloads/AvgdistIE.txt') as f:
        r = f.readlines()
        PDist_to_avg = {}
        for line in r:
            line = line.split(':')
            PDist_to_avg[Lg_codes[line[0][2:-1]]] = float(line[1][:-2])
    GDist_to_avg = {lg : Phon_to_geo(PDist_to_avg[lg], popt[0], popt[1]) for lg in PDist_to_avg}
    GDist_to_avg = np.array([Phon_to_geo(PDist_to_avg[lg], popt[0], popt[1]) for lg in PDist_to_avg])  
    lglist = list(PDist_to_avg.keys())
    N = len(lglist)
    # Precompute the grid once 
    width, height = 95 * 2, 65 * 2
    lon = np.linspace(-10, 85, width)
    lat = np.linspace(5, 70, height)
    lon_grid, lat_grid = np.meshgrid(lon, lat)  

    # Precompute language coordinates as arrays
    lang_lats = np.array([Coord_IE[lg][0] for lg in lglist])  
    lang_lons = np.array([Coord_IE[lg][1] for lg in lglist]) 

    # Vectorized geodesic distance: grid point -> each language 
    print("Precomputing distance grid...")
    dist_grid = geodesic_grid_to_langs(lat_grid, lon_grid, lang_lats, lang_lons)
    diff2 = (dist_grid - GDist_to_avg[None, None, :])**2

    
    # Precompute (dist - GDist_to_avg)^2 
    diff2 = (dist_grid - GDist_to_avg[None, None, :])**2  

    T_K = [1] * N                 
   
    
    khi2_grid = diff2 @ T_K                     
    idx_lat, idx_lon = np.unravel_index(khi2_grid.argmin(), khi2_grid.shape)

    lat_min, lon_min = lat[idx_lat], lon[idx_lon]  


    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    
    # remove sea cells
    land_shp = shapereader.natural_earth(
        resolution='110m',
        category='physical',
        name='land'
    )
    
    land_geom = list(shapereader.Reader(land_shp).geometries())
    
    # Merge all lands
    land_union = unary_union(land_geom)
    
    land = prep(land_union)
    # create mask
    data_ = khi2_grid
    mask = np.zeros_like(data_, dtype=bool)
    
    for i in range(len(lat)):
        for j in range(len(lon)):
            point = Point(lon[j], lat[i])
            if land.contains(point):
                mask[i, j] = True
    
    
    
    # Apply mask to remove sea
    masked_data = np.where(mask, np.log(data_), np.nan)



    # Plot
    img = ax.pcolormesh(
        lon, lat,
        masked_data,
        cmap=cm.buda.reversed(),
        shading='auto',
        transform=ccrs.PlateCarree(),
        edgecolors='none',   
        linewidth=0,         
        rasterized=True      
    ) 
    
    K = 2000
    
    T_K = dirichlet.rvs([1] * N, size=K)                 
    min_coords = [] # list of r* values 
    
    for m in tqdm(range(K)):
        weights = T_K[m]                                  
        khi2_grid = diff2 @ weights                        
        idx_lat, idx_lon = np.unravel_index(khi2_grid.argmin(), khi2_grid.shape)
        min_coords.append((lat[idx_lat], lon[idx_lon]))   
    dist_diric_real = [distance.distance([lat_min, lon_min], [lat, lon]).km for lat, lon in min_coords]
    R = np.percentile(dist_diric_real, 95) # 95% radius around r*
            
    
    def plot_circle_km(ax, lon_center, lat_center, radius_km, **kwargs):
        """Plot a circle of radius_km around (lon_center, lat_center) on a cartopy map."""
        angles = np.linspace(0, 2 * np.pi, 360)
        R_earth = 6371.0  # km
    
        lat_center_rad = np.radians(lat_center)
        lon_center_rad = np.radians(lon_center)
        d = radius_km / R_earth  # angular distance in radians
    
        lats = np.degrees(np.arcsin(
            np.sin(lat_center_rad) * np.cos(d) +
            np.cos(lat_center_rad) * np.sin(d) * np.cos(angles)
        ))
        lons = lon_center + np.degrees(np.arctan2(
            np.sin(angles) * np.sin(d) * np.cos(lat_center_rad),
            np.cos(d) - np.sin(lat_center_rad) * np.sin(np.radians(lats))
        ))
    
        ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)
    
    ax.coastlines()
    ax.plot(lon_min, lat_min, 'r*', ms = 10)
    
    norm = mpl.colors.Normalize(vmin=np.nanmin(masked_data),
                            vmax=np.nanmax(masked_data))

    cb = plt.colorbar(img, ax=ax, norm=norm, orientation='vertical')
    cb.set_label(label='$\log(\chi^2)$', fontsize=18, labelpad=15)
    cb.ax.tick_params(labelsize='x-large')
    if R:
        plot_circle_km(ax, lon_min, lat_min, R, color='red', linestyle = '--', linewidth = 1.5, zorder = 5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none', linewidth = 2)
    plt.show()  

IEOriginmap()



# ### Permutation test


def ShuffleCoord(dist, n, ref_dcor): # dist is either distances or distances_IE
    keys = list(dist.keys())
    dist_values = np.array(list(dist.values()))
    print(len(keys),len(dist_values))
    
    coord_keys = list({lg for pair in keys for lg in pair})
    coord_values = [Coord[lg] for lg in coord_keys]
    n_langs = len(coord_keys)

    Dcors = np.empty(n)

    for i in tqdm(range(n)):
        # Shuffle coordinates
        shuffled_indices = np.random.permutation(n_langs)
        SCoord = dict(zip(coord_keys, [coord_values[j] for j in shuffled_indices]))

        # Compute geographic distances
        geo_dist = np.array([
            distance.distance(SCoord[l1], SCoord[l2]).km
            for l1, l2 in keys
        ])

        Dcors[i] = dcor.distance_correlation(geo_dist, dist_values)

    p_value = np.sum(Dcors >= ref_dcor) / n
    return p_value    




p_value = ShuffleCoord(distances_IE, 1000, 0.496) # takes a while to run
