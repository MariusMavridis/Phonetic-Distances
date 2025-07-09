#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
import networkx as nx
import math
from pyvis.network import Network


# In[5]:


path_to_dist = '' # path to file with all the phonetic distances between languages (file Alldist.txt in data repository)

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


# In[6]:


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


# In[7]:


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
        if 'af' not in [lg1, lg2]: # remove Afrikaans for geographic correlations
            distances[(Lg_codes[lg1], Lg_codes[lg2])] = float(line[3][:-2])




# In[8]:


# compute geographic distances
Geodist = {(l1, l2) : distance.distance(Coord[l1], Coord[l2]).km for l1, l2 in distances.keys()}

Geodist_IE = {(l1,l2) : Geodist[(l1,l2)] for l1,l2 in Geodist if l1 in grouped["Indo-European"] and l2 in grouped["Indo-European"]}
distances_IE = {(l1,l2) : distances[(l1,l2)] for l1,l2 in Geodist_IE}


# #### Correlation coefficients

# In[9]:


corr_all = dcor.distance_correlation(np.array(list(Geodist.values())), np.array(list(distances.values())))


# In[10]:


print(f"Correlation between phonetic and geographic distance for all languages: {corr_all}")


# In[11]:


corr_IE = dcor.distance_correlation(np.array(list(Geodist_IE.values())), np.array(list(distances_IE.values())))


# In[12]:


print(f"Correlation between phonetic and geographic distance for IE languages: {corr_IE}")


# #### Mantel test

# In[13]:


mantel.test(list(Geodist.values()), list(distances.values()))


# In[14]:


mantel.test(list(Geodist_IE.values()), list(distances_IE.values()))


# #### Plots

# In[15]:


plt.scatter(Geodist.values(), distances.values(), s = 3)
plt.xscale('log')
plt.rcParams["font.family"] = "serif"
plt.xlabel('Geographic distance (km)')
plt.ylabel('Phonetic distance')
plt.title('Phonetic (Wasserstein) vs geographic distance for all languages ')

plt.show()


# In[16]:


plt.scatter(Geodist_IE.values(), distances_IE.values(), s = 3)
plt.xscale('log')
plt.xlabel('Geographic distance (km)')
plt.ylabel('Phonetic distance')
plt.title('Phonetic (Wasserstein) vs geographic distance for IE languages ')
plt.show()


# In[17]:


df = pd.DataFrame({
    "Geographic distance (km)": list(Geodist_IE.values()),
    "Phonetic distance": list(distances_IE.values()),
    "label": list(Geodist_IE.keys())
})

fig = px.scatter(df, x= "Geographic distance (km)", y="Phonetic distance", hover_name="label", log_y = False, log_x = True, height = 700, width = 800)
fig.update_traces(marker=dict(size=8), textposition="top center")
fig.show()


# In[18]:


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

def affine1(x, a, b):
    return np.log(x) * a + b

poptwsIE, pcovwsIE = cf(Model1, (Lats1, Longs1, Lats2, Longs2), list(distances_IE.values()))
poptwsall, pcovwsall = cf(Model1, (Lats1all, Longs1all, Lats2all, Longs2all), list(distances.values()))




# In[19]:


plt.figure(figsize = (8,6))
Phdist = np.array(list(distances_IE.values()))
xdata = list(Geodist_IE.values())
yfit = np.array(affine1(xdata, poptwsIE[0], poptwsIE[1]))

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
plt.plot(np.linspace(np.min(xdata),np.max(xdata),1000), affine1(np.linspace(np.min(xdata),np.max(xdata),1000), poptwsIE[0], poptwsIE[1]), color = 'black', label = f'${round(poptwsIE[0],3)}\ln(d) + {round(poptwsIE[1],3)}, R^2 = {round(r_squared,3)}  $', linestyle = '-')

plt.scatter(Geodist_IE.values(), distances_IE.values(), s = 3)
plt.legend(fontsize = 14)
plt.ylabel('Phonetic distance (Wasserstein)', fontsize = 14)
plt.xlabel('$d_{geo}$ (km)', fontsize = 14)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'in', length = 6)


plt.xscale('log')
plt.show()


# In[20]:


plt.figure(figsize = (8,6))
Phdist = np.array(list(distances.values()))
xdata = list(Geodist.values())
yfit = np.array(affine1(xdata, poptwsall[0], poptwsall[1]))

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
plt.plot(np.linspace(np.min(xdata),np.max(xdata),1000), affine1(np.linspace(np.min(xdata),np.max(xdata),1000), poptwsall[0], poptwsall[1]), color = 'black', label = f'${round(poptwsall[0],3)}\ln(d) + {round(poptwsall[1],3)}, R^2 = {round(r_squared, 3)}  $', linestyle = '-')

plt.scatter(Geodist.values(), distances.values(), s = 3)
plt.legend(fontsize = 14)
plt.ylabel('Phonetic distance (Wasserstein)', fontsize = 14)
plt.xlabel('$d_{geo}$ (km)', fontsize = 14)
plt.tick_params(axis = 'both', which = 'major', labelsize = 14, direction = 'in', length = 6)


plt.xscale('log')
plt.show()


# ## Origin of IE family

# In[21]:


path_to_avg_distances = '/home/mavridis/Documents/Marius/code/out/Distances/WS/AvgdistIE.txt' # path to file with distances to avg distru


# In[22]:


def Phon_to_geo(d_phon, a, b):
    return np.exp((d_phon-b)/a)
    
PDist_to_avg = {}
with open(path_to_avg_distances) as f:
    r = f.readlines()
    for line in r:
        line = line.split(':')
        PDist_to_avg[Lg_codes[line[0][2:-1]]] = float(line[1][:-2])
GDist_to_avg = {lg : Phon_to_geo(PDist_to_avg[lg], poptwsIE[0], poptwsIE[1]) for lg in PDist_to_avg}


# In[23]:


def khi2(lat, lon):
    return sum([(distance.distance([lat, lon],Coord[lg]).km - GDist_to_avg[lg])**2 for lg in GDist_to_avg])
    


# In[18]:


# Grid dimensions (pixels)
width, height =95, 65

# Latitude and longitude grids
lon = np.linspace(-10, 85, width)
lat = np.linspace(5, 70, height)
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Color function
def color_function(lat, lon):
    return khi2(lat,lon)

# Compute value for each point
data_sq = [[color_function(lat_grid[i][j], lon_grid[i][j]) for i in range(65)] for j in tqdm(range(95))]


# In[19]:


# Plot heatmap of khi2 values
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

img = ax.pcolormesh(lon, lat, np.log(np.transpose(data_sq)), cmap=cm.lipari, shading='auto', transform=ccrs.PlateCarree())
norm = mcolors.Normalize(vmin = np.min(data_sq), vmax = np.max(data_sq))

cb = plt.colorbar(img, ax=ax, norm=norm, orientation='vertical') 
cb.set_label(label='$\log(\chi^2)$', fontsize = 18, labelpad = 15)
cb.ax.tick_params(labelsize = 'x-large')
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')

plt.show()


# ### Permutation test

# In[34]:


def ShuffleCoord(n, calc_khi2):
    # performs n random shuffles of language coordinates and computes correlation coefficient between phonetic geographic distances
    # outputs:
    # lowest_coord: point that minimize khi2 for each random shuffle
    # minvals: minimum value of khi2 for each shuffle (associated to the point in lowest_coord)
    # Dcors: correlation coefficients for each shuffle
    
    lowest_coord = []
    minvals = []
    Dcors = []

    for i in tqdm(range(n)):
        
        # Shuffle coordinates
        C = list(Coord.values())
        random.shuffle(C)
        SCoord = {list(Coord.keys())[i] : C[i] for i in range(len(C)) }
        
        # Compute geographic distances 
        Geodist = {}
        for lang1, lang2 in distances_IE.keys():
            Geodist[lang1, lang2] = distance.distance(SCoord[lang1], SCoord[lang2]).km

        # Compute correlation coefficient
        Dcors.append(dcor.distance_correlation(np.array(list(Geodist.values())), np.array(list(distances_IE.values()))))

        # optional (but much longer): compute khi2 values for this random shuffle
        if calc_khi2:
            # fit a log curve
            popt, pcov = cf(affine1, list(Geodist.values()), list(distances_IE.values()))
    
            # convert the phonetic distances into geographic distances
            GDist_to_avg = {lg : Phon_to_geo(PDist_to_avg[lg], popt[0], popt[1]) for lg in PDist_to_avg}
    
            # define sum of residuals
            def khi2(lat, lon): 
                s = 0
                for lg in GDist_to_avg:
                    s += (distance.distance([lat, lon],SCoord[lg]).km - GDist_to_avg[lg])**2
                return np.log(s)
        
            # Grid dimensions (pixels)
            width, height = 95, 65
            
            # Latitude/longitude grids
            lon = np.linspace(-10, 85, width)
            lat = np.linspace(5, 70, height)
            lon_grid, lat_grid = np.meshgrid(lon, lat)
        
            
            # Compute khi2 for each grid point
            data_sq = [[khi2(lat_grid[i][j], lon_grid[i][j]) for i in range(65)] for j in range(95)]
            data = np.array(data_sq)
    
            # get smallest khi2 value and its coords
            lowest_coord.append(np.unravel_index(data.argmin(), data.shape))
            minvals.append(np.min(data))
            
            ind = [(i,val) for i,val in enumerate(data.flatten())]
        minvals = np.array(minvals)
    
        lowest_coord = np.array(lowest_coord)
    
    return(lowest_coord, minvals, Dcors)      


# In[35]:


_,_,Dcors = ShuffleCoord(1000, False) 


# In[33]:


p_value = ((np.sum(Dcors) >= corr_IE) + 1)/(len(Dcors) + 1)


# ### Genealogical distance

# In[20]:


Lg_names_IE = list(PDist_to_avg.keys())


# In[46]:


# Define edges of IE family tree (links between languages)
edges = [
    ("Proto-Indo-European", "Romance"),
    ("Romance", "Italo-Dalmatian"),
    ("Romance", "Western Romance"),
    ("Romance", "Eastern Romance"),
    ("Eastern Romance", "Romanian"),
    ("Gallo-Romance", "Catalan"),
    ("Gallo-Romance", "French"),
    ("Western Romance", "Iberian"),
    ("Western Romance", "Gallo-Romance"),
    ("Iberian", "Spanish"),
    ("Iberian", "Portuguese"),
    ("Italo-Dalmatian", "Italian"),

    ("Proto-Indo-European", "Germanic"),
    ("Germanic", "North Germanic"),
    ("North Germanic", "Swedish"),
    ("North Germanic", "Norwegian"),
    ("Germanic", "West Germanic"),
    ("West Germanic", "English"),
    ("West Germanic", "Dutch"),
    ("West Germanic", "German"),
   
    ("Proto-Indo-European", "Indo-Iranian"),
    ("Indo-Iranian", "Indic"),
    ("Indic", "Western Indo-Aryan"),
    ("Indic", "Southern Indo-Aryan"),
    ("Indic", "Eastern Indo-Aryan"),
    ("Western Indo-Aryan", "Hindi"),
    ("Western Indo-Aryan", "Nepali"),
    ("Western Indo-Aryan", "Gujarati"),
    ("Western Indo-Aryan", "Sindhi"),
    ("Southern Indo-Aryan", "Marathi"),
    ("Southern Indo-Aryan", "Sinhala"),
    ("Eastern Indo-Aryan", "Bengali"),
    ("Indo-Iranian", "Iranian"),
    ("Iranian", "Western Iranian"),
    ("Western Iranian", "Tajik"),
    ("Western Iranian", "Persian"),
    

    ("Proto-Indo-European", "Celtic"),
    ("Celtic", "Insular Celtic"),
    ("Insular Celtic", "Goidelic"),
    ("Insular Celtic", "Brittonic"),
    ("Goidelic", "Gaelic (Scots)"),
    ("Brittonic", "Welsh"),
    
    ("Proto-Indo-European", "Hellenic"),
    ("Hellenic", "Greek (Modern)"),

    ("Proto-Indo-European", "Albanoid"),
    ("Albanoid", "Albanian"),
    ("Albanian", "Tosk"),

    ("Proto-Indo-European", "Armenian"),
    ("Armenian", "Armenian (Western)"),
    ("Armenian", "Armenian (Eastern)"),

    ("Proto-Indo-European", "Balto-Slavic"),
    ("Balto-Slavic", "Baltic"),
    ("Baltic", "Lat"),
    ("Baltic", "Lit"),
    ("Lit", "Lithuanian"),
    ("Lat", "Latvian"),
    ("Lat", "Latgalian"),
    ("Balto-Slavic", "Slavic"),
    ("Slavic", "South Slavic"),
    ("South Slavic", "Bulgarian"),
    ("South Slavic", "Serbian-Croatian"),
    ("South Slavic", "Slovene"),
    ("South Slavic", "Bosnian"),
    ("Slavic", "West Slavic"),
    ("West Slavic", "Czech"),
    ("West Slavic", "Slovak"),
    ("West Slavic", "Polish"),
    ("Slavic", "East Slavic"),
    ("East Slavic", "Belorussian"),
    ("East Slavic", "Ukrainian"),
    ("East Slavic", "Russian"),
]


# Create graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Create pyvis network
net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
net.barnes_hut()  # Meilleure stabilisation

# Add nodes and edges
for node in G.nodes:
    net.add_node(node, label=node, title=node, shape="dot", size=10)

for source, target in G.edges:
    net.add_edge(source, target)

# Display options
net.set_options('''
var options = {
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "directed"
    }
  },
  "physics": {
    "hierarchicalRepulsion": {
      "nodeDistance": 120
    },
    "solver": "hierarchicalRepulsion"
  }
}
''')

# Export html
# net.write_html("ie_language_tree.html")



# In[26]:


def distance_between_leaves(G, leaf1, leaf2, root):
    # path from root
    try:
        path1 = nx.shortest_path(G, source=root, target=leaf1)
        path2 = nx.shortest_path(G, source=root, target=leaf2)
    except nx.NetworkXNoPath:
        return None

    # find last common ancestor
    min_len = min(len(path1), len(path2))
    lca_index = 0
    for i in range(min_len):
        if path1[i] == path2[i]:
            lca_index = i
        else:
            break

    lca = path1[lca_index]
    d1 = len(path1) - lca_index - 1
    d2 = len(path2) - lca_index - 1
    return d1 + d2


# In[42]:


Genealo_dist = {(l1, l2) : distance_between_leaves(G, l1, l2, "Proto-Indo-European") for l1, l2 in distances_IE}


# In[41]:


d2 = [distances_IE[(l1, l2)] for l1, l2 in distances_IE if Genealo_dist[(l1, l2)] == 2]
d4 = [distances_IE[(l1, l2)] for l1, l2 in distances_IE if Genealo_dist[(l1, l2)] == 4]
d5 = [distances_IE[(l1, l2)] for l1, l2 in distances_IE if Genealo_dist[(l1, l2)] == 5]
d6 = [distances_IE[(l1, l2)] for l1, l2 in distances_IE if Genealo_dist[(l1, l2)] == 6]
d7 = [distances_IE[(l1, l2)] for l1, l2 in distances_IE if Genealo_dist[(l1, l2)] == 7]
d8 = [distances_IE[(l1, l2)] for l1, l2 in distances_IE if Genealo_dist[(l1, l2)] == 8]
D = [d2,[], d4, d5, d6, d7, d8]
plt.boxplot(D, tick_labels = [2,3,4,5,6,7,8], notch = True)
plt.xlabel('Genealogical distance', fontsize = 14)
plt.ylabel('Phonetic distance', fontsize = 14)

plt.show()

