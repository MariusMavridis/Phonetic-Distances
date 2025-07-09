#!/usr/bin/env python
#/home/mavridis/Documents/Marius/Jobs/TocalcnonIE/Group0.py
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from tqdm import tqdm
import re
import csv
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import os
import scipy
import panphon
import panphon.distance
from scipy.spatial.distance import jensenshannon as js
import ot
import os.path
import numpy as np
import regex as re
import pkg_resources
from panphon import _panphon, permissive, featuretable, xsampa
from cmcrameri import cm as cramer
import seaborn
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

ft = panphon.FeatureTable()

path_to_IPA = ""             # path to IPA Bibles
path_to_distrib = ""         # path to probability distribution files 
path_to_avg_distrib = ""     # path to the txt file built in Proba_Distrib_Memory_Estimation.py (path_to_save_avg)


Language_codes = {
'af' : 'Afrikaans',
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
'aar-Latn' : 'Qafar',
'ava-Cyrl' : 'Avar',
'tgk-Cyrl' : 'Tajik',
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
'ms' : 'Malay'
}

IE_languages = ['af', 'sq', 'bg', 'cs', 'nl', 'en-gb','fr-fr', 'fa', 'de', 'el', 'gu', 'hi', 'it', 'lv', 'lt', 'mr', 'ne', 'nb', 'pl', 'pt', 'ro',
'ru', 'sr', 'sk', 'sl', 'sv', 'es', 'uk', 'bn', 'tgk-Cyrl', 'hy', 'hyw', 'be', 'bs', 'ca', 'gd', 'ltg', 'sd', 'si', 'cy']

# load panphon distance module
dst = panphon.distance.Distance()
   
   
# define distances on the feature vector space 
# it is a slight modification of panphon original code so that the distance functions directly take feature vectors as input and not IPA strings
def zerodiviszero(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ZeroDivisionError:
            return 0
    return wrapper


def xsampaopt(f):
    def wrapper(*args, **kwargs):
        if 'xsampa' in kwargs and kwargs['xsampa']:
            self, source, target = args
            source = self.xs.convert(source)
            target = self.xs.convert(target)
            args = (self, source, target)
        return f(*args, **kwargs)
    return wrapper


class VectDistance(object):
    

    def __init__(self, feature_set='spe+', feature_model='segment'):
        
        fm = {'strict': _panphon.FeatureTable,
              'permissive': permissive.PermissiveFeatureTable,
              'segment': featuretable.FeatureTable}
        self.fm = fm[feature_model](feature_set=feature_set)
        self.xs = xsampa.XSampa()
        

    def min_edit_distance(self, del_cost, ins_cost, sub_cost, start, source, target):
        
        # Get lengths of source and target
        n, m = len(source), len(target)
        source, target = start + source, start + target
        # Create "matrix"
        d = []
        for i in range(n + 1):
            d.append((m + 1) * [None])
        # Initialize "matrix"
        d[0][0] = 0
        for i in range(1, n + 1):
            d[i][0] = d[i - 1][0] + del_cost(source[i])
        for j in range(1, m + 1):
            d[0][j] = d[0][j - 1] + ins_cost(target[j])
        # Recurrence relation
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i][j] = min([
                    d[i - 1][j] + del_cost(source[i]),
                    d[i - 1][j - 1] + sub_cost(source[i], target[j]),
                    d[i][j - 1] + ins_cost(target[j]),
                ])
        return d[n][m]

    def feature_difference(self, ft1, ft2):
        return abs(ft1 - ft2) / 2

    def unweighted_deletion_cost(self, v1, gl_wt=1.0):
        assert isinstance(v1, list)
        return sum(map(lambda x: 0.5 if x == 0 else 1, v1)) / len(v1) * gl_wt

    def unweighted_substitution_cost(self, v1, v2):
        return sum([abs(ft1 - ft2) / 2 for (ft1, ft2) in zip(v1, v2)]) / len(v1)

    def unweighted_insertion_cost(self, v1, gl_wt=1.0):
        return sum(map(lambda x: 0.5 if x == 0 else 1, v1)) / len(v1) * gl_wt

    @xsampaopt
    def feature_edit_distance(self, source, target, xsampa=False):
        return self.min_edit_distance(self.unweighted_deletion_cost,
                                      self.unweighted_insertion_cost,
                                      self.unweighted_substitution_cost,
                                      [[]],
                                      source,
                                      target)

   
    def weighted_feature_difference(self, w, ft1, ft2):
        return self.feature_difference(ft1, ft2) * w

    def weighted_substitution_cost(self, v1, v2, gl_wt=1.0):
        return sum([abs(ft1 - ft2) * w
                    for (w, ft1, ft2)
                    in zip(self.fm.weights, v1, v2)]) * gl_wt

    def weighted_insertion_cost(self, v1, gl_wt=1.0):
        assert isinstance(v1, list)
        return sum(self.fm.weights) * gl_wt

    def weighted_deletion_cost(self, v1, gl_wt=1.0):
        assert isinstance(v1, list)
        return sum(self.fm.weights) * gl_wt

    def weighted_feature_edit_distance(self, source, target, xsampa=False):
        return self.min_edit_distance(self.weighted_deletion_cost,
                                      self.weighted_insertion_cost,
                                      self.weighted_substitution_cost,
                                      [[]],
                                      source,
                                      target)
vdst = VectDistance()


def vect_distance(x, y, d):
    # returns the distance between feature vectors x and y
    # d is the distance function (feature edit distance 'fed' or weighted feature edit distance 'wfed')
    distances = {'fed' : vdst.feature_edit_distance, 'wfed' : vdst.weighted_feature_edit_distance}
    dist = distances[d]
    return dist(x,y)


def Wasserstein_distance_3grams(lg1, lg2, seq, d): 
    # computes the Wasserstein distance between 3gram proba distributions of languages lg1 and lg2
    # seq is a boolean for choosing whether to consider word boundaries
    # d is the distance on the space of 3-grams ('fed' for feature edit distance and 'wfed' for weighted feature edit distance)
    # for Wasserstein distance between average IE distrib and IE languages see below
    L1 = {}
    L2 = {}
    typeoftext = {True : '-grams_one_sequence_vect.txt', False : '-grams_sep_words.txt'}
    
    # load probability distributions of both languages (in terms of feature vectors)
    # Warning: the path below must point to a txt file built with the Proba_distrib_seq_vect function in Proba_Distrib_Memory_Estimation.py
    with open(path_to_distrib + Language_codes[lg1] + '_' + str(blocksize) + typeoftext[seq], encoding = 'utf-8') as f1:
        r1 = csv.reader(f1)
        Counts_l1 = {tuple([tuple([int(row[i]) for i in range(0,24)]), tuple([int(row[j]) for j in range(24,48)]), tuple([int(row[k]) for k in range(48,72)])]) : int(row[72]) for row in [truc for truc in r1][1:]}

    with open(path_to_distrib + Language_codes[lg2] + '_' + str(blocksize) + typeoftext[seq], encoding = 'utf-8') as f2:
        r2 = csv.reader(f2)
        Counts_l2 = {tuple([tuple([int(row[i]) for i in range(0,24)]), tuple([int(row[j]) for j in range(24,48)]), tuple([int(row[k]) for k in range(48,72)])]) : int(row[72]) for row in [truc for truc in r2][1:]}

    # convert tuples into lists and separate 3-grams and counts in two lists
    L1_values = [list(list(u) for u in t) for t in Counts_l1.keys()]
    L2_values = [list(list(u) for u in t) for t in Counts_l2.keys()]
    L1_weights = list(Counts_l1.values())
    L2_weights = list(Counts_l2.values())

    # normalize distributions
    u_weights, v_weights = np.array(L1_weights, dtype = np.float64), np.array(L2_weights, dtype = np.float64)
    u_weights /= np.sum(u_weights)
    v_weights /= np.sum(v_weights) 
    
    # Create cost matrix
    m, n = len(L1_values), len(L2_values)
    D = np.empty((m,n),dtype=float) 
    for i in tqdm(range(m)):
        for j in range(n):
            D[i,j] = vect_distance(L1_values[i],L2_values[j],d)
    
    reg = 0.005 # entropic regularization term for the Sinkhorn algorithm 
    # solve linear opptimization problem
    G = ot.sinkhorn(u_weights, v_weights, D, reg, numItermax = 30000)
    s = np.sum(G * D)
    
    return s


def Wasserstein_distance_3grams_avg(lg1, d): 
    # computes the Wasserstein distance between 3gram proba distribution of language lg1 and the average IE distribution
    # d is the distance on the space of 3-grams ('fed' for feature edit distance and 'wfed' for weighted feature edit distance)
    L1 = {} # IE language
    L2 = {} # Average IE distribution
    typeoftext = {True : '-grams_one_sequence_vect.txt', False : '-grams_sep_words.txt'}
    
    # load both probability distributions (in terms of feature vectors)
    # Warning: the path below must point to a txt file built with the Proba_distrib_seq_vect function in Proba_Distrib_Memory_Estimation.py
    with open(path_to_distrib + Language_codes[lg1] + '_' + str(blocksize) + typeoftext[seq], encoding = 'utf-8') as f1:
        r1 = csv.reader(f1)
        Counts_l1 = {tuple([tuple([int(row[i]) for i in range(0,24)]), tuple([int(row[j]) for j in range(24,48)]), tuple([int(row[k]) for k in range(48,72)])]) : int(row[72]) for row in [truc for truc in r1][1:]}

    with open(path_to_avg_distrib, encoding = 'utf-8') as f2:
        r2 = csv.reader(f2)
        Counts_l2 = {tuple([tuple([int(row[i]) for i in range(0,24)]), tuple([int(row[j]) for j in range(24,48)]), tuple([int(row[k]) for k in range(48,72)])]) : int(row[72]) for row in [truc for truc in r2][1:]}

    # convert tuples into lists and separate 3-grams and counts in two lists
    L1_values = [list(list(u) for u in t) for t in Counts_l1.keys()]
    L2_values = [list(list(u) for u in t) for t in Counts_l2.keys()]
    L1_weights = list(Counts_l1.values())
    L2_weights = list(Counts_l2.values())

    # normalize distributions
    u_weights, v_weights = np.array(L1_weights, dtype = np.float64), np.array(L2_weights, dtype = np.float64)
    u_weights /= np.sum(u_weights)
    v_weights /= np.sum(v_weights) 
    
    # Create cost matrix
    m, n = len(L1_values), len(L2_values)
    D = np.empty((m,n),dtype=float) 
    for i in tqdm(range(m)):
        for j in range(n):
            D[i,j] = vect_distance(L1_values[i],L2_values[j],d)

    # normalize distributions
    u_weights, v_weights = np.array(L1_weights, dtype = np.float64), np.array(L2_weights, dtype = np.float64)
    u_weights /= np.sum(u_weights)
    v_weights /= np.sum(v_weights) 
    
   
    reg = 0.005 # entropic regularization term for the Sinkhorn algorithm 
    # solve linear optimization problem
    G = ot.sinkhorn(u_weights, v_weights, D, reg, numItermax = 30000)
    s = np.sum(G * D)
    
    return s


### Plot Heatmaps

# all 67 languages
def Clustermap_all(path, method):

    # plots a clustermap of all Wasserstein distances

    # path should lead to a txt file with all Wasserstein distances 
    # format of file should be 
    # lg1 lg2 : distance
    # lg1 lg3 : distance
    # ...
    
    # method is the linkage method (see scipy.cluster.hierarchy.linkage documentation)
    # best methods here are "complete" and "ward"
    

    # open file with all the distances and write the data in a dictionary {(lg1, lg2) : distance(lg1, lg2)}
    Dmatrix = np.zeros((67,67))
    Listlg = []
    Diclg = {}
    L = []
    with open(path, 'r') as f:
        r = csv.reader(f)
        for line in r:
            l = line[0].split(" ")
            if not l[0] in Listlg:
                Listlg.append(l[0]) 
            if not l[1] in Listlg:
                Listlg.append(l[1])
            Diclg[(l[0],l[1])] = float(l[3])
            L.append([l[0],l[1],float(l[3])])
            
    # write the data in a distance matrix        
    for i in range(67):
        for j in range(i+1, 67):
            lg1 = Listlg[i]
            lg2 = Listlg[j]
            Dmatrix[i,j] =  Diclg[(lg1, lg2)]
            Dmatrix[j,i] = Diclg[(lg1, lg2)]    
    
    # put the matrix in condensed form
    condensed_dist_matrix = squareform(Dmatrix)
    print(f"Matrix shape: {condensed_dist_matrix.shape}")

    # apply hierarchical clustering to the condensed distance matrix
    labels = [Language_codes[x] for x in Listlg]
    linkag = linkage(condensed_dist_matrix, method = method)
    plt.figure(figsize=(18,18))
    cm = seaborn.clustermap(Dmatrix, method=method, cmap = cramer.lipari, xticklabels = labels, yticklabels = labels, row_linkage = linkag, col_linkage = linkag, cbar_kws = {'label' : 'Distance'})

    # clustermap options
    cm.ax_row_dendrogram.set_visible(False)
    x0, _y0, _w, _h = cm.cbar_pos
    cm.ax_cbar.set_position((x0+0.05, .149, .03, .67))
    cm.ax_cbar.set_ylabel("Distance", ha="center", va = "center", fontsize = 18, labelpad = 15)
    cm.cax.yaxis.set_label_position('left')
    cm.cax.tick_params(labelsize=18)
    cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(), fontsize = 8)
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), fontsize = 8)

# only IE languages
def Clustermap_IE(path, method):
    # plots a clustermap of Wasserstein distances between IE languages

    # path should lead to a txt file with all Wasserstein distances 
    # format of file should be 
    # lg1 lg2 : distance
    # lg1 lg3 : distance
    # ...
    
    # method is the linkage method (see scipy.cluster.hierarchy.linkage documentation)
    # best methods here are "complete" and "ward"
    

    # open file with all the distances, filter only IE languages, and write the data in a dictionary {(lg1, lg2) : distance(lg1, lg2)}
    Dmatrix = np.zeros((40,40))
    Listlg = []
    Diclg = {}
    L = []
    with open(path, 'r') as f:
        r = csv.reader(f)
        for line in r:
            l = line[0].split(" ")
            if not l[0] in Listlg and l[0] in IE_languages:
                Listlg.append(l[0])
            if not l[1] in Listlg and l[1] in IE_languages:
                Listlg.append(l[1])
            Diclg[(l[0],l[1])] = float(l[3])
            L.append([l[0],l[1],float(l[3])])
            
    # write the data in a distance matrix        
    for i in range(40):
        for j in range(i+1, 40):
            lg1 = Listlg[i]
            lg2 = Listlg[j]
            if (lg1, lg2) in Diclg:
                Dmatrix[i,j] =  Diclg[(lg1, lg2)]
                Dmatrix[j,i] = Diclg[(lg1, lg2)]   
            elif (lg2, lg1) in Diclg:
                Dmatrix[i,j] =  Diclg[(lg2, lg1)]
                Dmatrix[j,i] = Diclg[(lg2, lg1)]  
            
    # put the matrix in condensed form 
    condensed_dist_matrix = squareform(Dmatrix)
    print(f"Matrix shape: {condensed_dist_matrix.shape}")
    
    # apply hierarchical clustering to the condensed distance matrix
    labels = [Language_codes[x] for x in Listlg]
    linkag = linkage(condensed_dist_matrix, method = method)
    plt.figure(figsize=(14,14))
    cm = seaborn.clustermap(Dmatrix, method=method, cmap = cramer.lipari, xticklabels = labels, yticklabels = labels, row_linkage = linkag, col_linkage = linkag, cbar_kws = {'label' : 'Distance'})

    # clustermap options
    cm.ax_row_dendrogram.set_visible(False)
    x0, _y0, _w, _h = cm.cbar_pos
    cm.ax_cbar.set_position((x0+0.05, .149, .03, .67))
    cm.ax_cbar.set_ylabel("Distance", ha="center", va = "center", fontsize = 18, labelpad = 15)
    cm.cax.yaxis.set_label_position('left')
    cm.cax.tick_params(labelsize=18)
    cm.ax_heatmap.set_xticklabels(cm.ax_heatmap.get_xticklabels(), fontsize = 16)
    cm.ax_heatmap.set_yticklabels(cm.ax_heatmap.get_yticklabels(), fontsize = 16)

