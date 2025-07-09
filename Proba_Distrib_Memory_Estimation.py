#!/usr/bin/env python
# coding: utf-8

# In[16]:


from tqdm import tqdm
import re
import os
import csv
import matplotlib.pyplot as plt
import ndd
from itertools import product
import numpy as np   
import panphon
ft = panphon.FeatureTable()


# In[8]:


path_to_IPA = "" 

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


# In[9]:


len(Language_codes)


# In[18]:


path_to_distrib = "" 
path_to_rgrams = ""
absurd_palatals = ['ʲʲ', 'eʲ', 'aʲ', 'uʲ', 'ɐʲ', 'iʲ', 'ɨʲ', 'oʲ', 'jʲ', 'ʈʲ', 'ɪʲ', 'ɛʲ']
absurd_asp = ['uʰ', 'eʰ', 'aʰ', 'ʰʰ', 'oʰ', 'nʰ', 'mʰ', 'ɪʰ', 'iʰ', 'ɛʰ', 'ʊʰ', 'ɔʰ', 'ʂʰ', 'ʌʰ', '̃ʰ', 'ɳʰ', 'rʰ']
translation_table = {ord('ː') : None, ord('̝') : None, ord('"') : None, ord('͡') : None, ord('̪') : None, ord('.') : None, ord('̺') : None, ord('̻') : None, ord('̊') : None, ord('̯') : None, ord('̩') : None, ord('^') : None, ord('ː') : None}


# ## r-gram probability distributions

# In[19]:


def isolate_phonemes(word, lg): 
    # returns a dictionary where the keys are the phonemes in input word and the values are the number of occurrence of each phoneme

    word = word.translate(translation_table)
    # language-specific corrections
    if lg == 'am':
        word = word.translate({ord('`') : ord('ʼ')})
    if lg == 'de':
        if '??' in word:
            parts = word.split('??')
            w = ''
            for part in parts:
                w += part
                w += 'ʊr'
            word = w[:-2]
    if len(word) == 0 or any(substring in word for substring in absurd_palatals + absurd_asp) or word[0] in ['ʰ', 'ʲ', 'ʱ']:
        return {}
    
    else:
        # find phonemes which have a string length 2
        positions_nasal = [m.start() for m in re.finditer('̃', word)] # find nasal vowels
        positions_asp = [m.start() for m in re.finditer('ʰ', word)] # find aspirated consonants
        positions_murmur = [m.start() for m in re.finditer('ʱ', word)] # find murmured consonants
        positions_palatal = [m.start() for m in re.finditer('ʲ', word)] # find palatalised consonants
        positions_ejectives = [m.start() for m in re.finditer('ʼ', word)] # find ejective consonants
        positions_pharyngealized = [m.start() for m in re.finditer('ˤ', word)] # find pharyngealized consonants
        phonemes_w = {}
        Positions_two_char = set(positions_nasal).union(set(positions_asp)).union(set(positions_murmur)).union(set(positions_palatal)).union(set(positions_ejectives)).union(set(positions_pharyngealized))
        Positions_one_char = {i-1 for i in set(range(1,len(word)+1)) - Positions_two_char if i-1 not in Positions_two_char}
        for i in range(len(word)):
            if i in Positions_two_char:
                if word[i-1]+word[i] in phonemes_w:
                    phonemes_w[word[i-1]+word[i]] += 1
                else: 
                    phonemes_w[word[i-1]+word[i]] = 1
            else:
                if i in Positions_one_char:
                    if word[i] in phonemes_w:
                        phonemes_w[word[i]] += 1
                    else :
                        phonemes_w[word[i]] = 1
        return phonemes_w
        


# In[21]:


def separate_phonemes(word, lg): 
    # returns the list of phonemes in input word, with repetition
    word = word.translate(translation_table)
    if lg == 'am':
        word = word.translate({ord('`') : ord('ʼ')})
    if lg == 'de':
        if '??' in word:
            parts = word.split('??')
            w = ''
            for part in parts:
                w += part
                w += 'ʊr'
            word = w[:-2]
    if len(word) == 0 or any(substring in word for substring in absurd_palatals + absurd_asp) or word[0] in ['ʰ', 'ʲ', 'ʱ']:
        return []
    else :
        positions_nasal = [m.start() for m in re.finditer('̃', word)] # find nasal vowels
        positions_asp = [m.start() for m in re.finditer('ʰ', word)] # find aspirated consonants
        positions_murmur = [m.start() for m in re.finditer('ʱ', word)] # find murmured consonants
        positions_palatal = [m.start() for m in re.finditer('ʲ', word)] # find palatalised consonants
        positions_ejectives = [m.start() for m in re.finditer('ʼ', word)] # find ejective consonants
        positions_pharyngealized = [m.start() for m in re.finditer('ˤ', word)] # find pharyngealized consonants
        phonemes_w = []
        Positions_two_char = set(positions_nasal).union(set(positions_asp)).union(set(positions_murmur)).union(set(positions_palatal)).union(set(positions_ejectives)).union(set(positions_pharyngealized))
        Positions_one_char = {i-1 for i in set(range(1,len(word)+1)) - Positions_two_char if i-1 not in Positions_two_char}
        for i in range(len(word)):
            if i in Positions_two_char:
                phonemes_w.append(word[i-1]+word[i])     
            elif i in Positions_one_char: 
                phonemes_w.append(word[i])
        return phonemes_w


# In[55]:


def find_phonemes(lg):
    # returns a dictionary where the keys are the phonemes found in input language and values are numbers of occurrence of each phoneme
    print('Starting language', lg)
    IPA_text = open(path_to_IPA + '/' + "sep_words/"  + Lg_codes[lg] +'_sep_words.txt', "r", encoding = 'utf-8').readlines()
    phonemes = {}
    for line in IPA_text:
        if not '(' in line: # remove transcription errors
            line = line.strip().split(" ")
            for word in line:
                phonemes_w = isolate_phonemes(word, lg)
                for p in phonemes_w:
                    if p in absurd_palatals + absurd_asp:
                        print(f"Word {word} contains an absurd phoneme {p}")
                    if not p in phonemes:
                        phonemes[p] = phonemes_w[p]
                    else:
                        phonemes[p] += phonemes_w[p]
    print('Language', lg, 'done')
    return phonemes


# In[23]:


def phon_length(w): # returns phonetic length of IPA string (number of phonemes)
    positions_nasal = [m.start() for m in re.finditer('̃', w)] # find nasal vowels
    positions_asp = [m.start() for m in re.finditer('ʰ', w)] # find aspirated consonants
    positions_murmur = [m.start() for m in re.finditer('ʱ', w)] # find murmured consonants
    positions_palatal = [m.start() for m in re.finditer('ʲ', w)] # find palatalised consonants
    positions_ejectives = [m.start() for m in re.finditer('ʼ', w)] # find ejective consonants
    positions_pharyngealized = [m.start() for m in re.finditer('ˤ', w)] # find pharyngealized consonants
    # count number of "special" phonemes ie ones that have string length = 2
    nb_spec = len(positions_pharyngealized) + len(positions_nasal) + len(positions_asp) + len(positions_murmur) + len(positions_palatal) + len(positions_ejectives)    
    return len(w) - nb_spec


# In[24]:


def n_grams(n, alphabet): # returns the list of all possible n-grams given input alphabet
    return [''.join(comb) for comb in product(alphabet, repeat=n):]


# In[54]:


def r_grams_sep(r, lg): 
    # returns a dictionary where the keys are the r-grams found in lg and the values are the number of occurrence
    # word boundaries are kept
    print('Starting language', lg)
    IPA_text = open(path_to_IPA + '/sep_words/' + Lg_codes[lg] +'_sep_words.txt', "r", encoding = 'utf-8').readlines()
    r_grams = {}
    for line in IPA_text:
        if not '(' in line: # remove lines with wrong language identification: (en)...(lg) for example
            line = line.strip().split(" ")
            for w in line:
                w = w.translate(translation_table)
                if r <= len(w):
                    phonemes = separate_phonemes(w, lg)
                    # get list of r-blocks in word w
                    r_grams_w = [phonemes[i:i+r] for i in range(len(phonemes)-r+1)]
                    for rgram in r_grams_w:
                        rgram = "".join(rgram)
                        if rgram in r_grams:
                            r_grams[rgram] += 1
                        else:
                            r_grams[rgram] = 1
    print('Language', lg, 'done')
    return r_grams
    


# In[44]:


def r_grams_one_seq(r, lg): 
    # same as r_grams_sep but without word boundaries
    # source text is already filtered (because once the word boundaries are erased we can't remove wrongly transcribed words)
    print('Starting language', lg)
    IPA_text = open(path_to_IPA + '/one_sequence/' + Lg_codes[lg] +'_one_seq.txt', "r", encoding = 'utf-8').readlines()
    IPA_text_ = "".join([line.split('\n')[0] for line in IPA_text if not '(en)' in line])
    r_grams = {}
    phonemes = separate_phonemes(IPA_text_, lg)
    r_blocks = [phonemes[i:i+r] for i in tqdm(range(len(phonemes)-r+1))]
    for rgram in r_blocks:    
        rgram = "".join(rgram)
        if rgram in r_grams:
            r_grams[rgram] += 1
        else:
            r_grams[rgram] = 1
    print('Language', lg, 'done')
    return r_grams


# In[77]:


def Proba_distrib_sep(lg, r, path_to_distrib): 
    # writes a txt file with the probability distribution of r-grams in the input language, with word boundaries 
    with open(path_to_distrib + Language_codes[lg] + '_' + str(r) + '-grams_sep_words.txt' , "w", newline = '', encoding='utf-8') as f:
        fieldnames = ['r-gram', 'Counts']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        r_grams = r_grams_sep(r, lg)
        for gram in r_grams:
            Row = {}
            Row['r-gram'] = gram
            Row['Counts'] = r_grams[gram]
            writer.writerow(Row)               
        


# In[41]:


def Proba_distrib_seq(lg, r, path_to_distrib): 
    # writes a txt file with the probability distribution of r-grams in the input language, without word boundaries 
    with open(path_to_distrib + Language_codes[lg] + '_' + str(r) + '-grams_one_sequence.txt' , "w", newline = '', encoding='utf-8') as f:
        fieldnames = ['r-gram', 'Counts']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        r_grams = r_grams_one_seq(r, lg)
        for gram in r_grams:
            Row = {}
            Row['r-gram'] = gram
            Row['Counts'] = r_grams[gram]
            writer.writerow(Row)               
      
        


# In[ ]:


def Proba_distrib_seq_vect(lg, r, path_to_distrib): 
    # same as Proba_distrib_seq but with the feature vector representaion of 3-grams
    with open(path_to_distrib + Language_codes[lg] + '_' + str(i) + '-grams_one_sequence_vect.txt' , "w", newline = '', encoding='utf-8') as f:
        fieldnames = [str(i) for i in range(72)] + ['Counts']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        r_grams = r_grams_one_seq(r, lg)
        for gram in r_grams:
            vect = list(np.concatenate(ft.word_to_vector_list(gram, numeric = True)))
            Row = {str(i) : vect[i] for i in range(72)}
            Row['Counts'] = r_grams[gram]
            writer.writerow(Row)          


# ### Coarse-grained phoneme category probability distributions 

# In[ ]:


# phoneme categories
front_nfront = ['i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ɛ', 'œ', 'æ', 'ɶ', 'a']
central = ['ɘ', 'ɵ', 'ɞ','ɐ','ɨ','ʉ','ə','ɜ']
back_nback = ['ɯ', 'u', 'ʊ', 'ɤ', 'o', 'ʌ', 'ɔ', 'ɑ', 'ɒ']
open_nopen = ['ɑ', 'ɒ','æ', 'ɶ', 'a','ɐ']
mid_cmid_omid = ['e', 'ø','ɘ', 'ɵ','ɤ', 'o','ə','ɜ','ɛ', 'œ','ɞ','ʌ', 'ɔ' ]
close_nclose = ['i', 'y', 'ɪ', 'ʏ','ɨ','ʉ','ɯ', 'u', 'ʊ']


# In[ ]:


def cf1(p): # first class function : separate consonants vs vowels
    v = ft.word_to_vector_list(u'%s' %p, numeric=True)[0]
    if v[2] == 1:
        return 'c'
    elif v[2] == -1 :
        return 'v'

def cf2(p): # voiced consonants, unvoiced consonants, vowels
    v = ft.word_to_vector_list(u'%s' %p, numeric=True)[0]
    if v[2] == 1:
        if v[8] == 1:
            return 'c'
        else:
            return 'u'
    elif v[2] == -1 :
        return 'v'

def cf3(p): # consonant vs open-mid-close
    v = ft.word_to_vector_list(u'%s' %p, numeric=True)[0]
    if v[2] == 1:
        return 'c'
    else :    # v[2] == -1 :
        if p in open_nopen:
            return 'o'
        elif p in mid_cmid_omid:
            return 'm'
        return 'f' # f for "ferme"

def cf4(p): # voiced/unvoiced vs high-mid-low
    v = ft.word_to_vector_list(u'%s' %p, numeric=True)[0]
    if v[2] == 1:
        if v[8] == 1:
            return 'c'
        return 'u'
    else :    # v[2] == -1 :
        if p in open_nopen:
            return 'o'
        elif p in mid_cmid_omid:
            return 'm'
        return 'f' # f for "ferme", or "close"

def cf5(p): # consonant vs open-mid-low
    v = ft.word_to_vector_list(u'%s' %p, numeric=True)[0]
    if v[2] == 1:
        return 'c'
    else :    # v[2] == -1 :
        if p in front_nfront:
            return 'f'
        elif p in central:
            return 'k' # kentr
        return 'b' # back

def cf6(p): # voiced/unvoiced vs open-mid-low
    v = ft.word_to_vector_list(u'%s' %p, numeric=True)[0]
    if v[2] == 1:
        if v[8] == 1:
            return 'c'
        return 'u'
    else :    # v[2] == -1 :
        if p in front_nfront:
            return 'f'
        elif p in central:
            return 'k' # kentr
        return 'b' # back


# In[ ]:


def separate_phonemes_feat(word, lg, classfunction): 
    # same as separate_phonemes but each phoneme is mapped to its category via the classfunction
    word = word.translate(translation_table)
    if lg == 'de':
        if '??' in word:
            parts = word.split('??')
            w = ''
            for part in parts:
                w += part
                w += 'ʊr'
            word = w[:-2]
    if lg == 'am':
        word = word.translate({ord('`') : ord('ʼ')})
    if len(word) == 0 or any(substring in word for substring in absurd_palatals + absurd_asp) or word[0] in ['ʰ', 'ʲ', 'ʱ']:
        return []
    else :
        positions_nasal = [m.start() for m in re.finditer('̃', word)] # find nasal vowels
        positions_asp = [m.start() for m in re.finditer('ʰ', word)] # find aspirated consonants
        positions_murmur = [m.start() for m in re.finditer('ʱ', word)] # find murmured consonants
        positions_palatal = [m.start() for m in re.finditer('ʲ', word)] # find palatalised consonants
        positions_ejectives = [m.start() for m in re.finditer('ʼ', word)] # find ejective consonants
        positions_pharyngealized = [m.start() for m in re.finditer('ˤ', word)] # find pharyngealized consonants
        phonemes_w = []
        Positions_two_char = set(positions_nasal).union(set(positions_asp)).union(set(positions_murmur)).union(set(positions_palatal)).union(set(positions_ejectives)).union(set(positions_pharyngealized))
        Positions_one_char = {i-1 for i in set(range(1,len(word)+1)) - Positions_two_char if i-1 not in Positions_two_char}
        for i in range(len(word)):
            if i in Positions_two_char:
                phonemes_w.append(classfunction(word[i-1]+word[i]))     
            elif i in Positions_one_char: 
                phonemes_w.append(classfunction(word[i]))
        return phonemes_w


# In[ ]:


def r_grams_one_seq_feat(lg, r, cf): 
    # source text is already filtered (because once the word boundaries are erased we can't remove problematic words)
    print('Starting language', lg)
    IPA_text = open(path_to_IPA + '/one_sequence/' + Lg_codes[lg] +'_one_seq.txt', "r", encoding = 'utf-8').readlines()
    IPA_text_ = "".join([line.split('\n')[0] for line in IPA_text if not '(en)' in line])
    r_grams = {}
    phonemes = separate_phonemes_feat(IPA_text_, lg, cf)
    r_blocks = [phonemes[i:i+r] for i in tqdm(range(len(phonemes)-r+1))]
    for rgram in r_blocks:    
        rgram = "".join(rgram)
        if rgram in r_grams:
            r_grams[rgram] += 1
        else:
            r_grams[rgram] = 1
    print('Language', lg, 'done')
    return r_grams


# In[ ]:


alphabets = {cf1 : ['c', 'v'], cf2 : ['u', 'c', 'v'], cf3 : ['c', 'o', 'm', 'f'], cf4 : ['c', 'u', 'o', 'm', 'f'], cf5 : ['c', 'f', 'k', 'b'], cf6 : ['c', 'u', 'f', 'k', 'b'] }
fileid = {cf1 : '_cv', cf2 : '_ucv', cf3 : '_comf', cf4 : '_ucomf', cf5 : '_cfkb', cf6 : '_ucfkb'}

def Proba_distrib_seq_vect_feat(lg, r, cf): 
    # word boundaries ignored here
    with open(path_to_distrib + Lg_codes[lg] + '_' + str(r) + '-grams_one_sequence_feat' + fileid[cf] + '.txt' , "w", newline = '', encoding='utf-8') as f:
        fieldnames = ['r-gram', 'Counts']
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        r_grams = r_grams_one_seq_feat(lg, r, cf)
        for gram in r_grams:
            Row = {}
            Row['r-gram'] = gram
            Row['Counts'] = r_grams[gram]
            writer.writerow(Row)


# ### Average of IE family

# In[ ]:


# compute the average 3gram probability distribution of the IE family
Average_Proba_Distrib = {}
for lg in IE_languages:
    if lg 1= 'af': # remove Afrikaans from geographical analysis
        with open(path_to_distrib + Language_codes[lg] + '_3-grams_one_sequence_vect.txt') as f:
            r = csv.reader(f)
            Counts = {tuple([tuple([int(row[i]) for i in range(0,24)]), tuple([int(row[j]) for j in range(24,48)]), tuple([int(row[k]) for k in range(48,72)])]) : int(row[72]) for row in [line for line in r][1:]}
            L1_values = [list(list(u) for u in t) for t in Counts.keys()]
            for gram in Counts:
                if not gram in Average_Proba_Distrib:
                    Average_Proba_Distrib[gram] = Counts[gram]
                else:
                    Average_Proba_Distrib[gram] += Counts[gram]


# In[ ]:


n = len(New_Language_codes)
for c in Average_Proba_Distrib:
    Average_Proba_Distrib[c] /= n


# In[ ]:


path_to_save_avg = ""


# In[ ]:


with open(path_to_save_avg, "w", newline = '', encoding='utf-8') as f:
    fieldnames = [str(i) for i in range(72)] + ['Counts']
    writer = csv.DictWriter(f, fieldnames = fieldnames)
    writer.writeheader()
    for gram in Average_Proba_Distrib:
        vect = list(np.concatenate(np.array(gram)))
        Row = { str(i) : vect[i] for i in range(72)}
        Row['Counts'] = Average_Proba_Distrib[gram]
        writer.writerow(Row)     


# ## Memory Estimation

# In[28]:


def PredGain_improved(lg, sep_words, with_zeros):  # returns G0, G1, G2, G3 calculated with text = [words] or [bigseq]
    # with_zeros = boolean variable to decide if zero-probability r-grams should be taken into account in the entropy calculation. 
    # it doesn't seem to have a huge influence on the result
    separation = {True : '-grams_sep_words.txt', False : '-grams_one_sequence.txt'}
    phonemes = find_phonemes(lg)
    alphabet = [p for p in phonemes.keys()]
    possible_1_grams = n_grams(1, alphabet)
    possible_2_grams = n_grams(2, alphabet)
    possible_3_grams = n_grams(3, alphabet)
    possible_4_grams = n_grams(4, alphabet)
    possible_5_grams = n_grams(5, alphabet)


    # Open each r-gram probability distribution and create dictionary {rgram : nb of occurence}
    
    with open(path_to_distrib + Language_codes[lg] + '_1' + separation[sep_words], encoding = 'utf-8') as f:
        counts_lg_1_withoutzeros = {}
        counts_lg_1_withzeros = {g : 0 for g in possible_1_grams}
        reader = csv.reader(f)
        for row in [line for line in reader][1:]:
            counts_lg_1_withoutzeros[row[0]] = int(row[1])
            counts_lg_1_withzeros[row[0]] = int(row[1])

    with open(path_to_distrib + Language_codes[lg] + '_2' + separation[sep_words], encoding = 'utf-8') as f:
        counts_lg_2_withoutzeros = {}
        counts_lg_2_withzeros = {g : 0 for g in possible_2_grams}
        reader = csv.reader(f)
        for row in [line for line in reader][1:]:
            counts_lg_2_withoutzeros[row[0]] = int(row[1])
            counts_lg_2_withzeros[row[0]] = int(row[1])
    
    with open(path_to_distrib + Language_codes[lg] + '_3' + separation[sep_words], encoding = 'utf-8') as f:
        counts_lg_3_withoutzeros = {}
        counts_lg_3_withzeros = {g : 0 for g in possible_3_grams}
        reader = csv.reader(f)
        for row in [line for line in reader][1:]:
            counts_lg_3_withoutzeros[row[0]] = int(row[1])
            counts_lg_3_withzeros[row[0]] = int(row[1])
    
    with open(path_to_distrib + Language_codes[lg] + '_4' + separation[sep_words], encoding = 'utf-8') as f:
        counts_lg_4_withoutzeros = {}
        counts_lg_4_withzeros = {g : 0 for g in possible_4_grams}
        reader = csv.reader(f)
        for row in [line for line in reader][1:]:
            counts_lg_4_withoutzeros[row[0]] = int(row[1])
            counts_lg_4_withzeros[row[0]] = int(row[1])
   
    with open(path_to_distrib + Language_codes[lg] + '_5' + separation[sep_words], encoding = 'utf-8') as f:
        counts_lg_5_withoutzeros = {}
        counts_lg_5_withzeros = {g : 0 for g in possible_5_grams}
        reader = csv.reader(f)
        for row in [line for line in reader][1:]:
            counts_lg_5_withoutzeros[row[0]] = int(row[1])
            counts_lg_5_withzeros[row[0]] = int(row[1])
    
    # Compute block entropies
    H_0 = (0,0)
    if with_zeros:
        H_1 = ndd.entropy(counts_lg_1_withzeros, k = len(possible_1_grams), return_std = True)
        H_2 = ndd.entropy(counts_lg_2_withzeros, k = len(possible_2_grams), return_std = True)
        H_3 = ndd.entropy(counts_lg_3_withzeros, k = len(possible_3_grams), return_std = True)
        H_4 = ndd.entropy(counts_lg_4_withzeros, k = len(possible_4_grams), return_std = True)
        H_5 = ndd.entropy(counts_lg_5_withzeros, k = len(possible_5_grams), return_std = True)
    else:
        H_1 = ndd.entropy(counts_lg_1_withoutzeros, k = len(counts_lg_1_withoutzeros), return_std = True)
        H_2 = ndd.entropy(counts_lg_2_withoutzeros, k = len(counts_lg_2_withoutzeros), return_std = True)
        H_3 = ndd.entropy(counts_lg_3_withoutzeros, k = len(counts_lg_3_withoutzeros), return_std = True)
        H_4 = ndd.entropy(counts_lg_4_withoutzeros, k = len(counts_lg_4_withoutzeros), return_std = True)
        H_5 = ndd.entropy(counts_lg_5_withoutzeros, k = len(counts_lg_5_withoutzeros), return_std = True)

    # Compute predictability gains (negative second discrete derivative of block entropy)
    G_0, uG_0 = -(H_2[0] - 2*H_1[0] + H_0[0]), H_2[1] + 2*H_1[1] + H_0[1]
    G_1, uG_1 = -(H_3[0] - 2*H_2[0] + H_1[0]), H_3[1] + 2*H_2[1] + H_1[1]
    G_2, uG_2 = -(H_4[0] - 2*H_3[0] + H_2[0]), H_4[1] + 2*H_3[1] + H_2[1]
    G_3, uG_3 = -(H_5[0] - 2*H_4[0] + H_3[0]), H_5[1] + 2*H_4[1] + H_3[1]
    return G_0, G_1, G_2, G_3


# In[ ]:


def Plot_Segmentations(lg): # compare the G values associated with the different classifications for language lg
    alphabets = {cf1 : ['c', 'v'], cf2 : ['u', 'c', 'v'], cf3 : ['c', 'o', 'm', 'f'], cf4 : ['c', 'u', 'o', 'm', 'f'], cf5 : ['c', 'f', 'k', 'b'], cf6 : ['c', 'u', 'f', 'k', 'b'] }
    fileid = {cf1 : '_cv', cf2 : '_ucv', cf3 : '_comf', cf4 : '_ucomf', cf5 : '_cfkb', cf6 : '_ucfkb'}
    lab = {cf1 : 'Cons/Vow', cf2 : 'Voiced/Unvoiced/Vowel', cf3 : 'Cons/Open/Mid/Close', cf4 : 'Voiced/Unvoiced/Open/Mid/Close', cf5 : 'Cons/Front/Central/Back', cf6 : 'Voiced/Unvoiced/Front/Central/Back'}
    def ProbaDistrib3(lg,cf): # joined words and sentences (only one sequence)
        for i in range(1,6):
            with open(path_to_distrib + Language_codes[lg] + '_' + str(i) + '-grams_one_sequence_feat' + fileid[cf] + '.txt' , "w", newline = '', encoding='utf-8') as f:
                fieldnames = ['r-gram', 'Counts']
                writer = csv.DictWriter(f, fieldnames = fieldnames)
                writer.writeheader()
                r_grams = r_grams_one_seq(i, lg, cf)
                for gram in r_grams:
                    Row = {}
                    Row['r-gram'] = gram
                    Row['Counts'] = r_grams[gram]
                    writer.writerow(Row)  
                    
    def PredGain_seg(lg, sep_words, with_zeros, cf):  # returns G0, G1, G2, calculated with text = [words] or [bigseq]
            separation = {False : '-grams_one_sequence_feat' + fileid[cf] + '.txt'}
            alphabet = alphabets[cf]
            possible_1_grams = n_grams(1, alphabet)
            possible_2_grams = n_grams(2, alphabet)
            possible_3_grams = n_grams(3, alphabet)
            possible_4_grams = n_grams(4, alphabet)
            possible_5_grams = n_grams(5, alphabet)
        
            with open(path_to_distrib + Language_codes[lg] + '_1' + separation[sep_words], encoding = 'utf-8') as f:
                counts_lg_1_withoutzeros = {}
                counts_lg_1_withzeros = {g : 0 for g in possible_1_grams}
                reader = csv.reader(f)
                for row in [line for line in reader][1:]:
                    counts_lg_1_withoutzeros[row[0]] = int(row[1])
                    counts_lg_1_withzeros[row[0]] = int(row[1])

            with open(path_to_distrib + Language_codes[lg] + '_2' + separation[sep_words], encoding = 'utf-8') as f:
                counts_lg_2_withoutzeros = {}
                counts_lg_2_withzeros = {g : 0 for g in possible_2_grams}
                reader = csv.reader(f)
                for row in [line for line in reader][1:]:
                    counts_lg_2_withoutzeros[row[0]] = int(row[1])
                    counts_lg_2_withzeros[row[0]] = int(row[1])
            
            with open(path_to_distrib + Language_codes[lg] + '_3' + separation[sep_words], encoding = 'utf-8') as f:
                counts_lg_3_withoutzeros = {}
                counts_lg_3_withzeros = {g : 0 for g in possible_3_grams}
                reader = csv.reader(f)
                for row in [line for line in reader][1:]:
                    counts_lg_3_withoutzeros[row[0]] = int(row[1])
                    counts_lg_3_withzeros[row[0]] = int(row[1])
            
            with open(path_to_distrib + Language_codes[lg] + '_4' + separation[sep_words], encoding = 'utf-8') as f:
                counts_lg_4_withoutzeros = {}
                counts_lg_4_withzeros = {g : 0 for g in possible_4_grams}
                reader = csv.reader(f)
                for row in [line for line in reader][1:]:
                    counts_lg_4_withoutzeros[row[0]] = int(row[1])
                    counts_lg_4_withzeros[row[0]] = int(row[1])
           
            with open(path_to_distrib + Language_codes[lg] + '_5' + separation[sep_words], encoding = 'utf-8') as f:
                counts_lg_5_withoutzeros = {}
                counts_lg_5_withzeros = {g : 0 for g in possible_5_grams}
                reader = csv.reader(f)
                for row in [line for line in reader][1:]:
                    counts_lg_5_withoutzeros[row[0]] = int(row[1])
                    counts_lg_5_withzeros[row[0]] = int(row[1])
            
              
            H_0 = (0,0)
            if with_zeros:
                H_1 = ndd.entropy(counts_lg_1_withzeros, k = len(possible_1_grams), return_std = True)
                H_2 = ndd.entropy(counts_lg_2_withzeros, k = len(possible_2_grams), return_std = True)
                H_3 = ndd.entropy(counts_lg_3_withzeros, k = len(possible_3_grams), return_std = True)
                H_4 = ndd.entropy(counts_lg_4_withzeros, k = len(possible_4_grams), return_std = True)
                H_5 = ndd.entropy(counts_lg_5_withzeros, k = len(possible_5_grams), return_std = True)
            else:
                H_1 = ndd.entropy(counts_lg_1_withoutzeros, k = len(counts_lg_1_withoutzeros), return_std = True)
                H_2 = ndd.entropy(counts_lg_2_withoutzeros, k = len(counts_lg_2_withoutzeros), return_std = True)
                H_3 = ndd.entropy(counts_lg_3_withoutzeros, k = len(counts_lg_3_withoutzeros), return_std = True)
                H_4 = ndd.entropy(counts_lg_4_withoutzeros, k = len(counts_lg_4_withoutzeros), return_std = True)
                H_5 = ndd.entropy(counts_lg_5_withoutzeros, k = len(counts_lg_5_withoutzeros), return_std = True)
            
            G_0, uG_0 = -(H_2[0] - 2*H_1[0] + H_0[0]), H_2[1] + 2*H_1[1] + H_0[1]
            G_1, uG_1 = -(H_3[0] - 2*H_2[0] + H_1[0]), H_3[1] + 2*H_2[1] + H_1[1]
            G_2, uG_2 = -(H_4[0] - 2*H_3[0] + H_2[0]), H_4[1] + 2*H_3[1] + H_2[1]
            G_3, uG_3 = -(H_5[0] - 2*H_4[0] + H_3[0]), H_5[1] + 2*H_4[1] + H_3[1]
            return G_0, G_1, G_2, G_3
     
    
    plt.style.use('tableau-colorblind10')
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8,6))
    plt.xlabel('$u$', fontsize = 18)
    plt.ylabel('$\mathcal{G}_u$', fontsize = 18)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18, direction = 'in', length = 6)
    for cf in [cf1,cf2,cf3,cf4,cf5,cf6]:   
        ProbaDistrib3(lg, cf)
        G0, G1, G2, G3 = PredGain_seg(lg, False, False, cf)
        x = [0,1,2,3]
        y = [G0, G1, G2, G3]
        plt.plot(x,y, '--+', label = lab[cf])
    
    plt.legend()
    plt.title(Language_codes[lg] + " Predictability gain with different segmentations")
    plt.show()

