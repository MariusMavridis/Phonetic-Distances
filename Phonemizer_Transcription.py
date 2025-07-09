#!/usr/bin/env python
# coding: utf-8

# In[6]:


from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from tqdm import tqdm # progress bar
import os


# In[7]:


# define characters to remove/filter
absurd_palatals = ['ʲʲ', 'eʲ', 'aʲ', 'uʲ', 'ɐʲ', 'iʲ', 'ɨʲ', 'oʲ', 'jʲ', 'ʈʲ', 'ɪʲ', 'ɛʲ']
absurd_asp = ['uʰ', 'eʰ', 'aʰ', 'ʰʰ', 'oʰ', 'nʰ', 'mʰ', 'ɪʰ', 'iʰ', 'ɛʰ', 'ʊʰ', 'ɔʰ', 'ʂʰ', 'ʌʰ', '̃ʰ', 'ɳʰ', 'rʰ', 'ʃʰ']
translation_table_seq = {ord('̝') : None, ord('"') : None, ord('͡') : None, ord('̪') : None, ord('.') : None, ord('̺') : None, ord('̻') : None, ord('̊') : None, ord('̯') : None, ord('̩') : None, ord('^') : None, ord('ː') : None}

# define input and output paths
path_to_text = ""
path_to_IPA = ""

Language_codes = {
'af' : 'Afrikaans',
'sq' : 'Albanian',
'am' : 'Amharic',
'ar' : 'Arabic (Modern Standard)',
'eu' : 'Basque',
'bg' : 'Bulgarian',
'ceb' : 'Cebuano',
'cs' : 'Czech',
'da' : 'Danish',
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
'is' : 'Icelandic',
'id' : 'Indonesian',
'it' : 'Italian',
'kn' : 'Kannada',
'lv' : 'Latvian',
'lt' : 'Lithuanian',
'ml' : 'Malayalam',
'mi' : 'Maori',
'mr' : 'Marathi',
'my' : 'Burmese',
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
'sw' : 'Swahili',
'te' : 'Telugu',
'th' : 'Thai',
'tr' : 'Turkish',
'uk' : 'Ukrainian',
'vi' : 'Vietnamese',
'jv' : 'Javanese',
'yo' : 'Yoruba',
'ur' : 'Urdu',
'tpi' : 'Tok Pisin',
'tir' : 'Tigrinya',
'az' : 'Azerbaijani',
'bsh' : 'Bashkir',
'ben' : 'Bengali',
'kab' : 'Kabardian',
'kyr' : 'Kirghiz',
'aar-Latn' : 'Qafar',
'aii-Syrc' : 'Neo-Aramaic (Assyrian)',
'ava-Cyrl' : 'Avar',
'kbd-Cyrl' : 'Kabardian',
'kmr-Latn' : 'Kurmanji',
'tgk-Cyrl' : 'Tajik',
'tir-Ethi' : 'Tigrinya',
'xho-Latn' : 'Xhosa',
'yor-Latn' : 'Yoruba',
'zul-Latn' : 'Zulu',
'bxk-Latn' : 'Bukusu',
'hau-Latn' : 'Hausa',
'ilo-Latn' : 'Ilocano',
'khm-Khmr' : 'Khmer',
'kin-Latn' : 'Kinyarwanda',
'lao-Laoo' : 'Lao',
'nya-Latn' : 'Chichewa',
'ood-Latn-sax' : "O'odham",
'quy-Latn' : 'Quechua (Ayacucho)',
'run-Latn' : 'Rundi',
'sag-Latn' : 'Sango',
'hy' : 'Armenian (Eastern)',
'hyw' : 'Armenian (Western)',
'as' : 'Assamese',
'ba_OT' : 'Bashkir',
'ba_NT' : 'Bashkir',
'cu' : 'Chuvash',
'be' : 'Belorussian',
'bn' : 'Bengali',
'bs' : 'Bosnian',
'ca' : 'Catalan',
'ga' : 'Irish',
'gd' : 'Gaelic (Scots)',
'ka' : 'Georgian',
'kok' : 'Konkani',
'ku' : 'Kurdish (Southern)', # not in WALS
'kk' : 'Kazakh',
'ky' : 'Kirghiz',
'ltg' : 'Latgalian', # not in WALS
'mk' : 'Macedonian',
'mt' : 'Maltese',
'nog' : 'Noghay',
'or' : 'Oriya',
'om' : 'Oromo (Boraana)',
'pa' : 'Panjabi',
'sd' : 'Sindhi',
'si' : 'Sinhala',
'ta' : 'Tamil',
'tk' : 'Turkmen',
'tt' : 'Tatar',
'ug' : 'Uyghur',
'cy' : 'Welsh',
'ur' : 'Urdu',
'uz' : 'Uzbek',
'haw' : 'Hawaiian',
'ms' : 'Malay',
'pap' : 'Papiamentu',
'tn' : 'Tswana',
'shn' : 'Shona'}

# temporarily removed japanese, icelandic and K'iche' because of unrecognized characters
# I needed to modify the french txt because the apostrophe ' appears as ` which phonemizer doesnt recognize
# I needed to modify the portuguese txt because the grave accent à appears as � which phonemizer doesnt recognize
# I needed to modify the nepali txt because there were (verse?) numbers polluting the text


# In[8]:


# this function uses phonemizer to transcribe a text in the IPA.
# path_to_text and path_to_IPA are input and output path files and need to be defined by the user

def Txt_to_IPA_phz(lg, path_to_text, path_to_IPA):
    file = path_to_text + '/' + Language_codes[lg] + '.txt'
    
    # read input file and remove punctuation
    text = open(file, "r").readlines()
    text = Punctuation(';:,.!"?()-`´').remove(text)
    
    path_to_IPA_file = path_to_IPA + '/' + 'sep_words' + '/' + Language_codes[lg] + '_sep_words.txt'
    separator=Separator(phone=None, word=' ', syllable=None)

    # Phonemize the text
    print('Starting phonemization of language', lg)
    phn = phonemize(tqdm(text), language = lg, backend = 'espeak', separator = separator, njobs = 1)
    print('Language', lg, 'phonemized')
    
    with open(path_to_IPA_file, 'w') as f:
        for line in tqdm(phn):
            #line = line.translate(str.maketrans('','','1234567890')) # this is for erasing tones (marked with numbers)
            f.write(line)
            f.write("\n")      
    print('Language', lg, 'done')


# In[9]:


def Txt_to_IPA_phz_one_seq(lg, path_to_IPA):
    if os.path.isfile(path_to_IPA + '/sep_words/' + Language_codes[lg] +'_sep_words.txt'):
        print('Starting language', Language_codes[lg])
        IPA_text = open(path_to_IPA + '/sep_words/' + Language_codes[lg] +'_sep_words.txt', "r", encoding = 'utf-8').readlines()
        IPA_text_ = ''
        for line in IPA_text:
            if not '(' in line: # removes lines with wrongly transcribed words: (en)...(lg)
                line_ = line.split()
                keep_line = True
                new_line = []
                for word in line_:
                    word = word.translate(translation_table_seq)
                    if lg == 'am': # right symbol for ejectives in Amharic
                        word = word.translate({ord('`') : ord('ʼ')})
                    # remove line if absurd transcriptions
                    if len(word) == 0 or any(substring in word for substring in absurd_palatals + absurd_asp) or word[0] in ['ʰ', 'ʲ', 'ʱ']:
                        keep_line = False
                    new_line.append(word)
                if keep_line:
                    IPA_text_ += "".join(new_line)

        path_to_IPA_file = path_to_IPA + '/' + 'one_sequence' + '/' + Language_codes[lg] + '_one_seq.txt'
        with open(path_to_IPA_file, 'w', encoding='utf-8') as f:
            f.write(IPA_text_)
        print('Language', Language_codes[lg], 'done')

