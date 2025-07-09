import epitran
from phonemizer.punctuation import Punctuation
from epitran.backoff import Backoff
import tqdm


path_to_text = ""
path_to_IPA = ""


Language_codes = {'eng-Latn' : 'English', 
                  'fra-Latn' : 'French', 
                  'deu-Latn' : 'German',
                 'afr-Latn' : 'Afrikaans',
                 'amh-Ethi' : 'Amharic',
                 'ara-Arab' : 'Arabic',
                 'ceb-Latn' : 'Cebuano',
                 'ces-Latn' : 'Czech',
                 'cmn-Hant' : 'Chinese',
                 'fas-Arab' : 'Farsi',
                 'fin-Latn' : 'Finnish',
                 'fra-Latn' : 'French',
                 'hin-Deva' : 'Hindi',
                 'hrv-Latn' : 'Croatian',
                 'hun-Latn' : 'Hungarian',
                 'ind-Latn': 'Indonesian',
                 'ita-Latn' : 'Italian',
                 'kor-Hang' : 'Korean',
                 'lav-Latn' : 'Latvian-NT',
                 'lit-Latn' : 'Lithuanian',
                 'mal-Mlym' : 'Malayalam',
                 'mar-Deva' : 'Marathi',
                 'mri-Latn' : 'Maori',
                 'mya-Mymr' : 'Myanmar',
                 'nld-Latn' : 'Dutch',
                 'pol-Latn' : 'Polish',
                 'por-Latn' : 'Portuguese',
                 'ron-Latn' : 'Romanian',
                 'rus-Cyrl' : 'Russian',
                 'sin-Sinh' : 'Sinhala',
                 'slv-Latn' : 'Slovene',
                 'sna-Latn' : 'Shona',
                 'som-Latn' : 'Somali',
                 'spa-Latn' : 'Spanish',
                 'sqi-Latn' : 'Albanian',
                 'srp-Latn' : 'Serbian',
                 'swa-Latn' : 'Swahili-NT',
                 'swe-Latn' : 'Swedish',
                 'tel-Telu' : 'Telugu',
                 'tgl-Latn' : 'Tagalog',
                 'tha-Thai' : 'Thai',
                 'tur-Latn' : 'Turkish',
                 'ukr-Cyrl' : 'Ukrainian-NT',
                 'vie-Latn' : 'Vietnamese',
                 'xho-Latn' : 'Xhosa',
                 'zul-Latn' : 'Zulu-NT'}




def Txt_to_IPA_epi(lg, path_to_text, path_to_IPA):
    file = path_to_text + '/' + Language_codes[lg] + '.txt'
    backoff = Backoff([lg])
    text = open(file, "r", encoding='utf-8').readlines()
    text = Punctuation(';:,.!"?()-`´').remove(text)
    new_text = []
    print('Starting phonemization of language', lg)
    path_to_IPA_file = path_to_IPA + '/' + 'sep_words' + '/' + Language_codes[lg] + '_sep_words.txt'
    for line in text:
        line = line.split()
        new_text.append(" ".join([backoff.transliterate(line[i]) for i in range(len(line))]))
    print('Language', lg, 'phonemized')
    print('Starting writing of file', lg)
    with open(path_to_IPA_file, 'w', encoding='utf-8') as f:
        for line in new_text:
            #line = line.translate(str.maketrans('','','1234567890')) # this is for erasing tones (marked with numerals)
            f.write(line)
            f.write("\n")      
    print('Language', lg, 'done')



def Txt_to_IPA_epi_one_seq(lg, path_to_IPA):
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

