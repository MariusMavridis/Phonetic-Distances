# Assessing Phonological Distances for Linguistic Typology and the Indo-European Origin

This repository contains the code and data used for a research project carried out at IFISC, in which we performed a stochastic modelling of phonetic distances between languages, and explored correlations between phonetic similarity and geographic proximity. 

### Abstract

We show that short-range phoneme dependencies encode large-scale patterns of linguistic relatedness, with direct implications for quantitative typology and evolutionary linguistics. Specifically, using an information-theoretic framework, we argue that phoneme sequences modeled as second-order Markov chains essentially capture the statistical correlations of a phonological system. This finding enables us to quantify distances among 67 modern languages from a multilingual parallel corpus. The resulting phonological distance matrix recovers major language families and reveals signatures of contact-induced convergence. Remarkably, we obtain a clear correlation with geographic distance, allowing us to constrain a plausible homeland region for the Indo-European family, consistent with the Steppe hypothesis.

### Project organization

```
├── README.md
├── Raw texts <- Pre-processed 67 Bible texts + Moby Dick (English)         
├── IPA texts <- Bible texts transcribed in the IPA, without word boundaries
├── ProbaDistrib       <- r-phone probability distributions for r = 1,...,5, with and without word boundaries
│   ├── ProbaDistrib_IPA   <- Probability distributions of IPA r-phones 
│   ├── ProbaDistrib_vect     <- Probability distributions of feature vectors (includes average IE probability distribution)
│  
├── Alldist.txt          <- Wasserstein distances between all pairs of languages
│
├── wals_languages.csv       <- contains info about languages in the WALS database, and was downloaded at https://zenodo.org/records/13950591
│
├── Epitran_Transcription.py           <- Scripts to transcribe Bible texts from Raw_texts to IPA texts using Epitran
│
├── Phonemizer_Transcription.py           <- Scripts to transcribe Bible texts from Raw_texts to IPA texts using Phonemizer
│
├── Geographic_correlations.py           <- Scripts to compute correlation between geographic and phonetic distances, and find plausible IE homeland
│
├── Proba_Distrib_Memory_Estimation.py   <- Scripts to compute r-phone probability distributions from IPA texts, as well as predictability gains to estimate the memory
│
├── Wasserstein_Distance.py           <- Scripts to compute the Wasserstein distance between r-phone probability distributions
```

`Alldist.txt` contains all phonetic distances between pairs of languages in our corpus, obtained by calculating the Wasserstein distance between 3-phone probability distributions.

Raw texts contains all 67 Bible text files + Moby Dick in English

IPA texts contains all IPA transcriptions of the files in Raw Texts, using Phonemizer or Epitran, and obtained with the scripts Phonemizer_Transcription.py or Epitran_Transcription.py, respectively.

Proba_Distrib contains 3-phone probability distributions of all languages, with or without word boundaries, and in a separate folder, 3-gram probability distributions using coarse-grained phoneme categories. It also contains the average 3-phone probability distribution of the Indo-European language family

wals_languages.csv contains info about languages in the WALS database, and was downloaded at https://zenodo.org/records/13950591
