This repository contains the code and data used during my 3-month research stay at IFISC in Palma de Mallorca, in which I carried out a stochastic modelling of phonetic distances between languages, and explored correlations between phonetic similarity and geographic proximity. 

Abstract:

Human speech can be seen as a complex system where meaning arises from the interaction between many sound or grammatical units. More specifically, since language is sequential, it can be accurately modeled as a high-order Markov chain.  While this approach has been successful in studying cross-linguistic syntactic distances, phonetics studies usually focus on dialect comparison, with little data for each language. In this work, we model the sequence of sounds (or phonemes) in a text as a high-order Markov chain, and show using an information-theoretic framework that order-2 transition probabilities are enough to statistically characterize a given language. Using 67 translations of the Bible transcribed into the International Phonetic Alphabet, we compute the  pairwise distances between the probability distributions of blocks of 3 phonemes. The resulting phonetic distance matrix is used to recover language clusters, which reflect previously known language families and highlight the influence of language contact. We show that phonetic and geographic distances are correlated, and constrain an origin region for the Indo-European language family, consistent with the Steppe hypothesis.

Description of the files:

Alldist.txt contains all phonetic distances between pairs of languages in our corpus, obtained by calculating the Wasserstein distance between 3-phone probability distributions.

Raw_Texts contains all 67 Bible text files 

IPA_Bibles contains all IPA transcriptions of the files in Raw Texts, using Phonemizer or Epitran, and obtained with the scripts Phonemizer_Transcription.py or Epitran_transcription.py, respectively

Probability_Distributions contains 3-phone probability distributions of all languages, with or without word boundaries, and in a separate folder, 3-gram probability distributions using coarse-grained phoneme categories. It also contains the average 3-phone probability distribution of the Indo-European language family
