# Introduction to NLP


### Week 1 
Below are the main topics which were focused on week 1
- From plain texts to classification
- Simple deep learning for text classification

__Text Preprocessing__
__What is "Text"__
First step of text preprocessing is to define what is "Text". So, "Text" can be defined as a sequence. Where it can be sequence of anything for example, sequence of characters or words, or  phrases, or even sentences and paragraphs. 

__What is "Word"__
Let's for now, define "Text" as sequence of words. Then again, "Word" can be described as __meaningful__ sequence of characters. One of the problems of NLP is boundaries of words. For example in English words are separated by spaces, but in languages like chinese and japanese, words are not separat

__Tokenization__
Process of splitting a text into meaningful pieces of chunks 


### Week 2

__Summary__
Below are the topics which were mainly focused on week 2
- Language modelling
- Sequence taggin with probabilistic models

__Language modelling__
Language modelling is introduced as the task of predicting next word or the next sequence of words
for example from the following corpus:
*This is the house that jack built
This is the malt*
__p(house | this is the ) = c(this is the house) / c(this is the)__
the above equation gives the probability of occurance of the word "house" given that previous sequence is "this is the".

In the field of computational linguistics n-gram is a contiguous sequence of n items from a given sample of text or speech. 
when n = 1
it is called __unigram__ for n = 2 it is __bigram__ for n = 3 it is __trigram__

To evaluate how good an n-gram language model is following tools are used
__Perplexity__
Perplexity describes how well a probability distribution predicts a sample. 
Perplexity of a bi-gram model: 
-   PP(W) = N root ( product N over all I, 1 / P(Wi|Wi-1) )
basically the lower perplexity, the higher is the probability, or the better is the model.



__Hidden Markov Models__
The problem of sequence taggin is solved using hidden markov models in this lesson. The problem can be defined such that, given a sequence of tokens, the most probable sequence of labels for this tokens needs to be inferred. 
For examples: 
- Parts of speech tagging. Example: Tagging noun, verb, adjective etc.
- Named entity recognition: Name of person, organizations, dates, locations, amounts, etc.

Following approaches work well with sequence labelling task
1. Rule based model
2. Separate label classifier for each token
3. Sequence models (HMM, MEMM, CRF)
4. Neural networks

### Week 3

Below are the topics which were focused on week 3
- Word and sentence embeddings
- Topic models


### Week 4
Below are the main topics which were focused on week 4
- Statistical Machine Translation
- Encoder-Decoder-attention architecture


### Week 5
Below are the main topics which were focused on week 5 
- Natural language understanding (NLU)
- Dialogue Manager
