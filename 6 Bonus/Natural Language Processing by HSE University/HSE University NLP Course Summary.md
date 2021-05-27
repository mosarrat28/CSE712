# Introduction to NLP

### Week 1
___
Below are the main topics which were focused on week 1
- From plain texts to classification
- Simple deep learning for text classification

__Text Preprocessing__
__What is "Text"__
First step of text preprocessing is to define what is "Text". So, "Text" can be defined as a sequence. Where it can be sequence of anything for example, sequence of characters or words, or  phrases, or even sentences and paragraphs. 

__What is "Word"__
Let's for now, define "Text" as sequence of words. Then again, "Word" can be described as __meaningful__ sequence of characters. One of the problems of NLP is boundaries of words. For example in English words are separated by spaces, but in languages like chinese and japanese, words are not separat

__Tokenization__
Process of splitting a text into meaningful pieces of chunks.
The python NLTK library contains useful tokenizer modules, such as 
*WhitespaceTokenizer()*
*TreebankWordTokenizer()*
*WordPunctTokenizer()*

Almost every NLP task requires preprocessing that uses techniques such as
- Normalization
- Stemming
- Lemmatization

The next step is to convert the tokens into features, and one of the way to do that is __text vectorization__. The method Bag of words uses text vectorization to turn tokens into features. Where the each token has a feature column, the test sentences are used to fill up the column with number of occurance of the token. 

These counted frequencies as columns can be replaced by TF(Term Frequency)-IDF(Inverse Document Frequency) values to have better performing Bag of words model.


### Week 2
---
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

__Smoothing__
ML model may estimate probability as 0 if they face unseen words, to solve this data sparsity problem smoothing is used. 
Following smoothing techniques are discussed:
1. Add-one or (Add-k) smoothing
2. Laplacian Smoothing
3. Katz backoff
4. Interpolation Smoothing
5. Absolute discounting
6. Kneser-Ney smoothing

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
___
__Summary__
Below are the topics which were focused on week 3
- Word and sentence embeddings
- Topic models
- GloVe
- Skip-gram models

__Word similarities__ 
Similarity between two words can be described as following:
First order co-occurences / relatedness : bee and honey
Second order  co-occurences / similarity: bee and bumblebee
or __syntagmatic__ and __paradigmatic__ relatedness.

__Skip-gram__ model is used to predict context words given a focus word. Each probability is then modeled using softmax.

__Topic modeling__ is described as a way to navigate through text collections. 


### Week 4
___
Below are the main topics which were focused on week 4
- Statistical Machine Translation
- Encoder-Decoder-attention architecture

**Machine translation** is described using parallel corpora, such as movie subtitles, translated books, news, wikipedia.
But the problems of machine translation using parallel corpora is as following:
- Noisy
- Specific Domain
- Rare language pairs etc.

**BLEU** score is a popular way to evaluate two arbitrary translations. 

Word alignments are very important for translation.	__Encoder-Decoder__ architecture is introduced. 
__Encoder__ maps the source sequence to hidden vector
__Decoder__ performs language modelling given the vector

__Attention Mechanism__ is used between encoder decoder architecture in generative models. The different types of attention mechanisms are: 
- Additive Attention
- Multiplicative Attention
- Dot Product 

Various methods of dealing with vocabularies are compared, which are __Softmax, Hierarchial Softmax__ 

Different methods of implementing a chat bot is discussed. A chat bot can be classified in following ways:
- Goal Oriented
- Chit
Chat bot can be build in either Retrieval-based or Generative ways.

### Week 5
___
Below are the main topics which were focused on week 5 
- Natural language understanding (NLU)
- Dialogue Manager

Task oriented dialog systems are discussed. For example talking to some of the available personal assistants, such as: 
- Apple Siri
- Google Assistant
Following tasks can be solved:
- Setting up a reminder
- Finding photos of pet
- Finding good restaurants

Dialog systems start with Intent Classification. There can be many different types of intents, and these intent needs to be classified and tagged to the correct answers. 

Form filling approach can be used for Dialogue managment. Intent classifier and slot taggers is used for dialogue based chat bot. Another important task is to track context.

__Context__ is used in __NLU__ to handle multi-turn dialogues. For example the following input: 
__Give me directions from Los Angeles__
- Intent classifier: nav.directions
- Slot tagger: @FROM{Los Angeles}
- Dialogue manager: required slot is missing, where to?
- User: San Francisco
- Intent classifier: nav.directions
- Slot tagger: @TO{San Francisco}
- Dialogue manager: Okay, here's the route
