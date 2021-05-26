# Udemy: Modern NLP Techniques

Two types of tasks exists in NLP:

1. Classification tasks - Spam  detection, sentiment analysis- CNN
2. Sequence to sequence tasks- translation, text memorization, chat bots etc. - Transformer



##### CNN

Google Colab file: https://colab.research.google.com/drive/1sFa6yosp2bBoB19YLQ07jn_Ic1urU836
Data link: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

##### Transformer 

 "Attention is all you need" paper: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

Transformer - Application (Eng - Fr translator):
 Google Colab file: https://colab.research.google.com/drive/1i4a26jVRVsAIfAGCe5pnl4l51B6GvjWs
 Data link: http://www.statmt.org/europarl/ for all pairs
             http://www.statmt.org/europarl/v7/fr-en.tgz for the En-Fr in the course
             https://drive.google.com/file/d/14wOcOrg8YJWXze6RvMy1cisDJKaFukff/view?usp=sharing for nonbreaking_prefix.en
             https://drive.google.com/file/d/1z4iM2n4t1guAxHZ0URfmzrjz3zknW64L/view?usp=sharing for nonbreaking_prefix_fr

__Materials:__

https://www.tfcertification.com/pages/modern-language-processing-in-python

## CNN

Standard steps for CNN:

Step 1. Create convolution feature "Feature detectors" that go through the image

Step 2: Max Pooling

Step 3: Flattening

Step 4: Full Connection (a feed forward network)

__Basic CNN__

__Convolution__

Feature detectors using strides.

Split anything into very small section. 

Input image  x Feature detector = Feature Map 

Convolutional layer consists of a bunch of feature maps:

__max pooling__

Reduces size and cost by being more global

Feature map -> pooled feature map (highest value in the stride from feature map, stored in pooled feature map)

No of Feature maps=No of pooled feature maps

__flattening__

Make everything into a single 1D vetor

Flattening-> input layerneurons



__Dense Layers__

Input layer -> fully connected layer/hidden layers -> output layer

 __Text to vector__

One hot encoding - 

Each word in a corpus will be a vector with  the length of the corpus, with mostly zeros and at the specific index it will be 1. 

__word embedding__ 

Mathematical relation between words.

Make each vector smaller -> adds relations between words.

King-man+woman= queen

Words with similar meaning will have close embeddings 

![image-20210526153730451](C:\Users\abir\AppData\Roaming\Typora\typora-user-images\image-20210526153730451.png)

__Skip gram mode__l - It finds the dimension reduction in vector representation of words while adding a semantic relation between them.



##### CNN for NLP

<img src="C:\Users\abir\AppData\Roaming\Typora\typora-user-images\image-20210526195219520.png" alt="image-20210526195219520" style="zoom:50%;" />

Main difference between CNN for vision and CNN for NLP, in NLP:

- Each filter has width = d_model. Splitting the  dimension embedding does not make sense

- one matric for each filter. position of a feature in the sentence is less important.
-  different size of filters, to capture different scale of correlation between words. 
- Output is 1D vector instead of matrix. 

__CNN for NLP: Application__

__Sentiment Analysis__

###### Stage 1: Importance dependies- 

import tensorflow, numpy, math, re, beautifulsSoup (To handle encoded data, in this case XML), layers, tensorflow_datasets

###### Stage 2: Data pre processing

- Load data from drive, the data do not have any headers. 
- Declare engine as python Encoding as laton1 

0 indicates negative sentiment
1 indicates positive sentiment

- Data cleaning 

Get rid of the unnecessary columns (data.drop id, date, query, user)

Clean tweets
Each tweet -> strings

- Use a regex to 
  -get rid or retweet
  -Get rid of links
  -get rid of numbers
  -get rid of white  spaces 

- Set data lebels to 0 and 1s 

- Tokenization 
  Sentence -> list of number
  Cleaned data now changed into input sequences.
-  Padding
  Find the maximum length of the sentences and pad all other sequences to post. 
  Split data to test and train by using random function. 



###### Stage 3: Model

Deep convolutional NN

- Embedding layer

- Conv layers - Bigram, trigram, fourgram

- Dense layers 

- dropout layer (to avoid overfitting)

- Last dense layer - a sigmoid function is used as binary classification is done


###### Stage 4: Application

Declare the hyper parameters

EMB_DIM = 200

NB_FILTERS = 100

FFN_UNITS = 256

NB_CLASSES = 2#len(set(train_labels))

DROPOUT_RATE = 0.2

BATCH_SIZE = 32

NB_EPOCHS = 5

###### Stage 5: Training

If number of classes = 2, loss= binary crossentropy

otherwise sparse_categorical _crossentropy is used.

Create a checkpoint path to save the model to use later. 

###### Stage 6: Evaluation

 Enter sentences to test. 



## Transformer

Used for sequence to sequence problem

#### Old fashioned NLP

RNN 

Encoder and decoder

Encoder encodes or summarizes the information we want to use and decoder will use the output  of the encoder as it's input and will produce the right output.

__Sequence to Sequence__:

- words are embedded into vectors
- Each new state of the encoder is computed from the previous one and the next word
- The final state of the encoder conveys the information to start decoding



Problem: information could be lost in encoding phase

To solve __attention mechanism__

It's a new global context vetor that uses all the input sequence and the last state/word  from decoder. It says how the current state of the decoder is related to the global input sequence. This improves the decoding phase. 

__The problem of RNN__

The sequential processing is not global enough and information is lost for long sentences

#### Transformer General Understanding

<img src="C:\Users\abir\AppData\Roaming\Typora\typora-user-images\image-20210526205339373.png" style="zoom:50%;" />

Still an encoder and decoder with attention mechanism.

The whole sentence is fed to the encoder at once.

The output of the decoder is fed to the decoder again for predicting the next word by shifting right. The decoder also receives output from the encoder. 

Attention mechanism is the most important part of transformers.

Self attention - used to encode sentences in the transformer.

We will see how each word of a sentence is related to other words of the sentence

![image-20210526210152708](C:\Users\abir\AppData\Roaming\Typora\typora-user-images\image-20210526210152708.png)

#### Attention Mechanism

Main Idea:

- 2 sequences (equal in case of self attention) A & B
- How each element of A is related to B
- recombine A 

Before- A given squence and a contxt B

After:  a new sequence where element is a mix of elements from A that were related to element B

__Dot Product__

provides the information about the similarity between 2 vectors. 

Matrix multiplication applied dot product to every pair of words.

<img src="https://paperswithcode.com/media/methods/SCALDE.png" alt="Scaled Dot-Product Attention Explained | Papers With Code" style="zoom:50%;" />

<img src="C:\Users\abir\AppData\Roaming\Typora\typora-user-images\image-20210526214120763.png" alt="image-20210526214120763" style="zoom:60%;" />

Q, K and V are matrices representing sentences/sequences  after embedding

QKT says how Q is related to K and V is the re-composition of V.

Softmax needed to produce valid weights



__Self attention__

beginning of encoding and decoding later. we recompose a sequence to know how each element is related to the others, grabbing a global information about a sentence/sequence



__Encoder-decoder attention__

We have the internal sequence in the decoder (Q) and a context from the encoder (K=V). our new sequence is the combination of information from our context, guided by the relation decoder sequence- encoder output. 



__Look-ahead mask__

during training, we feed a whole output sentence to the decoder, it to predict word n, it must not look at words after n.

-multiply the attention matrix with a triangle matrix- elementwise multiplication.

__Linear Projection__

Applying the attention mechanism to multiple learned subspaces.

_"Multi-headed attention allows the model to jointly attend to information from different representation subspaces at different positions. with a single attention head , averaging inhibits this"_ from the paper

Mathematically- One big linear function and then a splitting allows each subspace to compass with the full original vector. splitting and then applying a linear function restricts the possibilities.



#### Positional Encoding

In transformer there is no convolution, no recurrences.

The model has no clue about the order of the sentence. Permuting the forst and second word in every sentence would not make a difference.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT04m7Lr_LPB89a3W2uD_SIh7fD0Ndzel3wnA&usqp=CAU" alt="Dissecting the Transformer" style="zoom:50%;" />

 pos: index of a word in a sentence between 0 and seq_len-1

i: one of the dimension of the embedding between 0 and d_model-1

<img src="https://classic.d2l.ai/_images/output_transformer_ee2e4a_21_0.svg" alt="9.3. Transformer — Dive into Deep Learning 0.7 documentation" style="zoom:50%;" />



#### Feed- Forward layer

Feed forward layer in put at the end of each encoding or decoding sub layer:

- composed of 2 linear transformation
- applied to each position separately and identically
- different for each sub layer. 

FFN(x)= max(0,xW1+b1)W2+b2

 __Add & Norm or Residual connection__

helps learning during back propagation

__Dropout__

Shuts down some neurons during training to prevent overfitting.

__Last linear__

Output of the decoder goes through a dense layer with vocab_size units and a softmax layer to get probalities for each word



## Implementation

###### Stage 1: importing dependencies 

import tensorflow, numpy, math, re, beautifulsSoup (To handle encoded data, in this case XML), layers, tensorflow_datasets

###### Stage 2: Data preprocessing

- Load all the files from drive and clean data

  English data set- europarl-v7.fr-en.en

  French data set - europarl-v7.fr-en.fr

  mode r= only reading

- .Getting the non_breaking_prefixes as a clean list of words with a point at the end so it is easier to use.

- Multiple spaces clear
- Ad $$$ to non-ending sentence points
- Remove $$$ markers

Use Regex

- tokenizing text
- padded

###### Stage 3: Model building

- Hyper parameter in declared :

BATCH_SIZE = 64

BUFFER_SIZE = 20000

- Positional encoding function is declared based on the formula
- scaled dot product attention function is declared
- Multiheaded attention sublayer is created (calling self)
- encoder is defined
- decoder is defined
- Transformer is built with encoder, Decoder, Last linear

Stage 4: Training and evaluating

Hyper parameters are declared

D_MODEL = 128 

NB_LAYERS = 4 # 6

FFN_UNITS = 512 

NB_PROJ = 8

DROPOUT_RATE = 0.1 

Change to see the performance. 

Loss is SparseCategoriclaCrossentropy

Model saved/checkpoint path declared

Model is then trained on 10 epochs. 



