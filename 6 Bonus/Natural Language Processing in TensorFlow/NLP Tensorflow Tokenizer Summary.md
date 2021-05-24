# NLP Tensorflow Tokenizer Summary

### Week 1

__Tokenizer__
Tokenization- needed for changing words to numbers to be used in NN

Tokenizer - generated  the dictionary of word encodings and creates vectors out of sentences. 

tokenizer=Tokenizer(num_word=20)
tokenizer.fit_on_texts()
tokenizer.word_index

Tokenizer(numwords=100, oov-token="<oov>")

Sequence:

sequences = tokenizer.texts_to_sequences(training_sentences) -------- for turning into sequence of numbers with words being the key and number being the value



Padding:

padded=pad_sequence(sequences, padding='post', maxlen=5)
padding and trancuting is both done as "pre" by default, 
maxlen is used if i want my sentences with 5 words.

## Week 2

__Word Embedding__

IMDB reviews datasets

Word embedding steps

1. download dataset

2. load into arrays

3. change into numpy arrays

4. hyperparameter tuning- 

   -vocab_size = 10000

   -embedding_dim = 16

   -max_length = 120

   -trunc_type='post'

   -oov_tok = "<OOV>"

5. Tokenize the training sentences 

   -tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

   -tokenizer.fit_on_texts(training_sentences)

   -word_index = tokenizer.word_index

   -sequences = tokenizer.texts_to_sequences(training_sentences)

   -padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

6. Tokenize the testing sequence

   -the testing sequences are tokenized based on the word index that was earned from the training   words.

   -padded

7. neural network training

8. write into tsv files word and it's corresponding vector (vector and meta)

__Sarcasm Classification__

Data- Training and testing sentences and labels are loaded into 4 separate arrays

Classifier- NN build

Loss- change the hyper parameter to improve loss



__subwords tokenizer__

_Tensorflow dataset_

https://github.com/tensorflow/datasets/tree/master/docs/catalog

https://github.com/tensorflow/datasets/blob/master/docs/catalog/imdb_reviews.md



 The shape of the vectors coming from the tokenizer through the embedding, and it's not easily flattened.  Global Average Pooling 1D is used.  Trying to flatten them, will cause a TensorFlow crash. 



## Week 3   

![image-20210522213825842](C:\Users\abir\AppData\Roaming\Typora\typora-user-images\image-20210522213825842.png )

RNN - Recurrent Neural Network similar to fibonacci

<img src="C:\Users\abir\AppData\Roaming\Typora\typora-user-images\image-20210522214339141.png" alt="image-20210522214339141" style="zoom:50%;" />

x=input y=output and an element from the previous function

__LSTM__

Long Short Term Memory

To build an LSTM add an extra layer to a sequential model replace flatten layer and maxpooling layer with birectional layer. 

-tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), --> This will make cell state go in both direction

-tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)), Stacked LSTM.

An LSTM is fed into another one( return_sequences=True ) parameter into the first one. This ensures that the outputs of the LSTM match the desired inputs of the next one. 

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 64)          523840    
_________________________________________________________________
bidirectional (Bidirectional (None, 128)               66048     
_________________________________________________________________
dense (Dense)                (None, 64)                8256      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65        
=================================================================
Total params: 598,209
Trainable params: 598,209
Non-trainable params: 0
```

Loss comparison

![image-20210522234424945](C:\Users\abir\AppData\Roaming\Typora\typora-user-images\image-20210522234424945.png)

Accuracy is better in multilayer RNN, with less dips. 

loss in validation

 increases epochs by epochs

_GRU_ Graded Recurrent unit

LSTM increased accuracy on sarcasm dataset

But overfitting in LSTM caused confidence to decrease

__CNN__

Convolutional Neural Network

Loss increases in the validation set shows potential overfitting 

we'll see that we have 128 filters each for 5 words

__Comparison between different training models__

Sequential model

_with flatten layer_- time 5sec/epoch,  clear overfitting, 171,533 parameters, nice accuracy

_with LSTM layer_-time 43sec/epoch,  clear overfitting, 30,129  parameters, better accuracy

_with GRU layer_-time 20sec/epoch,  some overfitting, 169,997 parameters, very good accuracy

_with CNN layer_- time 5sec/epoch,  clear overfitting, 171,149  parameters, accuracy almost 100%, 83% on validation

There is more __over-fitting__ in text dataset than image dataset because of oov tokens.  words in the validation dataset that weren't present in the training, leads to overfitting. 

## Week 4 - Sequence model and literature

Predict the next word in a sentence

