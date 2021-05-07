# NLP Tensorflow Tokenizer Summary

### Week 1

__Tokenizer__
Tokenization- needed for changing words to numbers to be used in NN

Tokenizer - generated  the dictionary of word encodings and creates vectors out of sentences. 

tokenizer=Tokenizer(num_word=20)
tokenizer.fit_on_texts()
tokenizer.word_index

Tokenizer(numwords=100, oov-token="<oov>")
numbwords- parameter that takes the most common words from the body of input.
oov-token - to use when the word is not found in the dictionary

Padding:

padded=pad_sequence(sequences, padding='post', maxlen=5)
padding and trancuting is both done as "pre" by default, 
maxlen is used if i want my sentences with 5 words.

## Week 2

__Word Embedding__





