# Outline (Draft)

- Part I: Introduction

    - Intro to ANN 
        - (naive pure-Python implementation from `pybrain`)
        - fast forward
        - sgd + backprop
    - Intro to Theano
        - Model + SGD with Theano (simple logreg) 
        
    - Introduction to Keras
        - Overview and main features
            - Theano backend
            - Tensorflow backend
        - Same LogReg with Keras

- Part II: Supervised Learning + Keras Internals
    - Intro: Focus on Image Classification
    - Multi-Layer Perceptron and Fully Connected
        - Examples with `keras.models.Sequential` and `Dense`
        - HandsOn: MLP with keras 
        
    - Intro to CNN
        - meaning of convolutional filters
            - examples from ImageNet
        
        - Meaning of dimensions of Conv filters (through an exmple of ConvNet) 
        - HandsOn: ConvNet with keras 

    - Advanced CNN
        -  Dropout and MaxPooling
    - Famous ANN in Keras (likely moved somewhere else)
        - ref: https://github.com/fchollet/deep-learning-models
        - VGG16
        - VGG19
        - LaNet
        - Inception/GoogleNet
        - ResNet
        *Implementation and examples 
        - HandsOn: Fine tuning a network on new dataset 
        
- Part III: Unsupervised Learning + Keras Internals
    - AutoEncoders
    - word2vec & doc2vec (gensim) + `keras.dataset` (i.e. `keras.dataset.imdb`)   
    - HandsOn: _______

*should we include embedding here? 

- Part IV: Advanced Materials
    - RNN (LSTM)
        -  RNN, LSTM, GRU  
        -  Meaning of dimensions of rnn (backprop though time, etc)
        -  HandsOn: IMDB (?)

    - CNN-RNN
    - Time Distributed Convolution 
    - Some of the recent advances in DL implemented in Keras
        - e.g. https://github.com/snf/keras-fractalnet - Fractal Net Implementation with Keras


Notes:

1) Please, add more details in Part IV (i.e. /Advanced Materials/)
2) As for Keras internals, I Would consider this: https://github.com/wuaalb/keras_extensions/blob/master/keras_extensions/rbm.py
This is just to show how easy it is to extend Keras ( in this case, properly creating a new `Layer`).
