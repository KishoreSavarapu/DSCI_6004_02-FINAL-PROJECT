# DSCI_6004_02-FINAL-PROJECT by Amani Kambham and Lakshmi Sai Kishore Savarapu

# Project: EMOTION CLASSIFICATION OF TWEETS USING NATURAL LANGUAGE PROCESSING


# Overview:

This Natural Language Processing project aims to identify the emotion of the given text sentence using various Natural Language Processing Techniques and networks. The project explores and compares the performance of few different NLP models built using Vanilla Recurrent Neural Network, Long Short Term Memory, Gated Recurrent Unit, and Bidirectional Vanilla Recurrent Neural Network. Another objective of this project is to build models using the same networks but by training them by feeding the input in the reverse order. This technique is implemented to check if there is any improvement in the performance of the model by training it in the reverse order.


# Project Components:


# Dataset:

5 V’s of Data; Volume, Variety, Velocity, Veracity, Value are checked to ensure right set of data is collected for the problem statement. The dataset has 40,000 samples and 13 emotions with tweet_id, sentiment, content as features. It is a labeled and structured dataset.

Dataset link:  https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text


# Pre-Processing of Dataset:

Some of the Text cleaning and pre-processing like Word Tokenization, Label encoding of the target variable, removal of punctuations, finding number of unique words in the document, padding of the input sentences, etc; are the techniques performed to make the process smooth and effective. 




# Natural Language Processing Models:

Vanilla Recurrent Neural Network: A Vanilla Recurrent Neural Network (RNN) is a fundamental type of RNN that is particularly suited for processing sequences of data, such as text for sentiment classification.

Long Short Term Memory: Long Short-Term Memory (LSTM) networks are an advanced type of Recurrent Neural Network (RNN) specifically designed to address the limitations of traditional RNNs, such as the vanishing and exploding gradient problems. These issues make it difficult for Vanilla RNNs to learn long-term dependencies in sequence data. LSTMs are particularly effective for tasks requiring the understanding of long-range contextual information, making them well-suited for sentiment classification in text data.

Gated Recurrent Unit: Gated Recurrent Units (GRUs) are a type of recurrent neural network (RNN) architecture, introduced as a simplified alternative to Long Short-Term Memory (LSTM) networks. They have gained popularity in various natural language processing (NLP) tasks, including sentiment classification, due to their ability to efficiently model sequential data and capture long-term dependencies with fewer parameters than LSTMs.

Bidirectional Vanilla RNN: A Bidirectional Vanilla RNN combines the traditional structure of a Vanilla Recurrent Neural Network (RNN) with bidirectional processing of input sequences. This approach allows the network to have both forward and backward information about the sequence at every point, enhancing its ability to capture context and improving performance on tasks like sentiment classification.



# Process 1 (step by step):

The implementation is done on Google Colab platform.

Importing all necessary libraries and modules.

Importing the dataset by accessing the link.

Converting images into dataframes.

Performing Exploratory Data Analysis.

Implementation of Text cleaning and pre-processing techniques like Word Tokenizaion, removal of punctuations, padding the input sequences, label encoding the target variable, finding the total number of unique tokens in the document, etc.

Train-test splitting of the dataframe.

Feature extraction of the dataframe.

Model building for the task. As this is a multiclass classification problem, softmax activation function is used at the output layer. 

Training the models.

Evaluation of models by calculating the Accuracy, precision, recall for the classification task. Classification report, Confusion matrix can also be obtained during this phase.

Comparing the performance of all the models.



# Process 2 for training the models with reversed text sentences(step by step):

The implementation is done on Google Colab platform.

Importing all necessary libraries and modules.

Importing the dataset by accessing the link.

Converting images into dataframes.

Reversing the text sentences in the dataframe.

Performing Exploratory Data Analysis.

Implementation of Text cleaning and pre-processing techniques like Word Tokenizaion, removal of punctuations, padding the input sequences, label encoding the target variable, finding the total number of unique tokens in the document, etc.

Train-test splitting of the dataframe.

Feature extraction of the dataframe.

Model building for the task. As this is a multiclass classification problem, softmax activation function is used at the output layer. 

Training the models.

Evaluation of models by calculating the Accuracy, precision, recall for the classification task. Classification report, Confusion matrix can also be obtained during this phase.

Comparing the performance of all the models.

Comparing the performance of all the models with the models built during the process 1 to check for any improvement in models performance for training them with reversed text sentences.



# Conclusion:

This project offers a comprehensive exploration of EMOTION CLASSIFICATION OF TWEETS USING NATURAL LANGUAGE PROCESSING, providing a valuable resource for researchers and students interested in Natural Language Processing, Text processing, etc. The comparison of different models offers insights into their strengths and weaknesses, guiding future research in the field. The well-documented code and usage instructions ensure accessibility and reproducibility for other researchers and enthusiasts.
