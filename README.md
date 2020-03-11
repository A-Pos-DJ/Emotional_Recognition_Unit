#	DJ Cmar
#	A Project for CIST 2746
#	3/11/2020

#       The project presented is an emotional facial image recognition application. The application is named
#Emotional_Processing_Unit (EPU) in accordance to what the agent is designed to do. 
#The project is an attempt to implement the EmoPy module from github.com/thoughtworksarts/EmoPy 
#into Google's Colab python interpreter. When attempting to import and process data though the EmoPy module, 
#there were a few incompatibility issues that were discovered though live test runs in Colab. 
#Many of the project resources were spent to fix compatibility issues and provide workarounds for modules 
#and methods that would normally function on an offline python interpreter. Even with this being the case, 
#all FER pre-trained modules were able to be implemented successfully with little to no bugs. The method in which
# all FER models are trained is with using TensorFlow?s neural net training module ?Keras?. The pre-trained models
# used Microsoft?s FER2013 and the Extended Cohn-Kanade datasets at the time they were produced. The training data
# would then go through a convolutional neural net (CNN) algorithm while using Keras to produce models. When it 
#comes to processing an image, the CNN algorithm divides and simplifies the image as follows: 
#
#https://www.thoughtworks.com/insights/blog/emopy-machine-learning-toolkit-emotional-expression (convolutional diagram)
#
#Convolutional Layers:
#	These layers are designed to divide up a photo (or photo block) into a number of smaller blocks for the purpose 
#	of analysis on a much smaller scale.
#Pooling Layers:
#	These layers are designed to reduce input space to reduce complexity and reduce the amount of 
#	time it will take for the agent to process an image
#Flatten:
#	Convert the output of the prior layer into a one-dimensional vector
#Fully Connected:
#	Take the converted one-dimensional vector and calculate a classification output

#       When it comes to training, the CNN algorithm tied to a model still applies. 
#When training a single Epoch, The analyzed data is used to make a prediction on which emotion the user appears 
#to be experiencing. While the agent is being trained, the prediction is then compared to training data to determine
# accuracy. This process repeats for a single batch of non-trained data. The batch size is determined by the user.
# The batch size could be 10 photos, or 100, even 1000 photos. The CNN training function is careful not to let the 
#batch size become a large number, due to the possibility of overfitting. If the agent is not accurate in its
# classifications during a batch, the agent?s model then makes adjustments to provide a more accurate classification
# in the future. This process is repeated for each Epoch until the model is finished with its training. The amount
#of training data plays a huge factor in how successful the model is. The pre-trained FER Model was trained on a 
#database if 35,000 facial expressions with the classifications of 
#anger, disgust, fear, happiness, sadness, surprise, and calm. 
#Each FER model was trained with only up to four classifications in mind at one time. If a user wanted to see more
# than a specific set of emotions, the user would need to create their own CNN model. Creating a model is something
# that the EmoPy had capabilities of. The only issue is that the original CNN training method was not compatible with
# Colab. The EPU is currently lacking the ability to upload and provide additional training data for the
# ConvModelTrainer to be train properly. 

#Future Goals: 
#1. Provide the EPU with the ability to load other users. pre-trained models from the EmoPy github and/or search
# for a directory to be uploaded. 
#2. Provide the same ability for the ConvModelTrainer to upload datasets or attach a dataset link for processing.
#3. Implement the time-delay convolutional model training/live classification
