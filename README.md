Fake News Detector
This repository contains a project aimed at detecting fake news using a Bidirectional Gated Recurrent Unit (BiGRU) neural network. The project includes data preprocessing, model training, and evaluation components to classify news articles as real or fake.

Repository Contents
Fake_News_Detector.ipynb: Jupyter Notebook detailing the data preprocessing steps, model architecture, training process, and evaluation metrics.
fake_news_detector_model.h5: Trained BiGRU model saved in HDF5 format.
tokenizer_Fake_or_True.pkl: Tokenizer object used to preprocess text data, saved as a pickle file.
Real_Fake_Dataset: Directory containing the dataset used for training and testing the model.
Fake_News_Detection.py: Python script providing a user-friendly interface to input news articles and receive predictions on their authenticity.
Key Features
Data Preprocessing: Utilizes pandas and numpy for data manipulation, including handling missing values, text cleaning, and tokenization.
Model Architecture: Implements a Bidirectional GRU neural network using TensorFlow and Keras, designed to capture contextual information from text data effectively.
Training and Evaluation: The model is trained on a labeled dataset and evaluated using metrics such as accuracy, precision, recall, and F1-score.
User Interface: The Fake_News_Detection.py script allows users to input news articles and receive real-time predictions on whether the news is real or fake.
Dependencies
The project requires the following Python libraries:

pandas
numpy
tensorflow
keras
scikit-learn
nltk
You can install the dependencies using the following command:
pip install pandas numpy tensorflow keras scikit-learn nltk
Usage
Clone the Repository:
git clone https://github.com/yourusername/Fake_News_Detector.git
cd Fake_News_Detector
Run the User Interface:

Execute the following command to start the user interface:
python Fake_News_Detection.py
Follow the on-screen instructions to input a news article and receive a prediction.

Dataset
The dataset used for training and testing the model is located in the Real_Fake_Dataset directory. It contains labeled news articles categorized as real or fake. Ensure that the dataset is preprocessed appropriately before training the model.

Model Training
To train the model from scratch:

Open the Fake_News_Detector.ipynb notebook.
Follow the steps outlined, including data preprocessing, model definition, training, and evaluation.
Adjust hyperparameters as necessary to improve model performance.
Evaluation
The model's performance is evaluated using:

Confusion Matrix: Visual representation of true positives, true negatives, false positives, and false negatives.
Classification Report: Includes precision, recall, F1-score, and support for each class.
These metrics provide insights into the model's effectiveness in distinguishing between real and fake news.
