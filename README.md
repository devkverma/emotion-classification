# Emotion Classifier

This project is an **Emotion Classifier** web application built using **Streamlit** and **PyTorch**. The model classifies emotions based on user input text, categorizing it into one of six emotions: sad, joy, love, anger, fear, and surprise.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Training](#model-training)
- [License](#license)

## Features

- **Text Input**: Users can type or paste any text into the text area.
- **Emotion Classification**: Upon clicking the classify button, the app predicts the emotion of the input text.
- **Real-time Predictions**: The model provides instant feedback on the predicted emotion.

## Technologies Used

- **Python**
- **Streamlit**: For building the web application.
- **PyTorch**: For model training and inference.
- **NLTK**: For natural language processing tasks such as tokenization, stopword removal, and lemmatization.
- **Pickle**: For loading the trained vectorizer used for text feature extraction.

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-folder>

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:

   ```bash
   pip install streamlit torch nltk
   ```

4. **Download necessary NLTK resources**:

   The script automatically downloads required NLTK resources such as stopwords and wordnet.

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open the application in your web browser at `http://localhost:8501`.

3. Type or paste text into the input area and click the "Classify" button to see the predicted emotion.

## How It Works

- **Text Preprocessing**: The application preprocesses the input text by converting it to lowercase, removing stopwords, and lemmatizing the words.
- **Vectorization**: The preprocessed text is transformed into a numerical format using a TF-IDF vectorizer stored in `vectorizer.pkl`.
- **Model Prediction**: The transformed text is fed into a trained PyTorch model, which outputs probabilities for each emotion class. The class with the highest probability is selected as the predicted emotion.

## Model Training

The model used in this application is a simple feedforward neural network that has been trained on a dataset of text labeled with emotions. The model architecture consists of:
- An input layer that receives the text features.
- Two hidden layers with ReLU activation functions.
- An output layer with six neurons corresponding to the six emotion classes.

The model's weights are saved in `model.pth`.
