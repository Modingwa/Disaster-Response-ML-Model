# Load Libraries
import pandas as pd
import numpy as np
import re
import warnings
import sys
import pickle

from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    """Function to load data from database

    Parameters
    ----------
    database_filepath : str
        Path to database file

    Returns
    -------
    Pandas DataFrames
        The dataframes consists of messages, categories features and category names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)

    X = df['message']
    Y = df.iloc[:, 4:]

    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """Function to tokenize text

    Parameters
    ----------
    text : str
        String to tokenize

    Returns
    -------
    List
        List of tokenized words
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """Function to build a machine learning pipeline
    The pipeline takes in the message column as input and output classification results on the other 36 categories in the dataset
    Parameters
    ----------
    none

    Returns
    -------
    sklearn.pipeline.Pipeline
        sklearn.pipeline.Pipeline object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    # use the model to make predictions
    y_pred = model.predict(X_test)

    # calculate overall prediction accuracy
    overall_accuracy = (y_pred == Y_test).mean().mean() * 100

    # convert y_pred to dataframe for convinience
    y_pred = pd.DataFrame(y_pred, columns=Y_test.columns)

    for col in Y_test.columns:
        print('Category feature : {}'.format(col.capitalize()))
        print('.................................................................\n')
        print(classification_report(Y_test[col], y_pred[col]))
        accuracy = (y_pred[col].values == Y_test[col].values).mean().mean() * 100
        print('Accuracy: {0:.1f} %\n'.format(accuracy))

    # print overall model accuracy
    print('Overall Accuracy: {0:.1f} %'.format(overall_accuracy))
    pass


def save_model(model, model_filepath):
    """Pickle model to the given file path
    Parameters
    ----------
    model   : model object
        fitted model

    model_filepath: str
        File path to save the model to
    Returns
    -------
    none
    """
    with open(model_filepath, 'wb') as f:
        # Pickle the 'model' to disk
        pickle.dump(model, f)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()