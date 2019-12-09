import sys
# import libraries
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer,classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
import pickle

def load_data(database_filepath):
    """
    Load Data Function

    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame (Messages)
        Y -> label DataFrame (pre-labeled columns )
        category_names -> used for data visualization (app)
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)

    #Read data from the table we created earlier and we call it 'disaster_response'
    df = pd.read_sql_table('disaster_response', engine)

    # the X is the feature we are going to use to train and test our ML model
    X = df['message']

    # the y contain result of the calssification we used to to classify each message
    y = df.drop(['id', 'message', 'original', 'genre', 'child_alone', 'related'], axis=1)

    category_names= y.columns.values

    return X, y, category_names

def tokenize(text):
    """
    Tokenize function

    Arguments:
        text -> list of text messages
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    """
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]


def build_model():
    """
    Build Model function

    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    """
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(LinearSVC(multi_class="crammer_singer"), n_jobs=1))
                    ])

    parameters = {
        'clf__estimator__C': 1,
        'clf__estimator__max_iter': 1000    }
        
    model = GridSearchCV(pipeline, param_grid=parameters)


    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function

    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and f1score)

    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names
    """
# Print out Precision , recall F1_score and support for each column using classification_report function
    y_pred_test = model.predict(X_test)
    print(classification_report(Y_test, y_pred_test, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save Model function

    This function saves trained model as Pickle file, to be loaded later.

    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file

    """
    pickle.dump( model, open( model_filepath, "wb" ) )


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
