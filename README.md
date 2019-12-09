# Disaster Response Pipeline Project

![Intro Pic](screenshots/main.png)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installing)
	3. [Executing Program](#executing)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Screenshots](#screenshots)

<a name="descripton"></a>
## Description
This project is the fifth project of Data Science Nanodegree provided by Udacity and Figure Eight.
The project contains 2 dataset consist of Message and Categories. We combine both file to have one csv file
containing pre-labelled tweets that have been collected from real-life disaster.
The purpose of this project is to build Natural Language Processing tool using Machine Learning techniques
that categories and route each message to correct team.

The Project is divided in the three Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
2. Machine Learning Pipeline to train a model able to classify text message in categories
3. Web App to show model results in real time.

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.7
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
git clone https://github.com/HasanTariq/Data-Science-Disaster-Response-Pipeline.git
```
<a name="executing"></a>
### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



<a name="authors"></a>
## Authors

* [Hassan Jummah](https://github.com/HasanTariq)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing complete course that has interesting projects
* [Figure Eight](https://www.figure-eight.com/) for providing messages dataset to train my model

<a name="screenshots"></a>
## Screenshots


![Death Barcharts](screenshots/barchart_1.png)


![Distribution of Message](screenshots/barchart_2.png)


![Distribution of Target Categories](screenshots/barchart_1.png)
