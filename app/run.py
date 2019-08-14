import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from plotly.graph_objs import Scatter

from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    genre_food_counts = df[['genre','food']].groupby('genre').sum()['food']
    genre_food_names = list(genre_food_counts.index)
    
    # group messages by genre
    grouped = df.groupby('genre').sum()
    grouped = grouped.drop(columns=['id'])
    gt = grouped.transpose()
    
    top_social_messages = gt.sort_values(by='social', ascending=False)['social']
    top_social_messages = top_social_messages.head(5)
    
    top_social_messages_names = list(top_social_messages.index)
    
    top_direct_messages = gt.sort_values(by='direct', ascending=False)['direct']
    top_direct_messages = top_direct_messages.head(5)
    
    top_direct_messages_names = list(top_direct_messages.index)
    
    top_news_messages = gt.sort_values(by='news', ascending=False)['news']
    top_news_messages = top_news_messages.head(5)
    
    top_news_messages_names = list(top_news_messages.index)
    
    # used to plot scatter plot of message classes
    class_ones_counts = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    class_zeros_counts = df.shape[0] - class_ones_counts
    
    class_ones_counts.sort_values(ascending = False, inplace=True)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
                
        {
            'data': [
                Bar(
                    x=list(class_ones_counts.index),
                    y=class_ones_counts,
                    name = 'Class = 1'
                ),
                Bar(
                    x=list(class_zeros_counts.index),
                    y=class_zeros_counts,
                    name = 'Class = 0'
                )
            ],

            'layout': {
                'title': 'Total labels within Classes'
            }
        },
        {
            'data': [
                Pie(
                    labels=top_social_messages_names,
                    values=top_social_messages,
                    hole=0.5
                )
            ],

            'layout': {
                'title': 'Social genre top 5 labels'
            }
        },
        
        {
            'data': [
                Pie(
                    labels=top_news_messages_names,
                    values=top_news_messages,
                    hole=0.5
                )
            ],

            'layout': {
                'title': 'News genre top 5 labels'
            }
        },
        
        {
            'data': [
                Pie(
                    labels=top_direct_messages_names,
                    values=top_direct_messages,
                    hole=0.5
                )
            ],

            'layout': {
                'title': 'Direct genre top 5 labels'
            }
        },
        
         {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()