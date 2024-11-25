import pandas as pd # type: ignore
import re
from textblob import TextBlob # type: ignore

def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  
    tweet = re.sub(r'@\w+', '', tweet)    
    tweet = re.sub(r'#', '', tweet)       
    tweet = re.sub(r'\n', ' ', tweet)     
    tweet = re.sub(r'[^\w\s]', '', tweet) 
    return tweet

def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity == 0:
        return "neutral"
    else:
        return "negative"

def preprocess_data(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath)

    
    df['cleaned_text'] = df['text'].apply(clean_tweet)

    
    df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)

    
    df.to_csv(output_filepath, index=False)
    print(f"Preprocessed data saved to {output_filepath}")
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    
    input_filepath = "C:/Users/matha/OneDrive/Desktop/Twitter_Stock Analysis/data/tweets.csv"
    output_filepath = "C:/Users/matha/OneDrive/Desktop/Twitter_Stock Analysis/data/processed_tweets.csv"

    preprocess_data(input_filepath, output_filepath)