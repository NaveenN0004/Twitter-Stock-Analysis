import tweepy  # type: ignore
import pandas as pd  # type: ignore
import os
import time

def twitter_auth(bearer_token):
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    return client

def fetch_tweets(api, query, max_tweets=500):
    tweets = []
    max_results_per_request = 100
    num_requests = (max_tweets + max_results_per_request - 1) // max_results_per_request

    try:
        for _ in range(num_requests):
            response = api.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, max_results_per_request),
                tweet_fields=['author_id', 'created_at', 'public_metrics', 'text']
            )
            if response.data is not None:
                for tweet in response.data:
                    tweets.append({
                        'author_id': tweet.author_id,
                        'created_at': tweet.created_at,
                        'text': tweet.text,
                        'likes': tweet.public_metrics['like_count'],
                        'retweets': tweet.public_metrics['retweet_count']
                    })
            else:
                print("No data returned from Twitter API.")
                break

            max_tweets -= max_results_per_request
            if max_tweets <= 0:
                break
    except tweepy.TooManyRequests:
        print("Rate limit exceeded. Waiting before retry...")
        time.sleep(15 * 60)  # 15-minute wait as a fallback
    except Exception as e:
        print(f"Error Fetching tweets: {e}")
    return tweets

def save_to_csv(tweets, filepath):
    df = pd.DataFrame(tweets)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    print("Tweet scraping and saving completed successfully!")

if __name__ == "__main__":
    BEARER_TOKEN = "YOUR_BEARER_TOKEN"  # Replace with your bearer token
    api = twitter_auth(BEARER_TOKEN)

    query = "stocks OR #trading OR #investing"
    max_tweets = 500
    tweets = fetch_tweets(api, query, max_tweets)

    filepath = "C:/Users/matha/OneDrive/Desktop/Twitter_Stock Analysis/data/tweets.csv"
    save_to_csv(tweets, filepath)
