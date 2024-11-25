import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def prepare_data(df):
    features = df[['sentiment', 'text_length']]
    
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    features['sentiment'] = features['sentiment'].map(sentiment_map)
    
    target = (features['sentiment'] == 1).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

# SVM model
def svm_model(X_train, y_train, kernel='linear', C=1.0):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVC(kernel=kernel, C=C)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def evaluate_model(X_train, y_train, X_test, y_test):
    kernel = 'linear'  
    C = 1.0            
    
    model, scaler = svm_model(X_train, y_train, kernel, C)
    
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    
    results_filepath = "C:/Users/matha/OneDrive/Desktop/Twitter_Stock Analysis/results/evaluation_results.txt"
    with open(results_filepath, "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, predictions)}\n")
        f.write(f"Classification Report:\n{classification_report(y_test, predictions)}\n")
    print(f"Evaluation results saved to {results_filepath}")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    df = load_data("C:/Users/matha/OneDrive/Desktop/Twitter_Stock Analysis/data/preprocessed_tweets.csv")
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    evaluate_model(X_train, y_train, X_test, y_test)
