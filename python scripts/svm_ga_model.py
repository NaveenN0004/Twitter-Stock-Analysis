import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from deap import base, creator, tools, algorithms  # type: ignore
import random

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

def evaluate(individual, X_train, y_train):
    kernel = individual[0]
    C = individual[1]
    
    model, scaler = svm_model(X_train, y_train, kernel, C)
    X_test_scaled = scaler.transform(X_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_train, predictions)
    
    return accuracy, 

def run_genetic_algorithm(X_train, y_train):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_kernel", random.choice, ['linear', 'poly', 'rbf'])
    toolbox.register("attr_C", random.uniform, 0.1, 10.0)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_kernel, toolbox.attr_C), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, X_train=X_train, y_train=y_train)

    population = toolbox.population(n=10)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=5, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    return best_individual

def final_model(X_train, y_train, X_test, y_test, best_individual):
    kernel = best_individual[0]
    C = best_individual[1]
    
    model, scaler = svm_model(X_train, y_train, kernel, C)
    
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    
    results_filepath = "C:/Users/matha/OneDrive/Desktop/Twitter_Stock Analysis/results/svm_results.txt"
    with open(results_filepath, "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, predictions)}\n")
        f.write(f"Classification Report:\n{classification_report(y_test, predictions)}\n")
    print(f"Results saved to {results_filepath}")
    print("Successful Analysis!")

if __name__ == "__main__":
    df = load_data("C:/Users/matha/OneDrive/Desktop/Twitter_Stock Analysis/data/preprocessed_tweets.csv")
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    best_individual = run_genetic_algorithm(X_train, y_train)
    
    final_model(X_train, y_train, X_test, y_test, best_individual)
