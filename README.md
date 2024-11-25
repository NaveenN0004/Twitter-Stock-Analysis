# Twitter Stock Analysis

This project analyzes Twitter sentiment to predict stock movements using sentiment analysis and machine learning techniques, including Support Vector Machines (SVM) optimized with a Genetic Algorithm (GA).

## Project Structure

### Files
1. **`data_scrapper.py`**:
   - Scrapes tweets related to stock trading using Twitter API.
   - Saves raw tweet data to a CSV file.
   - **Input**: Twitter API credentials and a query string.
   - **Output**: `tweets.csv` in the `data` folder.

2. **`data_preprocessor.py`**:
   - Cleans and preprocesses raw tweet data.
   - Performs sentiment analysis to label tweets as positive, neutral, or negative.
   - **Input**: `tweets.csv` (raw data).
   - **Output**: `processed_tweets.csv` in the `data` folder.

3. **`svm_ga_model.py`**:
   - Trains an SVM classifier optimized using a Genetic Algorithm (GA).
   - Saves the training results to a text file.
   - **Input**: `processed_tweets.csv`.
   - **Output**: Results saved to `results/svm_results.txt`.

4. **`model_evaluate.py`**:
   - Evaluates the SVM model's performance on test data.
   - Saves evaluation metrics to a text file.
   - **Input**: `processed_tweets.csv`.
   - **Output**: Results saved to `results/evaluation_results.txt`.

### Folders
- **`data`**: Contains raw and processed tweet data files.
  

## Workflow
1. **Scrape Tweets**:
   - Use `data_scrapper.py` to collect tweets about stocks and save them in the `data` folder.

2. **Preprocess Tweets**:
   - Run `data_preprocessor.py` to clean the tweets and perform sentiment analysis.

3. **Train the Model**:
   - Use `svm_ga_model.py` to train an SVM model optimized with GA.

4. **Evaluate the Model**:
   - Run `model_evaluate.py` to test the model's performance and save evaluation metrics.

## Requirements
Install the following Python libraries before running the scripts:

```bash
pip install tweepy pandas textblob scikit-learn deap
