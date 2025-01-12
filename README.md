
# Data Science Portfolio: Tweet Analysis

This repository contains two Python scripts demonstrating data science skills in tweet analysis. These projects were originally developed as part of Codecademy's portfolio-building exercises and have been refined for showcasing to potential recruiters.

## Scripts Overview

### 1. viral_tweets.py
- **Objective:** Predict whether a tweet will go viral based on its features such as length, follower count, and retweet count.
- **Techniques Used:** 
  - Feature engineering
  - Data preprocessing (scaling)
  - K-Nearest Neighbors (KNN) classifier to determine optimal `k` value.
- **Libraries:** pandas, numpy, scikit-learn, matplotlib

### 2. tweet_location.py
- **Objective:** Classify tweets' geographic origin (New York, London, or Paris) based on their text content.
- **Techniques Used:** 
  - Text preprocessing and vectorization using CountVectorizer
  - Naive Bayes classification for text data.
- **Libraries:** pandas, scikit-learn

## Project Highlights
- These scripts showcase practical applications of data preprocessing, feature engineering, and machine learning algorithms.
- The `viral_tweets.py` script demonstrates predictive modeling and hyperparameter tuning using KNN.
- The `tweet_location.py` script highlights text-based classification with Naive Bayes.

## How to Use
1. Clone the repository.
2. Ensure you have Python 3.6+ installed.
3. Install the required libraries using `pip install -r requirements.txt`.
4. Place the necessary JSON datasets (`random_tweets.json`, `new_york.json`, `london.json`, `paris.json`) in the same directory as the scripts.
5. Run the scripts:
   - `python viral_tweets.py`
   - `python tweet_location.py`

## About
These projects are part of Codecademy's data science portfolio, completed to build foundational skills in data analysis and machine learning.

---

Feel free to explore the scripts and reach out with any questions or suggestions!

**Author:** Rayan Roshan  
