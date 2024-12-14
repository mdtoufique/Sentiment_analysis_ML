# Movie Sentiment Analysis

This project is a machine learning-based analysis of movie reviews to determine their sentiment (positive or negative). It uses the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) as the primary dataset for training and testing.

## Project Overview

The goal of this project is to classify movie reviews as positive or negative based on their textual content. This is achieved by implementing machine learning algorithms and natural language processing (NLP) techniques.

## Dataset

The dataset contains 50,000 IMDB movie reviews categorized as positive or negative. It is a balanced dataset, with an equal number of reviews for each sentiment class.

### Dataset Features:
- `review`: The text of the movie review.
- `sentiment`: The sentiment associated with the review (positive/negative).



## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the `data/` folder.

2. Run the Jupyter notebooks provided in the repository:
   - `bayes.ipynb`: Implements sentiment analysis using the Naive Bayes classifier.
   - `knn.ipynb`: Implements sentiment analysis using the K-Nearest Neighbors algorithm.
   - `by_built_in_func.ipynb`: A demonstration of sentiment analysis using built-in Python functions and libraries.


## Key Features

- Preprocessing of textual data using techniques such as tokenization, stemming, and stop-word removal.
- Implementation of various machine learning models:
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- Comparison of model performance metrics.

## Results

The performance of each model was evaluated using metrics such as accuracy and confusion matrices. The results are as follows:

- **Naive Bayes**: Achieved an accuracy of **83%**.
- **K-Nearest Neighbors (KNN)**: Achieved an accuracy of **68.34%**.

The details can be found in their respective notebooks.

## Requirements

- Python 3.7+
- Jupyter Notebook
- pandas
- scikit-learn
- nltk
- numpy


## Contributing

Contributions are welcome! If you have any suggestions or find any issues, feel free to open an issue or submit a pull request.



## Acknowledgements

- Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Libraries: scikit-learn, nltk, pandas, numpy.
