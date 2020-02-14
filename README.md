# Binary review classification with text mining 
## 1. Introduction
This is an open-ended project which starts from exploratory data analysis (EDA) of basic statistics and properties, reports findings of interests. Then identify a predictive task that can be studied on this dataset. Define the measurements of model performance at this predictive task and set a baseline model. Next, come up with suitable machine learning models and make optimization. Finally, compare the results and make conclusion.
* [Dataset](#dataset)
* [Exploratory Data Analysis](#eda)

## 2. Dataset<a name="dataset"></a>
This dataset comes from github page containing **Amazon Review Data** created by
[Jianmo Ni](https://nijianmo.github.io/amazon/index.html). Considering the computation cost and training time, I will simply use the 5-core (77,071 reviews) **Industrial and Scientific** category from **"Small" subsets for experimentation**, which can be found at the lower part of the page. 

**Sample data format: JSON**
```python
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "vote": 5,
  "style": {
    "Format:": "Hardcover"
  }
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```
- `reviewerID` - ID of the reviewer, e.g. A2SUAM1J3GNN3B
- `asin` - ID of the product, e.g. 0000013714
- `reviewerName` - name of the reviewer
- `vote` - helpful votes of the review
- `style` - a disctionary of the product metadata, e.g., "Format" is "Hardcover"
- `reviewText` - text of the review
- `overall` - rating of the product
- `summary` - summary of the review
- `unixReviewTime` - time of the review (unix time)
- `reviewTime` - time of the review (raw)

## 3. Exploratory data analysis <a name="eda"></a>

## 4. Preductive task
Pre-analyzed ratings distribution histogram, count of reviews and average rating over month plot.

### 4.1 Baseline: Latent-factor model

## 5. Feature engineering

### 5.1 Bag of words: unigram and bigram

### 5.2 TF-IDF vectorizer

### 5.3 Dimensionality reduction: SVD/LSA

## 6 Machine learning models

### 6.1 Naive Bayes

### 6.2 Logistic Regression

### 6.3 Random Forest

## 7 Conclusion

