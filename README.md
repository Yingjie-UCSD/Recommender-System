# Binary review classification with text mining 
## 1. Introduction
This is an open-ended project which starts from exploratory data analysis (EDA) of basic statistics and properties, reports findings of interests. Then identify a predictive task that can be studied on this dataset. Define the measurements of model performance at this predictive task and set a baseline model. Next, come up with suitable machine learning models and make optimization. Finally, compare the results and make conclusion.
* [Dataset](#dataset)
* [Exploratory Data Analysis](#eda)
* [Predictive task](#predictivetask)
  * [Baseline: Latent-factor model](#baseline)
* [Feature engineering](#feature)
  * [Bag of words: unigram and bigram](#bag)
  * [TF-IDF vectorizer](#tfidf)
  * [Dimensionality reduction: SVD/LSA](#lsa)
* [Machine learning models](#ml) 
  * [Naive Bayes](#bayes)
  * [Logistic Regression](#lr)
  * [Random Forest](#rf)
* [Conclusion](#conclusion)

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

## 3. Exploratory data analysis<a name="eda"></a>
To summarize the main characteristics of basic statistcs, such as mean, median, variance, I often make EDA with visual methods, such as histogram for counts, scatter plot for continuous statistics, etc.
### 3.1 Distribution histogram over rating
From the distribution of ratings, it is obviously more 5-star ratings and 4-star ratings than others. There is probably an **imbalanced distribution problem**.

<p align="center">
 <img src="https://github.com/Yingjie-UCSD/Recommender-System/blob/master/image/Star%20Rating%20Distribution.png" width="480">
</p>



### 3.2 Seasonality
In the context of product review data, it is important to figure out whether there exists seasonality or not, which may influence the feature engineering process. 

![seasonality over year](https://github.com/Yingjie-UCSD/Recommender-System/blob/master/image/seasonality%20over%20year.png)

![seasonality over month](https://github.com/Yingjie-UCSD/Recommender-System/blob/master/image/seasonality%20over%20month.png)
## 4. Predictive task<a name="predictivetask"></a>
**Labels** 
- reviews with rating greater than 4 stars as positive
- reviews with rating less than 4 stars as negative

**Predictive task:** classify reviews into positive and negative.

**Problems in dataset**
- Imbalanced data problem
- (Text Mining) High dimensionality problem

### 4.1 Baseline: Latent-factor model<a name="baseline"></a>



## 5. Feature engineering<a name="feature"></a>

### 5.1 Bag of words: unigram and bigram<a name="bag"></a>
**Unigrams** without punctuation and stopwords
<p align="left">
<img src="https://github.com/Yingjie-UCSD/Recommender-System/blob/master/image/unigram-positive.png" width="400">
<img src="https://github.com/Yingjie-UCSD/Recommender-System/blob/master/image/unigram-negative.png" width="400">
</p>

* Left figure: 80 most common unigrams in positive reviews.
* Right figure: 80 most common unigrams in negative reviews.

From these two word cloud, notice that many of the common unigrams are mostly the same. That is probably because people tends to express their dissatisfaction in a less offensive way. For example, in negative reviews, people would like to use ‘not so good’ rather than ‘just so bad’ to express their disappointment, which will means the ungrams model may be inaccurate.

Unigrams lose information on the orders and combinations. Therefore, I tried adding bigrams into bag of words, which is assumed to be a better choice. In the following sections, there will be comparison between the results from unigram model
and bigram model.

**Unigrams & Bigrams** without punctuation and stopwords
<p align="left">
<img src="https://github.com/Yingjie-UCSD/Recommender-System/blob/master/image/mixgram-positive.png" width="400">
<img src="https://github.com/Yingjie-UCSD/Recommender-System/blob/master/image/mixgram-negative.png" width="400">
</p>

* Left figure: 80 most common unigrams and bigrams in positive reviews.
* Right figure: 80 most common unigrams and bigrams in negative reviews.

### 5.2 TF-IDF vectorizer<a name="tfidf"></a>
```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
### 5.3 Dimensionality reduction: SVD/LSA<a name="lsa"></a>
```python
from sklearn.decomposition import TruncatedSVD
```
Truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in `sklearn.feature_extraction.text`. In that context, it is known as latent semantic analysis (LSA). **[Reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)**

## 6 Machine learning models<a name="ml"></a>

### 6.1 Naive Bayes<a name="bayes"></a>

### 6.2 Logistic Regression<a name="lr"></a>

### 6.3 Random Forest<a name="rf"></a>

## 7 Conclusion<a name="conclusion"></a>

