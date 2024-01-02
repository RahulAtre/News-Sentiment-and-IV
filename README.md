# Effect of News Sentiment on Implied Volatility
This project examines the relationship between news sentiment and implied volatility and determines whether there is a significant correlation between the two

## Table of Contents

- [Introduction](#introduction)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Sentiment Analysis and Implied Volatility](#sentiment-analysis-and-implied-volatility)
- [Results](#results)
  - [Inferences from COVID-19](#inferences-from-covid-19)
  - [Comparison of Various Regression Models](#comparison-of-various-regression-models)
- [Conclusion](#conclusion)
  - [Ethical Concerns](#ethical-concerns)
- [References](#references)

## Introduction

Quantitative analysts are professional who use mathematical statistical methods to help companies make business and financial decisions, as well as identifying patterns, trends, and opportunities in investment. They can analyze markets, assess risk, and develop models for pricing and trading. One such area of interest can be understanding the impact of news on market behavior. We attempt to make statistical inferences based on financial data similar to a quantitative analyst. This project serves as a stepping stone into the world of finance, specifically implied volatility.

We will be looking at a few of the crucial barometers of market impact: implied volatility (IV), sentiment, and polarity. The question we would like to answer is the following: Is there a relationship (significant or not) between news sentiment and implied volatility? Before we define implied volatility, let us compare it to volatility as they represent different aspects of market behavior. 

**Volatility** is a measure of the standard deviation of an asset's price, reflecting the historical price fluctuations. It is measured by the standard deviance of the data and provides information about past risk and uncertainty in the market. According to Investopedia.com, **Implied volatility** is the expected future volatility of an asset that is “implied” by the prices of its options contracts. It is a measure of the expected future volatility of an asset. 

Both metrics address the price fluctuation of an asset; however, implied volatility reflects the forward-looking outlook of market participants. Thus, IV can be influenced by external factors beyond the asset history, including investor sentiments and economic news. In simpler terms, volatility can be thought of as the market’s “fear gauge”, reflecting the degree of price fluctuations that investors anticipate for any particular asset, whereas implied volatility paints a forward-looking picture, reflecting the market’s collective beliefs and anxieties. For instance, it can be influenced by factors beyond pure data, like investor sentiment and economic news. 

Sentiment, or news sentiment, captures the emotional tenor of news articles. Our sentiment analysis examines the following four metrics:

**Content subjectivity**: Measures the degree to which a news article expresses personal opinions, beliefs, or judgement. Ex. for a news article debating the feasibility of an economic policy, the subjectivity score would be high, reflecting the presence of individual viewpoints. 

**Content polarity**: Captures the overall emotional slant of the news article. Ex. a positive polarity indicates a cheerful and optimistic tone, while negative polarity suggests a critical or pessimistic one. 

**Headline subjectivity**: Zooms in on the specific wording of the headline, evaluating its potential for bias or personal interpretation. For instance, a headline stating “The economy of Canada is in ruins!” is likely high in subjectivity, hinting at an emotional angle rather than objective reporting. 

**Headline polarity**: Focuses on the emotional direction conveyed by the headline. Ex. A headline like “Investors Cheer Record Profits!” is positive polarity, whereas something like “Big Tech Faces Backlash Over Privacy Concerns” is negative. 

We will be exploring the relationship between IV and these four news sentiment metrics, and understanding the emotional tone and bias of news content from a statistical lens. 

## Data Cleaning and Preparation

In this section, we talk about the methods used to analyze the data and to determine whether there is a significant relationship between news sentiment and implied volatility of news sources and companies.

We used several data sets of news articles collected and posted on Kaggle. These datasets included articles containing a wide range of categories such as lifestyle, business, tech, politics, etc. The news outlets from which we collected were New York Times, Washington Post, Wall Street Journal, and several others. The category labels were an essential part of our exploration, as we were looking to find articles that pertain to specific sectors that we will be conducting our analysis on. 

Most of our news article data were not clearly categorized or labeled. We needed to use our labeled dataset to predict labels on the other datasets to look for relevant news articles. We achieved this by using a combination of simple machine learning models to predict the categories of each article, and some basic pattern matching to find specific keywords. The first step was to prepare the data for our machine learning models. We removed all punctuation, HTML tags, Markdown tags, and stopwords. The list of stopwords we used came from the “spaCy” library’s default set of stopwords. 

Then, we tokenized the strings using the same package, and created new dataframe columns for these clean text headlines and articles, as well as their tokens. Next, we used the “scikit-learn” package to vectorize our data and extract features. We ran several machine learning models to test the classification performance. We tried using logistic regression, naive bayes, decision tree classifiers, and random forest classifiers. With a 75/25 train-test split, the highest accuracy was obtained from the logistic regression model, achieving a test accuracy of 90%. 

![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure1.png?raw=true) 
<p align="left">
    Figure 1 - Confusion Matrix for Predicted Labels vs. True Label for Topics
</p>

The above confusion matrix compares the predicted labels for the topic we would like to classify into (Ex. world news) and the true labels. We can see visually that most of the true and predicted are the same for world news in the top left corner. However, we did notice issues with multiple categories being assigned 0 for its topic, and also many of the world news articles were assigned a topic number “3” as well. Most of the world news articles were assigned a topic number of 0 however, so for our purposes we are only interested in collecting those that receive a 0, so we are not classifying or mapping the values 1-3.

At this point, we ran into issues with classification. Since many of the articles have overlapping categories, it is hard to separate articles from each, such as “US news” versus “tech news”. In the end, we decided to only look for 1 category in specific, “world news”, since our objective was also to see how geopolitical and foreign news sentiment affects the implied volatility on commodity indexes, namely the S&P Commodity Indexed Trust (GSCI). Applying our model to the unlabeled data, we were able to extract about 15000 news articles that we will be using when examining the implied volatility of GSCI.

Our next step was to obtain historical implied volatility data, originally we used Interactive Brokers’ API to get our data, however in the end we used Market Chameleon since it provided us with more preprocessed data, aside from the historical implied volatility data we received from Interactive Brokers. We used implied volatility data from 3 tickers; SPY, which is an exchange traded fund that tracks the S&P 500, a collection of the 500 largest companies in the United States, QQQ, an exchange traded fund that tracks the NASDAQ 100 which is a collection of the 100 largest non-finance companies and is comprised of mostly technology companies, and finally GSG, which tracks the S&P Commodity Indexed Trust GSCI.

## Sentiment Analysis and Implied Volatility

The next stage of our project involved performing sentiment analysis (as described in the introduction) and comparing it to the IV to test for any relationship. Fortunately, we found a python package called TextBlob that uses natural language processing techniques (tokenization, lexical lookup, sentiment scoring). Since it is outside the scope of our project, we will assume it to be a mathematical black-box that outputs sentiment and polarity for any given news article, and its respective headline. 

As mentioned in the introduction, options contracts reflect market expectations about future pricing movements. IV is derived from these contracts, providing a rough estimate of the expected volatility over a specific time frame. Hence, we collected options data from three stock tickers: SPY, QQQ, and GSG, which are index funds that reflect a large segment of the market. These data files contain the implied volatility and the expiration date of their respective options contracts. 

We combined specific news data (tech news with QQQ, US and business news with SPY, commodities and world news with GSG). Then, the modified dataset was resampled from and we computed standard deviations of polarity and subjectivity and stored them in a new dataframe. The standard deviation is used to find the spread of sentiment. Then, we merged the standard deviations of polarity, subjectivity and the IV data for all three. After fitting a linear regression model of IV against 4 predictor variables (headline polarity, headline subjectivity, content polarity, and content subjectivity), and performed tests on correlations on them.

## Results

We now discuss the results obtained from analyzing the data. Below we have visualizations of the relationship between sentiment and implied volatility. We made scatterplots of implied volatility against each of the four metrics of sentiment, which are the standard deviations of headline polarity, headline subjectivity, content polarity, and content subjectivity. The relationship pairing (between IV and sentiment) that we have chosen as a subset is as follows: IV of SPY and sentiment of US and business news, IV of QQQ and sentiment of technology news, IV of GSG and sentiment of world and commodities news. We performed two different visualization techniques: one using scatterplots and the other using correlation heatmaps.

We first look at the relationship between IV of SPY and the sentiment of US and business news combined. Figure 2 shows the scatterplots of IV plotted against the four sentiment metrics, respectively.

![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure2.png?raw=true) 
<p align="left">
    Figure 2 - Comparing IV SPY Index to Sentiment of US and Business News
</p>

From Figure 2, looking at all four charts plotting the relationship between implied volatility and sentiment, we can see that there appears to be a very weak relationship between the IV of SPY and all of the sentiment metrics of the US and business news combined. There does not appear to be a significant increasing or decreasing pattern among all 4 plots. This hints at a very weak correlation between IV and changes in subjectivity and polarity. 

Now, we look at the relationship between the implied volatility of QQQ and sentiment of technology news. Figure 3 shows the scatterplots of implied volatility plotted against the four sentiment metrics.

![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure3.png?raw=true) 
<p align="left">
    Figure 3 - Comparing IV QQQ Index to Sentiment of Tech News
</p>

From Figure 3, there appears to be a slight downward trend between IV and headline polarity, and between IV and content polarity, which implies a weak negative relationship. The other two scatterplots do not show any signs of a strong relationship. So overall, there is a weak link between IV of QQQ and sentiment of tech news. 

Finally, we look at the relationships between the IV of GSG and the sentiment of world and commodities news. Figure 4 shows the scatterplots of the implied volatility plotted against the four sentiment metrics.

![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure4.png?raw=true) 
<p align="left">
    Figure 4 - Comparing IV GSG Index to Sentiment of World and Commodities News
</p>

From Figure 4, we see that there appears to be a weak to moderate negative relationship between IV and content polarity,  IV and content subjectivity. The other two scatterplots do not show any signs of a strong relationship. So overall, there is a weak to moderate relationship between the IV of GSG and sentiment metrics of world and commodities news.

Now, we analyze the correlation heatmaps of IV (from each of the three indexes) and the sentiment of its corresponding news sources. By doing so, we can have a preliminary understanding of how much correlation there is between IV, polarity and subjectivity in general, as well as the correlation between subjectivity and polarity for both news headlines and content.

<ins>IV vs. Sentiment (SPY)<ins>
![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure5.png?raw=true) 
<p align="left">
    Figure 5 - Correlation Matrix of IV and Sentiment Metrics (SPY)
</p>

<ins>IV vs. Sentiment (QQQ)<ins>
![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure6.png?raw=true) 
<p align="left">
    Figure 6 - Correlation Matrix of IV and Sentiment Metrics (QQQ)
</p>

<ins>IV vs. Sentiment (GSG)<ins>
![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure7.png?raw=true) 
<p align="left">
    Figure 7 - Correlation Matrix of IV and Sentiment Metrics (GSG)
</p>

From looking at the correlation heatmaps of all three combinations, we find that there is an overall very weak to negligible correlation between IV and the four sentiment metrics. However, there is a moderate to strong relationship between headline polarity and headline subjectivity, and between content polarity and content subjectivity. 

![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure8.png?raw=true) 
<p align="left">
    Figure 8 - Boxplots for Content Polarity and Subjectivity (Combined)
</p>

Plotting the combined boxplot (Using all news sources) for content subjectivity and content polarity, we also note a few interesting observations. First, that the overall subjectivity has more spread than polarity, and secondly, there are less outliers for subjectivity in comparison to the polarity. 

### Inferences from COVID-19

<ins>IV vs. Sentiment (SVP) [COVID]<ins>
![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure9.png?raw=true) 
<p align="left">
    Figure 9 - Comparing IV SPY Index to Sentiment of US and Business News (COVID)
</p>

<ins>IV vs. Sentiment (GSG) [COVID]<ins>
![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Figure10.png?raw=true) 
<p align="left">
    Figure 10 - Comparing IV GSG Index to Sentiment of World and Commodities News (COVID)
</p>

For the COVID analysis, we can see that for IV of SPY and QQQ, the linear regression models lead to a worse fit. The linear regression model for GSG IV however, leads to a better fit. We are unsure of the exact cause of why the correlation factor for the various indexes increased and decreased. However, it does show that when there is uncertainty and fear in the market, it leads to fluctuations in the relationship between IV and sentiment. 

### Comparison of Various Regression Models

In addition to the linear relationships that were explored for IV and sentiment, we tried using more complex regression techniques to better fit the data. Here are the results we obtained for the three indexes: 

<p align="left">
    Table 1: Comparison of Different Regression Models for SPY 
</p>
![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Table1.png?raw=true) 

<p align="left">
    Table 2: Comparison of Different Regression Models for QQQ 
</p>
![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Table2.png?raw=true) 

<p align="left">
    Table 3: Comparison of Different Regression Models for GSG 
</p>
![alt text](https://github.com/RahulAtre/News-Sentiment-and-IV/blob/main/Project-Images/Table3.png?raw=true) 

We can see immediately that XGBoost performed drastically better than the other regression models. For all three IV indexes, XGBoost had a performance above 97%. It had a mean-squared error of roughly 2.09% on GSG, and nearly 0% on SPY and QQQ. The likely explanation for this after exploring the model's algorithm, is due to its exceptional ability of capturing complex nonlinear relationships. We are also sure that it is not overfitting, since XGBoost incorporates regularization techniques and generalizes well to unseen data. Traditional linear models like least-squares and ridge struggle to capture non-linearities, hence explaining their significantly lower R^2 values of around 11-12%.

## Conclusion 

Our analysis ventured into the complex domain of how news sentiment could influence implied volatility (IV), a key indicator of market anxiety. Although we applied various regression techniques, sentiment analysis, models, visualizations, the results paint a fairly nuanced picture. 

Overall, there are no clear signals of a relationship in the predictors and response. That is, the linear regression models in particular revealed insignificant relationships between IV and the four sentiment metrics, except for XGBoost. This could suggest that news sentiment, at least in the way we captured it, doesn’t seem to hold a strong sway over future market expectations reflected in IV.

Although the relationships might have been weak, there were some hints of potential connections. For instance, the GSG index, representing commodities, showed a moderately negative correlation between IV and both content subjectivity and polarity. One possible explanation for these results could be that uncertainty and negativity in world news might influence IV for these particular assets. Also, the COVID-19 analysis hinted at fluctuations in the relationship between IV and sentiment during periods of heightened market anxiety. 

Our exploration also extended beyond linear models, utilizing complex regression techniques such as ridge regression, polynomial regression, and XGBoost. In all instances, the XGBoost method achieved extremely high R-squared values. When looking into why this could be the case, it seems that the decisive verdict we arrived at was due to the model’s complexity, and we would need to further investigate its predictive capabilities in the event that it is overfitting and does not generalize well. Further analysis beyond the scope of this course could be to examine the performance of XGBoost and other advanced deep learning models to perhaps gain further insight, however these models are not as interpretable and the main goal of this project was to look for human insight. Ultimately, this could suggest that the relationship between sentiment and IV may be non-linear and requires further investigation using more sophisticated approaches. 

While our findings do not definitively establish a strong connection between news sentiment and IV, future research could delve deeper into non-linear relationships, explore sentiment from alternative sources like social media, and incorporate more nuanced sentiment analysis techniques. It is important to note however, that finding market insights can be a relatively challenging process, and while our analysis hasn't had a definitive answer, it adds a piece to the ongoing research, urging us to continue exploring the fascinating connection between emotions and markets.

### Ethical Concerns

In regards to concerns raised about our project, there are a few aspects to carefully consider. First, the classification model that we created to label the news articles had an accuracy of around 77%, which likely led to some inaccuracies in the end result and relationship fitting of IV and sentiment analysis. Also, since we collected data from Kaggle, we are dependent on the assumption of its validity, rather than creating our own data engineering pipeline. It is plausible that the news sources we collected contained misinformation and invalid entries. Lastly, there could also be some multicollinearity between subjectivity and polarity which makes the predictions less accurate and can make for less reliable models.

As for any ethical concerns about the nature of this project, there are several discussions and key points to think about. Firstly, if a strong link were established between news sentiment and market behaviour, it could potentially be exploited by actors with advanced sentiment analysis tools (i.e. manipulating news narratives or artificially inflating sentiment to influence market movements).  Also, sentiment analysis algorithms (such as the one in TextBlob package) are trained on large datasets of text and could contain biases that we as an end-user are unaware of. The “black-box” nature of many sentiment models can lead to a lack of transparency.

## References

Nickolas, S. (2022, May 20). Implied volatility. Investopedia. 
  https://www.investopedia.com/ask/answers/032515/what-options-implied-volatility-and-how-it-calculated.asp
  
News Category Dataset. (2022, September 24). Kaggle. 
  https://www.kaggle.com/datasets/rmisra/news-category-dataset/
  
All the news. (2017, August 20). Kaggle. 
  https://www.kaggle.com/datasets/snapcrack/all-the-news
  
News articles. (2017b, April 30). Kaggle. 
  https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles
