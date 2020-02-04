# Real or Not? NLP with Disaster Tweets

I decided to publish my solution of corresponding [competitions](https://www.kaggle.com/c/nlp-getting-started/overview) on Kaggle in this repository.
The main idea of most notebooks is to extract new features to build ensemble in future.

## TagsAsFeatures
As hashtags are popular in social networks, I decided to use them as features. During exploring I found that intersect of tags in train and in test sets isn't big. I tried to use stemming and lemmatization, but it didn't give anything. To build vector representation of each row I used CountVectorizer from sklearn. You can find more details in the [notebook](https://github.com/koren-v/SocialMediaClassification/blob/master/TagsAsFeatures.ipynb).

<p align="center">
  <img src="/images/intrsect.PNG">
</p>

## TagsWithGlove

When I get a pretty bad result in the previous notebook, I decided that the main reason for it was small intersect of tags. So I used pre-trained GloVe vectors from Twitter. As some of the tweets have more than one tag, I used 'Math with Words' to get one vector. Using this approach, I raised the metric by 8 points and I used the same trick for locations. [More details](https://github.com/koren-v/SocialMediaClassification/blob/master/TagsWithGlove.ipynb). 

## VaderForFeatures

To get more features from tweets I used Vader. [Vader](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. I expected to get a god correlation between new features and target, but it was not so. [More details](https://github.com/koren-v/SocialMediaClassification/blob/master/VaderForFeatures.ipynb).

<p align="center">
  <img src="/images/correlation.PNG">
</p>

## Bert

I just use pre-trained bert from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers). During some time of training it, I found good hyperparameters and found that better to use minimal data preprocessing. [More details](https://github.com/koren-v/SocialMediaClassification/blob/master/Bert.ipynb).

## To Do

Since the competition is not over yet, I plan to do:
- Build own word embeddings and use them for NN.
- Build ensemble.
