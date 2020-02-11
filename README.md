# Real or Not? NLP with Disaster Tweets

I decided to publish my solution of corresponding [competitions](https://www.kaggle.com/c/nlp-getting-started/overview) on Kaggle in this repository.
The main idea of most notebooks is to extract new features to build ensemble in future.

## TagsAsFeatures
As hashtags are popular in social networks, I decided to use them as features. During exploring I found that intersect of tags in train and in test sets isn't big. I tried to use stemming and lemmatization, but it didn't give anything. To build vector representation of each row I used CountVectorizer from sklearn. It was founded that the KNN can give as much more score than Random Forest on these features. You can find more details in the [notebook](https://github.com/koren-v/SocialMediaClassification/blob/master/Classic/TagsAsFeatures.ipynb).

<p align="center">
  <img src="/images/intrsect.PNG">
</p>

## TagsWithGlove

When I get a pretty bad result in the previous notebook, I decided that the main reason for it was small intersect of tags. So I used pre-trained GloVe vectors from Twitter. As some of the tweets have more than one tag, I used 'Math with Words' to get one vector. Using this approach, I raised the metric by 8 points and I used the same trick for locations. In this case, the KNN gave better results. [More details](https://github.com/koren-v/SocialMediaClassification/blob/master/Classic/TagsWithGlove.ipynb). 

## VaderForFeatures

To get more features from tweets I used Vader. [Vader](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. I expected to get a good correlation between new features and target, but it was not so. Only on this data, Random Forest was a little bit better than KNN. [More details](https://github.com/koren-v/SocialMediaClassification/blob/master/Classic/VaderForFeatures.ipynb).

<p align="center">
  <img src="/images/correlation.PNG">
</p>

## Bert

I just use pre-trained bert from [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers). During some time of training it, I found good hyperparameters and found that better to use minimal data preprocessing. When I picked predictions of all models I realized, that predictions for Bert were in the wrong order. I think that the reason of this is a bug and after spending 2 days to fix it I didn’t succeed. So I decided to use an average of Bert's prediction and predictions that I plan to get from stacking. [More details](https://github.com/koren-v/SocialMediaClassification/blob/master/Neural%20Nets/Bert.ipynb).


## LSTM

To build a good ensemble I decided to use not pretrained Net with pretrained GloVe vectors. At this time, I tried just one model that I gave not bad predictions for sarcasm detection, but I want to try some other architectures for this task. [The notebook](https://github.com/koren-v/SocialMediaClassification/blob/master/Neural%20Nets/PyTorchModelsForTwitter.ipynb) with this approach. The model's script [here](https://github.com/koren-v/SocialMediaClassification/blob/master/Neural%20Nets/nets.py).

## To Do

Since the competition is not over yet, I plan to do:
- Build ensemble.
- Use FastAi's pre-trained LSTM.
