# Forensic Linguistics Applied To Tweets By Bots and Trolls
Project by Varun Tandon as a part of Stanford's CS109 Final Project. 

**Warning: Since much of this uncensored data was obtained from troll/bot accounts as well as the general Twittersphere, there may be profane, racist, sexist, or other inflammatory content shown as output.** The output of snippets of code and the content of the data processed are not indicative of my personal views. All forms of bigotry should be condemned. 

#### Imports/Setup


```python
import pandas as pd
import numpy as np
import re
```

## Some Helpful Functions

I began by writing up some helpful functions for processing and cleaning the data that I had gathered. Since I'm running an analysis on tweets, which can contain anything from Chinese characters to emoji, I needed a function to clean up a tweet by converting it to alphabetic letters. 

I also wrote a function for generating word counts and frequencies, with the hope that I could use the differences in word frequencies between the two sets to classify unknown tweets (similar to the Federalist Papers). 


```python
def clean_tweet(col):
    col = col.apply(lambda x: str(x).lower())
    alpha_only = re.compile('[^a-zA-Z\s]')
    col = col.apply(lambda x: alpha_only.sub('', x))
    
    col = col.apply(lambda x: str(x).split())
    return col

def generate_word_count(col):
    word_count = dict()
    for i in range(col.size):
        for word in col[i]:
            if (word in word_count.keys()):
                word_count[word] += 1
            else:
                word_count[word] = 1
    return word_count

def generate_word_freq(count, total):
    return (count / total)
```

## Clean Data (Troll)

The data used here was acquired from a Kaggle dataset of tweets that are known to be posted by Russian bots/trolls. The dataset can be found here: https://www.kaggle.com/vikasg/russian-troll-tweets


```python
# Read CSV
troll_df = pd.read_csv('tweets.csv')

# Clean the text
troll_df['text'] = clean_tweet(troll_df['text'])
```

## Clean Data (Rest of Twitter)

Ideally I would have found some Twitter data from the same timeframe as the Russian data (as tweets will often skew according to current events); however, I could not find any unbiased, random datasources containing data from the same time frame. Most of the datasets on Kaggle tend to have some focus (ie. tweets from Russian trolls, unhappy tweets, etc.), so I had to generate my own random sample. 

To do so, I used the twarc command line tool. 

Specifically, I sampled on Wednesday, May 29th, 2018, with the following command:

twarc sample > tweets.jsonl

Unfortunately, there's no way for me to select just English tweets using this, so of the 70,747 tweets extracted, only 22,595 are in English. Still, this is a sizeable number of tweets, and hopefully this provides a good representation of tweet word frequencies. 


```python
# Read in the JSON of random tweets
twitter_df = pd.read_json('tweets.jsonl', lines=True)

# Isolate English tweets
twitter_df = twitter_df[twitter_df.lang == 'en'].reset_index()

# Clean the tweets
twitter_df['text'] = clean_tweet(twitter_df['text'])
```

## Establishing Evaluation Sets

In order to evaluate the accuracy of my system of classifying tweets as troll or normal, I need to have some unbiased tweets to classify whose answers I know. To generate this set, I will randomly remove 2500 tweets from both the normal and troll datasets


```python
# Generate random indices to isolate
norm_to_remove = np.random.choice(twitter_df.shape[0], 2500)
troll_to_remove = np.random.choice(troll_df.shape[0], 2500)

norm_test = pd.DataFrame(twitter_df['text'].iloc[norm_to_remove])
norm_test['is_troll'] = False
troll_test = pd.DataFrame(troll_df['text'].iloc[troll_to_remove])
troll_test['is_troll'] = True

test_data = norm_test.append(troll_test).reset_index()

twitter_df = twitter_df.drop(norm_to_remove).reset_index()
troll_df = troll_df.drop(troll_to_remove).reset_index()
```

## Generate Word Frequencies For Both Datasets


```python
# Generate word counts
word_count = generate_word_count(troll_df['text'])

# Generate the word frequencies
troll_wf = pd.DataFrame.from_dict(word_count, orient='index').sort_values(by=[0], ascending=False)
total_count = sum(troll_wf[0])
troll_wf['word_freq'] = troll_wf.apply(lambda x: generate_word_freq(x, total_count))

troll_wf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>word_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rt</th>
      <td>147967</td>
      <td>5.272635e-02</td>
    </tr>
    <tr>
      <th>the</th>
      <td>69279</td>
      <td>2.468678e-02</td>
    </tr>
    <tr>
      <th>to</th>
      <td>55427</td>
      <td>1.975078e-02</td>
    </tr>
    <tr>
      <th>a</th>
      <td>38941</td>
      <td>1.387618e-02</td>
    </tr>
    <tr>
      <th>of</th>
      <td>33856</td>
      <td>1.206420e-02</td>
    </tr>
    <tr>
      <th>in</th>
      <td>32325</td>
      <td>1.151864e-02</td>
    </tr>
    <tr>
      <th>is</th>
      <td>29379</td>
      <td>1.046887e-02</td>
    </tr>
    <tr>
      <th>trump</th>
      <td>27359</td>
      <td>9.749066e-03</td>
    </tr>
    <tr>
      <th>and</th>
      <td>26415</td>
      <td>9.412683e-03</td>
    </tr>
    <tr>
      <th>for</th>
      <td>24815</td>
      <td>8.842541e-03</td>
    </tr>
    <tr>
      <th>you</th>
      <td>22607</td>
      <td>8.055746e-03</td>
    </tr>
    <tr>
      <th>i</th>
      <td>20279</td>
      <td>7.226189e-03</td>
    </tr>
    <tr>
      <th>on</th>
      <td>18281</td>
      <td>6.514225e-03</td>
    </tr>
    <tr>
      <th>this</th>
      <td>13796</td>
      <td>4.916047e-03</td>
    </tr>
    <tr>
      <th>that</th>
      <td>13247</td>
      <td>4.720417e-03</td>
    </tr>
    <tr>
      <th>it</th>
      <td>13148</td>
      <td>4.685139e-03</td>
    </tr>
    <tr>
      <th>with</th>
      <td>11964</td>
      <td>4.263234e-03</td>
    </tr>
    <tr>
      <th>are</th>
      <td>11538</td>
      <td>4.111434e-03</td>
    </tr>
    <tr>
      <th>be</th>
      <td>11286</td>
      <td>4.021637e-03</td>
    </tr>
    <tr>
      <th>clinton</th>
      <td>10870</td>
      <td>3.873400e-03</td>
    </tr>
    <tr>
      <th>hillary</th>
      <td>10470</td>
      <td>3.730865e-03</td>
    </tr>
    <tr>
      <th>not</th>
      <td>10142</td>
      <td>3.613986e-03</td>
    </tr>
    <tr>
      <th>we</th>
      <td>9108</td>
      <td>3.245532e-03</td>
    </tr>
    <tr>
      <th>amp</th>
      <td>9073</td>
      <td>3.233060e-03</td>
    </tr>
    <tr>
      <th>at</th>
      <td>9052</td>
      <td>3.225577e-03</td>
    </tr>
    <tr>
      <th>my</th>
      <td>8906</td>
      <td>3.173551e-03</td>
    </tr>
    <tr>
      <th>your</th>
      <td>8635</td>
      <td>3.076983e-03</td>
    </tr>
    <tr>
      <th>have</th>
      <td>8330</td>
      <td>2.968300e-03</td>
    </tr>
    <tr>
      <th>will</th>
      <td>8319</td>
      <td>2.964380e-03</td>
    </tr>
    <tr>
      <th>obama</th>
      <td>8031</td>
      <td>2.861755e-03</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>mrsrep</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>wonderfulwomank</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcooszfzksnv</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcozwpaigfw</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcohsgpmtrms</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>facthillary</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcovznhvcsliu</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>dwpolitics</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcowyelrmo</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcoculpqbpl</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>umtata</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>verflixt</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstconurrgwyyy</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>bemht</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>autoren</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>lesenswert</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>tribalnationtni</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcojuipdbpup</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>apadeo</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcosxwgtqxne</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>heisyourpres</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>dramatics</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcoorjxjb</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcogpvghbfh</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcoxfuggvpoj</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>francisski</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>sosokoba</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>releasethetranscripts</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>httpstcozoqzkqtyt</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
    <tr>
      <th>qual</th>
      <td>1</td>
      <td>3.563386e-07</td>
    </tr>
  </tbody>
</table>
<p>234118 rows × 2 columns</p>
</div>

We notice "rt" ranking as the most frequent word here, and in this case "rt" indicates a "retweet" by a Twitter user. At this point, I'm not sure whether or not to leave it in. On the one hand, perhaps a bot is more/less likely to retweet. On the other hand, this might just ruin the accuracy of predictions because it doesn't have any bearing on the content of the tweet. 

For now, I'm going to leave it in, and then later on I'm going to see how it affects the prediction accuracy to remove it. 

We also notice that "trump" and "clinton" rank fairly highly on the word frequencies list. This seems to be a good sign, because assuming that tweets are similar to normal human language, these should not be very frequent terms in normal tweets. We'll verify that hypothesis when generating word frequencies on a random sample of Twitter data. 


```python
# Generate word counts
twitter_word_count = generate_word_count(twitter_df['text'])

# Generate word frequencies
twit_wf = pd.DataFrame.from_dict(twitter_word_count, orient='index').sort_values(by=[0], ascending=False)
total_count = sum(twit_wf[0])
twit_wf['word_freq'] = twit_wf.apply(lambda x: generate_word_freq(x, total_count))

twit_wf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>word_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rt</th>
      <td>12503</td>
      <td>0.043709</td>
    </tr>
    <tr>
      <th>the</th>
      <td>7648</td>
      <td>0.026736</td>
    </tr>
    <tr>
      <th>to</th>
      <td>5726</td>
      <td>0.020017</td>
    </tr>
    <tr>
      <th>i</th>
      <td>5124</td>
      <td>0.017913</td>
    </tr>
    <tr>
      <th>a</th>
      <td>4972</td>
      <td>0.017381</td>
    </tr>
    <tr>
      <th>you</th>
      <td>3866</td>
      <td>0.013515</td>
    </tr>
    <tr>
      <th>and</th>
      <td>3772</td>
      <td>0.013186</td>
    </tr>
    <tr>
      <th>is</th>
      <td>3280</td>
      <td>0.011466</td>
    </tr>
    <tr>
      <th>of</th>
      <td>3175</td>
      <td>0.011099</td>
    </tr>
    <tr>
      <th>in</th>
      <td>2898</td>
      <td>0.010131</td>
    </tr>
    <tr>
      <th>this</th>
      <td>2724</td>
      <td>0.009523</td>
    </tr>
    <tr>
      <th>for</th>
      <td>2509</td>
      <td>0.008771</td>
    </tr>
    <tr>
      <th>my</th>
      <td>2364</td>
      <td>0.008264</td>
    </tr>
    <tr>
      <th>me</th>
      <td>2141</td>
      <td>0.007485</td>
    </tr>
    <tr>
      <th>that</th>
      <td>2031</td>
      <td>0.007100</td>
    </tr>
    <tr>
      <th>it</th>
      <td>1950</td>
      <td>0.006817</td>
    </tr>
    <tr>
      <th>on</th>
      <td>1911</td>
      <td>0.006681</td>
    </tr>
    <tr>
      <th>be</th>
      <td>1438</td>
      <td>0.005027</td>
    </tr>
    <tr>
      <th>with</th>
      <td>1402</td>
      <td>0.004901</td>
    </tr>
    <tr>
      <th>im</th>
      <td>1340</td>
      <td>0.004684</td>
    </tr>
    <tr>
      <th>so</th>
      <td>1305</td>
      <td>0.004562</td>
    </tr>
    <tr>
      <th>your</th>
      <td>1199</td>
      <td>0.004192</td>
    </tr>
    <tr>
      <th>not</th>
      <td>1187</td>
      <td>0.004150</td>
    </tr>
    <tr>
      <th>if</th>
      <td>1156</td>
      <td>0.004041</td>
    </tr>
    <tr>
      <th>are</th>
      <td>1153</td>
      <td>0.004031</td>
    </tr>
    <tr>
      <th>like</th>
      <td>1151</td>
      <td>0.004024</td>
    </tr>
    <tr>
      <th>have</th>
      <td>1097</td>
      <td>0.003835</td>
    </tr>
    <tr>
      <th>at</th>
      <td>1081</td>
      <td>0.003779</td>
    </tr>
    <tr>
      <th>just</th>
      <td>1054</td>
      <td>0.003685</td>
    </tr>
    <tr>
      <th>but</th>
      <td>1040</td>
      <td>0.003636</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>kittytranny</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>stephan</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>euvaille</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>assignments</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>penmanship</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>killajunsters</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>ageedior</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>httpstcodmzjgev</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>vinnybrack</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>michaelme</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>calicurmudgeon</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>billyever</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>einsteinmaga</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>americanalina</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>httpstcopzxxwknf</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>httpstconykbifxhf</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>naughtyk</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>adrianfknpetrov</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>stic</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>deaths</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>tamnna</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>aquariusunite</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>irukkura</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>httpstcojsgscndh</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>httpstcosaparugyf</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>chua</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>welson</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>barnabychuck</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>elmuss</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>httpstcomnmlxphv</th>
      <td>1</td>
      <td>0.000003</td>
    </tr>
  </tbody>
</table>
<p>43453 rows × 2 columns</p>
</div>



## Classification

Consider the words in a given tweet as <img src="https://latex.codecogs.com/gif.latex?T"/>. Let us also denote the troll/bot writer as <img src="https://latex.codecogs.com/gif.latex?B"/> and the normal twitter user as <img src="https://latex.codecogs.com/gif.latex?N"/>. 

We want to find

<img src="https://latex.codecogs.com/gif.latex?\frac{P(N|T)}{P(B|T)}"/>  
<img src="https://latex.codecogs.com/gif.latex?\frac{P(T|N)P(N)}{P(T|B)P(B)}"/>

Googling around for the percentage of tweets posted by bots indicates some alarming statistics (see: https://www.pewresearch.org/fact-tank/2018/04/09/5-things-to-know-about-bots-on-twitter/), but none of these statistics give a valid prior for the probability of a tweet being posted by a bot. To represent this ambiguity, we will say that

<img src="https://latex.codecogs.com/gif.latex?P(N)%20=%20P(B)%20=%200.5"/>

We observe that this cancels out in our equation above, so we are left with

<img src="https://latex.codecogs.com/gif.latex?\frac{P(T|N)}{P(T|B)}" />

As we did in class, we can rewrite these using multinomials, and the multinomial terms in the numerator and denominator cancel, yielding

<img src="https://latex.codecogs.com/gif.latex?\frac{\prod_i%20p_{i}^{c_i}}{\prod_i%20q_{i}^{c_i}}" />

We can use logarithms to make this computationally stable, and write

<img src="https://latex.codecogs.com/gif.latex?\log(\frac{P(T|N)}{P(T|B)})%20=%20\log(\frac{\prod_i%20p_{i}^{c_i}}{\prod_i%20q_{i}^{c_i}})%20=%20\sum_i%20c_i\log(p_i)%20-%20\sum_i^{c_i}%20\log(q_i)" />

(To reiterate, this process is identical to the process done in lecture with the Federalist Papers, so some steps in the math were ommited)

Now to convert this to code!

The one issue that we run into is the case where a Twitter user writes a unique word, a word that has not been used by a bot or a normal human in our datasets. In this case, I simply assume a word frequency equal to 0.000004, which is the frequency for words that are observed once in the normal tweets word frequency table. 


```python
def calculate_LL(word_list, is_bot):
    unique_word_error = 0.000004
    sum = 0
    freq_list = None
    if is_bot:
        freq_list = troll_wf['word_freq']
    else:
        freq_list = twit_wf['word_freq']
    for word in word_list:
        if (word in freq_list):
            sum += np.log(freq_list[word])
        else:
            sum += np.log(unique_word_error)
    return sum

def is_troll(likelihood_norm):
    if (likelihood_norm > 1):
        return False
    else:
        return True
    
def print_results(data):
    class0_count = data[data['is_troll'] == False].shape[0]
    class1_count = data[data['is_troll'] == True].shape[0]

    class0_correct = data[(data['pred_correct'] == True) & (data['is_troll'] == 0)].shape[0]
    class1_correct = data[(data['pred_correct'] == True) & (data['is_troll'] == 1)].shape[0]

    print("Normal Tweets: tested {}, correctly classified {}.".format(class0_count, class0_correct))
    print("Troll Tweets: tested {}, correctly classified {}.".format(class1_count, class1_correct))
    print("Overall: tested {}, correctly classified {}.".format(class0_count + class1_count, class0_correct + class1_correct))
    print("Accuracy = {}.".format((class0_correct + class1_correct) / (class0_count + class1_count)))

test_data['norm_LL'] = test_data['text'].apply(lambda x: calculate_LL(x, False))
test_data['bot_LL'] = test_data['text'].apply(lambda x: calculate_LL(x, True))
test_data['norm_LL - bot_LL'] = test_data['norm_LL'] - test_data['bot_LL']
test_data['e^(prob)'] = np.exp(test_data['norm_LL - bot_LL'])
test_data['pred_is_troll'] = test_data['e^(prob)'].apply(is_troll)
test_data['pred_correct'] = test_data.apply(lambda x: x['is_troll'] == x['pred_is_troll'], axis=1)

print_results(test_data)
```

    Normal Tweets: tested 2500, correctly classified 2282.
    Troll Tweets: tested 2500, correctly classified 1818.
    Overall: tested 5000, correctly classified 4100.
    Accuracy = 0.82.


## Looking To The Future

While an accuracy of 0.82 is pretty good, I'm sure there are a lot of adjustments that can be made to improve the accuracy of the predictions. 

### 1. Word Similarity
Right now the comparisons are being run using distinct words, rather than clustering similar words. For example, we notice that at the tail end of the troll tweets there are a lot of URLs. A study by Pew (linked above) found that 66% of all links posted on Twitter were from trolls, so clustering URLs could be provide a boost to accuracy. 

Along the same lines, hashtags tend to join words together, so clustering hashtags with their composite words could improve the model's understanding of what is being talked about. 

### 2. Retweets
As I alluded to earlier, it could be interesting to try removing and adding retweets to see if this is a practice that is more common among bots than humans. 

### 3. Other Algorithms
Since I started this project, CS109 has explored logistic regression and neural networks, whereas the approach taken in this project was more along the lines of Naive Bayes. Perhaps trying logistic regression and neural nets would have higher accuracy in predicting whether a user is a bot. 

### 4. More Data
I was somewhat limited by the processing power of my laptop and the API rate limits imposed by Twitter. Since the publication of the Kaggle dataset I used in this project, there has been the publication of a new dataset with over 3 million troll tweets (https://www.kaggle.com/fivethirtyeight/russian-troll-tweets/version/1). 

Similarly, the "control" dataset can definitely be expanded, and increasing this dataset size would likely give a more accurate picture of the norm of word frequencies on Twitter. 

### 5. A Big Picture Look
This project mainly focused on word frequencies as a mechanism for determining whether a Twitter user was a troll or not, and 8.7% of the tweets in the test set were labelled incorrectly as troll. Analyzing these false positive could give a good picture of what this model is really predicting. Are normal users who tweet about politics being labelled as bots? Is the model simply classifying content as toxic/bigoted? Looking at what's going on with some individual cases could indicate more accurately what content is being labelled as troll. 
