import snscrape.modules.twitter as sntwitter
import pandas as pd
import os

os.system('clear')    # Arranquemos con una pantalla fresca y limpia
# Query setup
query = "elon musk"
exact_query = False
since_date = ' since:' + '2020-01-01'
until_date = ' until:' + '2023-02-28'
hashtags = ' ' + '#USA'
min_replies = ' min_replies:' + '10'
min_faves = ' min_replies:'   + '20'
min_retweets = ' min_retweets:'   + '30'
lang = ' lang:es'

tweets = []
limits = 10

# Query preparation
if exact_query:
    query = '"' + query + '"'
query = query + min_replies + min_faves + min_retweets + lang + since_date + until_date

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    # Vars del objeto tweet
    print ('Vars -------------------')
    print(vars(tweet))
    print ('Tweet -------------------')
    print(tweet)    # break
    
    if len(tweets) == limits:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.rawContent[:30], tweet.viewCount, tweet.replyCount, tweet.likeCount, tweet.retweetCount])

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', 'Views', 'Replies', 'Likes', 'Retweets']).drop_duplicates(subset=['User', 'Tweet'])
print(df.to_string())