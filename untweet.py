import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
import timeit

# ML Sentiment Analysis in spanish packages
#    pip install sentiment-analysis-spanish
#    pip install keras tensorflow
from sentiment_analysis_spanish import sentiment_analysis
import seaborn as sns
import matplotlib.pyplot as plt
  
# Función para preprocesar el contenido de los tuits y quitar palabras no relevantes para el analisis
def preprocess_tweet(tweet):
    tweet_words = []
    
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)
    tweet_proc = ' '.join(tweet_words)
    return (tweet_proc)
            


starting_time = timeit.default_timer()

os.system('clear')    # Arranquemos con una pantalla fresca y limpia
# Query setup
query = "Nicolás Maduro"
exact_query = True
since_date = ' since:' + '2023-01-01'
until_date = ' until:' + '2023-02-28'
hashtags = ' '          # ' #Venezuela' 
min_replies = ''        # ' min_replies:' + '10'
min_faves = ''          # ' min_replies:' + '20'
min_retweets = ''       # ' min_retweets:'+ '30'
lang = ' lang:es'

tweets = []
limits = 100

# Query preparation
if exact_query:
    query = '"' + query + '"'
query = query + min_replies + min_faves + min_retweets + lang + since_date + until_date

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    # Vars del objeto tweet
    # print ('Vars -------------------')
    # print(vars(tweet))
    if len(tweets) == limits:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.rawContent, tweet.viewCount, tweet.replyCount, tweet.likeCount, tweet.retweetCount, 0.0])

#
if len(tweets) == 0:
    print('No se produjeron resultados para esta búsqueda')
    quit()

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', 'Views', 'Replies', 'Likes', 'Retweets', 'Score']).drop_duplicates(subset=['User', 'Tweet'])
# print(df[['Date', 'User', 'Tweet']].to_string())
# df['Time'] = df['Date'].apply(lambda x: x[11:])
df['Date'] = df['Date'].apply(lambda x: str(x))
print(df)



sentiment = sentiment_analysis.SentimentAnalysisSpanish()

acum_feeling = 0
min_feeling  = 1
max_feeling  = 0
my_tweets = df.Tweet
min_i = -1
max_i = -1

print(my_tweets)
for i, tweet in enumerate(my_tweets):
    preproc_tweet = preprocess_tweet(tweet)
    feeling = sentiment.sentiment(preproc_tweet)
    acum_feeling += feeling
    if feeling < min_feeling:
        min_feeling = feeling
        min_i = i
    if feeling > max_feeling:
        max_feeling = feeling
        max_i = i
    df.Score[i] = feeling

print('Guardando a archivo Excel...')
df.to_excel('Tweet Results.xlsx', sheet_name='Results')
print('Listo!')
    
print('\n\nResultados del sentiment analysis --------------')
print('    Promedio: ', feeling/i)
print('    Máximo:   ', max_feeling)
print('    Mínimo:   ', min_feeling)
if max_i >= 0:
    print(f'\nTweet más positivo[{max_feeling}]: ')
    print(my_tweets[max_i])
if min_i >= 0:
    print(f'\nTweet más negativo[{min_feeling}]: ')
    print(my_tweets[min_i])
print(f'\n\nTiempo transcurrido para el análisis de {limits} tweets (segundos): ', timeit.default_timer()-starting_time)


# creating a histogram
plt.hist(df['Score'], 10)
plt.title('Sentiment Analysis')
plt.xlabel('Positividad')
plt.ylabel('Frecuencia')
plt.show()