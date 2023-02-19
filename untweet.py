import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
import timeit

# ML Sentiment Analysis in spanish packages
#    pip install sentiment-analysis-spanish
#    pip install keras tensorflow
from sentiment_analysis_spanish import sentiment_analysis

# ML Sentiment Analysis in spanish packages
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Librerías para grafica
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
            
# Hace el sentiment analysis NLTK para un texto en español
def nltk_analyze(text):
    # Tokenize the text into words
    words = word_tokenize(text, language='spanish')

    # Remove stopwords
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if not word in stop_words]

    # Instantiate a sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Calculate sentiment scores for the text
    scores = analyzer.polarity_scores(" ".join(words))

    # Print the sentiment scores
    return scores   


########################################
#   M A I N                            #
########################################
starting_time = timeit.default_timer()
os.system('clear')    # Arranquemos con una pantalla fresca y limpia

# Query setup
query = "Lionel Messi"
exact_query = True
since_date = ' since:' + '2022-12-18'
until_date = ' until:' + '2022-12-21'
hashtags = ' '          # ' #Venezuela'  
min_replies = ''        # ' min_replies:' + '10'
min_faves = ''          # ' min_replies:' + '20'
min_retweets = ''       # ' min_retweets:'+ '30'
lang = ' lang:es'
location = ''           # ' near:Caracas' + ' within:20mi'

tweets = []
limits = 100

# Query preparation
if exact_query:
    query = '"' + query + '"'
query = query + min_replies + min_faves + min_retweets + lang + since_date + until_date + location

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


# SENTIMENT ANALYSIS SPANISH  
sentiment = sentiment_analysis.SentimentAnalysisSpanish()

acum_feeling = 0
min_feeling  = 1
max_feeling  = 0
my_tweets = df.Tweet
min_i = -1
max_i = -1
acum_positive = 0
acum_neutral = 0
acum_negative = 0

print(my_tweets)
for i, tweet in enumerate(my_tweets):
    preproc_tweet = preprocess_tweet(tweet)
    feeling = sentiment.sentiment(preproc_tweet)
    
    # SA de NLTK
    nltk_feeling = nltk_analyze(preproc_tweet)
    positive_score = nltk_feeling['pos']
    neutral_score = nltk_feeling['neu']
    negative_score = nltk_feeling['neg']
    acum_positive += positive_score
    acum_neutral += neutral_score
    acum_negative  += negative_score
    print(f'>> Positive: {positive_score:.2f}      Neutral: {neutral_score:.2f}        Negative: {negative_score:.2f}')
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
    
print('\n\nResultados del Sentiment Analysis --------------')
print('    Promedio: ', feeling/i)
print('    Máximo:   ', max_feeling)
print('    Mínimo:   ', min_feeling)
print('    Promedio NLTK:     Pos: ', acum_positive/i,    'Neu: ', acum_neutral/i, '      Neg: ', acum_negative/i )
if max_i >= 0:
    print(f'\nTweet más positivo [{max_feeling}]: ')
    print(my_tweets[max_i])
if min_i >= 0:
    print(f'\nTweet más negativo [{min_feeling}]: ')
    print(my_tweets[min_i])
print(f'\n\nTiempo transcurrido para el análisis de {limits} tweets (segundos): ', timeit.default_timer()-starting_time)


# creating a histogram
plt.hist(df['Score'], 10)
plt.title('Sentiment Analysis')
plt.xlabel('Positividad')
plt.ylabel('Frecuencia')
# plt.show()



###########
## TO DO ##
###########
# 1. Convertir toda la parte de twitter a una clase para poder usar más amplicamente, con los filtros y un metodo que se llame twitter_search
# 2. Conseguir obtener información de quien publica el twitter, para generar el Excel con esa data 
# 3. Buscar más librerarías que permitan hacer Sentiment Analysis a textos en español