# impor library GUI
from flask import Flask,render_template,request
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, validators
from wtforms.validators import InputRequired, Length

# import library aplikasi
from tweepy import OAuthHandler
import tweepy
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

# instance flask
app = Flask(__name__)
app.config["DEBUG"]=True
app.config['SECRET_KEY'] = 'Thisisasecret!'

# melakukan validasi form
class CrawlForm(FlaskForm):
    keyword = StringField('Key Word', 
            validators=[InputRequired
            ('a Key Word is required!'), 
            Length(min=3, 
            message='must be at least 3 characters!')])
    tweets = IntegerField('Number of Tweets', 
            validators=[InputRequired
            ('a Number of Tweets is required!'), 
            validators.NumberRange(min=1, 
            max=1000, 
            message='The Number of Tweets must be between 1 and 1000!')])

# halaman utama
@app.route('/', methods=("POST", "GET"))
def home_page():
    return render_template('home.html')

# halaman aplikasi
@app.route('/application', methods=("POST", "GET"))
def crawling_page():
    form = CrawlForm()
    # melakukan pengujian klasifikasi sentimen (versi minimize)
    if form.validate_on_submit():
        # membaca dan membuka file
        df_model = pd.read_csv('static\data_label.csv')
        df_model.sort_values("sentimen", inplace = True) # sorting by first name
        df_model.index = np.arange(1,len(df_model)+1) # mengatur ulang index
        df_model = df_model.loc[:1000]
        
        # Splitting dataset
        X = df_model['cuitan']
        y = df_model['sentimen']
        X_train, X_test, y_train, y_test=train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=0,
                                                        shuffle=True)

        # Pembobotan kata dengan TF-IDF
        vec = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None)
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)
        y_train = y_train.astype('int')

        # Pelatihan model klasifikasi menggunakan SVM
        clf = svm.SVC(kernel='linear')
        classify = clf.fit(X_train_vec,y_train)
        pred = classify.predict(X_test_vec)
        predict = pred

        accuracy = accuracy_score(y_test, predict)
        precision = precision_score(y_test, predict)
        recall = recall_score(y_test, predict)
        f1 = f1_score(y_test, predict)

        # API twitter
        access_token = "636919082-k5KnVG8VwgRizdtAr8hBHinphwspFC5fScy1Zc3l"
        access_token_secret = "PnR4GF6HeUl3nh1pg2ASisWk39WPpBesJlYRtF9q0TW0a"
        consumer_key = "OyXkDKhUZrPmzQJdNzrBkMVO7"
        consumer_secret = "8t2Hz3bs74ZQ09753ICSb4RVFJ5B5ntJ1GflIW8187qbOBHVpY"

        # Autentikasi API Twitter
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        
        # proses crawling data tweet
        keyword = request.form['keyword']
        tweets = int(request.form['tweets'])

        t = []
        for tweet in tweepy.Cursor(api.search,
                                q=keyword + " -filter:retweets -filter:replies",
                                lang="id",
                                result_type="recent",
                                exclude_replies=True,
                                wait_on_rate_limit=True
                                ).items(tweets):
            t.append(tweet.text.encode("utf-8"))
        dictTweets = {"cuitan":t}
        df = pd.DataFrame(dictTweets,columns=["cuitan"])
        df.index = np.arange(1,len(df)+1)

        df_1 = df
        def hapus_tanda_baca(tweet):
            tanda_baca = set(string.punctuation)
            tweet = ''.join(tb for tb in tweet if tb not in tanda_baca)
            return tweet
        def cleansing(tweet):
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet) # menghapus url atau pranala
            tweet = re.sub('@[^\s]+','',tweet) # menghapus username
            tweet = re.sub(r'#([^\s]+)','',tweet) # menghapus hashtag
            tweet = re.sub(r"\d", "", tweet) # menghapus digit atau angka
            tweet = re.sub(r'\b\w{1,2}\b', '', tweet) # menghapus kata 2 huruf
            tweet = re.sub(' +',' ',tweet) # menghapus spasi ganda
            tweet = hapus_tanda_baca(tweet) # menghapus tanda baca
            tweet = tweet.lstrip(' ') # menghapus satu spasi yang tersisa di depan tweet
            tweet = tweet.strip(' ') # menghapus satu spasi yang tersisa di belakang tweet.
            tweet = tweet.rstrip("\n") # menghapus spasi enter antar baris    
            return tweet
        # menghapus emoji
        df_1 = df_1.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')) 
        # melakukan proses Cleansing
        df_1['cuitan'] = df_1['cuitan'].apply(lambda x: cleansing(x))

        # melakukan proses Case Folding
        def case_folding(text):
            text = text.lower()
            return text
        df_1['cuitan'] = df_1['cuitan'].apply(lambda x: case_folding(x))

        # melakukan proses Tokenization
        def tokenization(text):
            text = re.split('\W+', text)
            return text
        df_1['cuitan_prep'] = df_1['cuitan'].apply(lambda x: tokenization(x))

        # melakukan proses Stemming Bahasa Indonesia
        def stemming(text):
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            text = [stemmer.stem(y) for y in text]
            return text
        df_1['cuitan_prep'] = df_1['cuitan_prep'].apply(lambda x: stemming(x))

        # melakukan proses Stopword Removal Bahasa Indonesia
        def stopwords_remove(text):
            factory = StopWordRemoverFactory()
            stopword = factory.get_stop_words()
            # stopword = nltk.corpus.stopwords.words('indonesian')
            text = [word for word in text if word not in stopword]
            return text
        df_1['cuitan_prep'] = df_1['cuitan_prep'].apply(lambda x: stopwords_remove(x))

        # melakukan proses Untokenize
        def untokenize(words):
            text = ' '.join(words)
            step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
            step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
            step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
            step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
            step5 = step4.replace(" '", "'").replace(" n't", "n't")
            step6 = step5.replace(" ` ", " '")
            return step6.strip()
        df_1['cuitan_prep'] = df_1['cuitan_prep'].apply(lambda x: untokenize(x))

        # melakukan pembobotan TF-IDF
        df_1_vec = vec.transform(df_1['cuitan_prep'])

        # prediksi kelas sentimen
        result_predict = clf.predict(df_1_vec)

        # prediksi sentimen disatukan dalam dataframe
        sentimen = pd.DataFrame({'sentimen': result_predict})
        sentimen.index = np.arange(1,len(sentimen)+1) # reset index
        result = pd.concat([df_1['cuitan'], sentimen], axis=1)

        # inisialisasi polaritas
        b = []
        for i in result_predict:
            b.append(i)    
        leng = len(b)
        positif = b.count(1)
        negatif = b.count(0)

        # visualisasi diagram pie
        pers_positif = positif
        pers_negatif = negatif
        data = {'' : '', 'Positif (1)' : pers_positif, 'Negatif (0)' : pers_negatif}

        return  render_template('result.html', tables=[result.to_html(classes='data')], 
                                                        titles=df.columns.values, 
                                                        data=data, 
                                                        keyword=keyword, 
                                                        tweets=tweets,
                                                        accuracy=accuracy,
                                                        precision=precision,
                                                        recall=recall,
                                                        f1=f1
                                                        )
    return render_template('application.html', form=form)

# halaman hasil
@app.route('/result')
def result_page():
    # visualisasi diagram pie
    pers_positif = 30
    pers_negatif = 70
    data = {'' : '', 'Positif' : pers_positif, 'Negatif' : pers_negatif}
    return render_template('result.html', data=data)

