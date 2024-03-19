from flask import Flask,render_template,request
import re
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import joblib

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


app=Flask(__name__)
stop_words = set(nltk.corpus.stopwords.words('english'))



@app.route('/')
def home():
    return render_template("home.html")


@app.route('/prediction',methods=['get','post'])
def prediction():
    review=request.form.get("review")
    sentence = re.sub("[^a-zA-Z]", " ", review)
    sentence = sentence.lower()
    tokens = sentence.split()
    clean_tokens = [t for t in tokens if not t in stop_words]

    clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
    text=" ".join(clean_tokens)
    words_to_remove = ['product', 'quality','shuttle', 'read','flipkart','buy','cork','more']
    for word in words_to_remove:
        text= text.replace(word, '')

    text_data=[text]
    df = pd.DataFrame({'Text': text_data})
    vocab = joblib.load('count.joblib')
    X_train = vocab.transform(df['Text'])
    model=joblib.load("sentiment_model.pkl")
    output=model.predict(X_train)

    return render_template("output.html",text=text,output=output)



if(__name__)=="__main__":
    app.run(debug=True,host="0.0.0.0")