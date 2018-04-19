import pickle
from flask import Flask, render_template, request, redirect
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
app.vars = {}

app.vars['tokenizer'] = pickle.load(open('comment-tokenizer.pkl','rb'))
app.vars['model'] = load_model('cat-cross-model-e2.h5')
graph = tf.get_default_graph()
max_len = 200

lorem_ipsum = "Paste your comment, email, message board post, or other communcation here\n\nThen hit the submit button above"

@app.route('/', methods =['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', comment = lorem_ipsum, score = "")
    else:
        comment = str(request.form['comment'])
        global graph
        with graph.as_default():
            #do inference here
            tokens = app.vars['tokenizer'].texts_to_sequences([comment])
            arr = pad_sequences(tokens, maxlen=max_len)
            pred = app.vars['model'].predict(arr)[0][1]
            print(comment, pred)
            if pred > 0.5:
                tox_score = f'We\'re {pred*100:{5}.{5}}% certain that this comment is Toxic'
            else:
                tox_score = f'We\'re {(1 - pred)*100:{5}.{5}}% certain that this comment is Civil'
        return render_template('index.html', comment=comment, score = tox_score)
if __name__ == '__main__':
    app.run(debug=True)
