from flask import request, Flask, render_template
import pandas as pd 
import numpy as np
import time
import keras
import pickle
import tensorflow as tf
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('base.html')


@app.route('/get',methods = ['GET'])
def getname():
    name = request.args.get('name')
    return render_template('get.html', **locals())



@app.route('/form',methods=['GET','POST'])
def form():
    return render_template('form.html')


@app.route('/submit',methods=['POST'])
#df = pd.read_csv('crawler_complete.csv')


def submit():
    #df = pd.read_csv('crawler_complete.csv')#記得要加路徑 Don't forget add the whole path.
    query_submit = request.values['query']

    reconstructed_model = keras.models.load_model("Lauraceae_LSTM_sigmoid.tf")#,custom_objects={'CustomMetric':CategoricalAccuracy('balanced_accuracy')})
    f = open('Lauraceae_encode_dict.txt','r')
    encode_dict=eval(f.read())

    with open('Lauraceae_LSTM_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    def make_clickable(val,name):
        return f'<a href="{val}">{name}</a>'

    df = pd.read_csv('crawler_result_Lauraceae.csv')

    def Query(str):
        x_query = tokenizer.texts_to_sequences([str])
        print(x_query)
        for seq in x_query:
            print([tokenizer.index_word[idx] for idx in seq])
        x_query = tf.keras.preprocessing.sequence.pad_sequences(x_query, maxlen=30)
        validation = reconstructed_model.predict(x_query)
        #result = pd.DataFrame(columns=['name','score','link'])
        result = pd.DataFrame(columns=['name', 'score'])
        for key , value in zip(encode_dict.keys(), validation[0]*100):
            u = df[df["scname"]==key]['url'].squeeze()
            value = round(value, 3)
            #df_new_row = pd.DataFrame({'name':key,'score':value,'link':make_clickable(u, key)},index=[0])
            df_new_row = pd.DataFrame({'score':value, 'name':make_clickable(u, key)},index=[0])
            result = pd.concat([result, df_new_row])
            
        return result.sort_values(by='score', ascending=False, ignore_index=True)

    Query(query_submit)

    '''
    name_entry = request.values['rank']

    name = df['植物名']
    if name_entry == "":
        name_reslt = '本次並沒有進行複查排名'
    elif name_entry not in list(name):
        name_reslt = '查無此植物'
    else:
        search = list(name).index(name_entry)
        plant_name_rank = str(df.iloc[search,16])
        name_reslt = '複查植物：'+name_entry+'；  排名：'+plant_name_rank.replace('.0','')

    '''

    head_form = Query(query_submit)
    result_form = head_form.to_html

    return render_template('submit.html', **locals(), tables = [result_form(classes='data', header='true', index=False, render_links=True, escape=False)])

if __name__ == "__main__":
    app.run(debug=True)