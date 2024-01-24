import os
import dotenv
import pandas as pd
from pandasai import Agent , SmartDataframe
from flask import Flask, request, jsonify
from pandasai.llm import OpenAI
dotenv.load_dotenv()
key = os.getenv('OPENAI_API_KEY')

df = pd.read_csv('processed_combined_new (1996 included)v.csv')
llm1 = OpenAI(api_token=key)
#sdf=SmartDataframe(df)
agent=SmartDataframe(df,description="A census dataframe which contains information about south africa where it is essential to filter data as per the user requirements",config={"llm": llm1,"verbose": True,"enable_cache":False,"custom_instructions":"First thing to do is to filter the data by column : year of record. If not mentioned then please filter the data by the year 2022 If the answer is not available in the dataframe then decline to answer."})

#agent.chat("what is the population of swartland")

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.data.decode('utf-8')
    response = agent.chat(str(query_text))
    return str(response)




if __name__ == '__main__':
    app.run(debug=True)
