from flask import Flask, request, jsonify
import os
import dotenv
import pandas as pd
from langchain.chat_models import ChatOpenAI
from pandasai import SmartDataframe
from langchain.agents import initialize_agent, Tool, AgentType
from pandasai.llm import OpenAI
from langchain_core.messages import SystemMessage
dotenv.load_dotenv()
key = os.getenv('OPENAI_API_KEY')

# Initialize Flask application
ap = Flask(__name__)

# Load dataframes
df = pd.read_csv('processed.csv')
df1 = pd.read_csv('processed_1.csv')
df2 = pd.read_csv('processed_2.csv')

llm_agent = ChatOpenAI(api_key=key,temperature=0, model="gpt-4")
llm1 = OpenAI(api_token=key)
llm2 = OpenAI(api_token=key)
llm3 = OpenAI(api_token=key)


sdf = SmartDataframe(df, config={"llm": llm1,"verbose": True,"enable_cache":False,"custom_instructions":"If the answer is not available in the dataframe then decline to answer"})
sdf1 = SmartDataframe(df1, config={"llm": llm2,"enable_cache":False,"custom_instructions":"If the answer is not available in the dataframe then decline to answer"})
sdf2 = SmartDataframe(df2,config={"llm": llm3,"enable_cache":False,"custom_instructions":"If the answer is not available in the dataframe then decline to answer"})

tools=[
    Tool(name="pdai1", func=sdf.chat, description="Always use this tool to respond to queries which do not mentions a date prior to 2022. Do not use this tool if the query is a year prior to 2022.Do not use this tool to queries which mentions year before 2022"),
    Tool(name="pdai2", func=sdf1.chat, description="use this tool to respond unless the query mentions a year or a date between 2011 and 2021."),
    Tool(name="pdai3", func=sdf2.chat, description="use this tool to respond unless the query mentions a year or a date before 2011.")
    ]

system_messag="""You are an AI assisstant named kreeda who answers questions about South Africa and facts about its cities, provinces such as population, education, energy, etc.Decline to answer if question is not related to you description You must respondonly using the three tools : pdai1, pdai2 and pdai3 You have only three options to respond to a query. respond with you don't know if an options does not return an answer

1. If the query does not mentions a date or a year. Use pdai1
2. If the query mentions a date or year between 2011 and 2021 then use pdai2
3. If the query mentions a date or a year before 2011 then use pdai3 tool 
If none of the tools is suitable for the query then respond with you don't know.
"""
system_message = SystemMessage(content=system_messag)
chat_history = [system_message]
agent = initialize_agent(
    tools,
    llm_agent,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)
@ap.route('/',methods=['GET'])
def index():
    return "Hello World"
@ap.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query', '')


    response = agent.run(input=query,chat_history=chat_history)

    return jsonify({'response': str(response)})

if __name__ == '__main__':
    ap.run(debug=True)