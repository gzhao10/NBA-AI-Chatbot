from typing import Annotated, Literal, Dict, List

import os

import autogen

import pandas as pd
import pandasql as psql

from autogen import ConversableAgent
from autogen import register_function


from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


load_dotenv()

# Setup flask server
app = Flask(__name__)
CORS(app)


# Load the data into a dataframe
file_name = "nba.csv"

data = pd.read_csv(file_name)
table_dict = {
    'data': data 
}
cols = data.columns.tolist()
context = "Table Name: data" + ". Columns: " + ", ".join(cols)

images = []
chat_history = []

# Splitter agent's system message
splitter_system_message = f"""You are a helpful AI assistant that breaks prompts down into parts when possible.
Prompts may only have one part. In this case just return the original prompt. Each independent clause in a prompt is likely a different part. 
If a part mentions any of the columns in the dataset, provided by {context}, preface the line with the word "Query: " and write the part within the same line. 
Separate each part with a newline in your response.
"""

# Generic conversation agent's system message
conversation_system_message = f"""You are a friendly and helpful AI assistant that specializes in the data provided by {context}.
Respond to any general questions or conversational prompts. If the question is not related to the dataset, you may answer if you can do so without accessing the internet.
Direct them to ask a more relevant question instead.
It is very important to check the previous messages in the conversation to avoid redundancy.
"""

# Summarizer agent's system message
summary_system_message = f"""You are a helpful AI assistant.
Generate a SQL query based on the prompt. All column names must be surrounded by quotes. The table name must not be surrounded by quotes. Make sure the column names are accurate. Use {context} as context. 
If a prompt asks for data not in the dataset, or to predict the future, use the column with the most relevance in the SQL query.
Run the SQL query and provide a summary of the results in natural language. Keep in mind the question to be answered. Each line in the response should be separated by one newline.
"""

# Visualizer Agent's system message
visualizer_system_message = f"""You are a helpful AI assistant that creates data visualizations.
Generate a SQL query based on the prompt. All column names must be surrounded by quotes. The table name must not be surrounded by quotes. Make sure the column names are accurate. Use {context} as context. 
Keep in mind the user prompt when generating the title of the graph and the axes labels. 
"""


llm_config = {"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]}

def create_agent(name: str, system_message: str, llm_config: Dict) -> ConversableAgent:
    return ConversableAgent(
        name=name,
        system_message=system_message,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

# Human agent
user_proxy = ConversableAgent(
    "user_proxy",
    llm_config=False,
    human_input_mode="NEVER",  
)


# Splits prompt into parts and either generate sql/call conversationalist
splitter = create_agent("splitter", splitter_system_message, llm_config)
# Engages in unrelated conversation
conversationalist = create_agent("conversationalist", conversation_system_message, llm_config)
# Summarizes SQL results
summarizer = create_agent("summarizer", summary_system_message, llm_config)
# Visualizes SQL results
visualizer = create_agent("visualizer", visualizer_system_message, llm_config)


# Get the most recent message from the agent 
def get_response(chat_result):
    entry = chat_result.chat_history[-1]
    return entry['content']


# Function to execute SQL code on a CSV file treated as a database using pandasql
@user_proxy.register_for_execution()
@summarizer.register_for_llm(description="Execute the SQL")
def execute_sql(sql_code: Annotated[str, "Code to execute."]):
    try:
        # Use pandasql to run the SQL query on the DataFrame
        result = psql.sqldf(sql_code, table_dict)
        result_dict = result.to_dict(orient='records')
        return result_dict
    except Exception as e:
        return None, f"Error: {str(e)}"

# Function to create a graph
@user_proxy.register_for_execution()
@visualizer.register_for_llm(description="Visualize the results")
def visualize(
    sql_code: Annotated[str, "SQL query"],
    title: Annotated[str, "The title of the chart"],
    ):

    plt.figure()
    data = execute_sql(sql_code)

    plt.title(title)
    axes = list(data[0].keys())
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.subplots_adjust(bottom=0.3)

    xvals = [entry[axes[0]] for entry in data]
    yvals = [entry[axes[1]] for entry in data]
    x = np.array(xvals)
    y = np.array(yvals)

    plt.bar(x,y)

    timestamp = str(int(time.time()))
    filename = f'visual_{timestamp}.png'
    images.append(filename)
    plt.savefig(filename)
    plt.close()
    return filename


# Function to retrieve chat history
def get_history():
    res = ""
    if len(chat_history ) > 0:
        res += ". Previous question: " + chat_history[-2] + "; " + "Previous answer: " + chat_history[-1]
    return res


@app.route('/get-image/<filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(directory='.', path=filename, mimetype='image/png')



# Orchestrate the agents to perform the SQL operations
@app.route("/get-ai-message", methods=["POST"])
def get_ai_message():
    user_prompt = request.json.get('query', '')
    response = ""
    image_url = None

    start_time = time.time()

    #Split the user prompt into smaller tasks if possible
    splitter_result = user_proxy.initiate_chat(
        splitter,
        message=user_prompt,
        max_turns = 1,
    )
    prompts = get_response(splitter_result).split("\n")

    splitter_time = time.time() - start_time
    print(f"Splitter time: {splitter_time:.2f} seconds")
    
    for prompt in prompts:
        # Use conversationalist if no SQL was generated in the response
        if prompt != "":
            if 'Query' not in prompt:
                convo_start_time = time.time()
                convo_result = user_proxy.initiate_chat(
                    conversationalist,
                    message = prompt + get_history(),
                    max_turns = 1
                )
                response += get_response(convo_result)
                convo_time = time.time() - convo_start_time
                print(f"Conversationalist time: {convo_time:.2f} seconds")
            else:
                # Use the visualizer if the prompt contains the word visualize
                if "visual" in prompt.lower():
                    visual_start_time = time.time()
                    visual_result = user_proxy.initiate_chat(
                        visualizer,
                        message=prompt + get_history(),
                        max_turns = 2
                    )
                    filename = images[-1]
                    image_url = f"http://127.0.0.1:5000/get-image/{filename}"
                    visualizer_time = time.time() - visual_start_time
                    print(f"Visualizer time: {visualizer_time:.2f} seconds")
                    
                # Otherwise summarize the SQL query
                else:
                    summary_start_time = time.time()
                    summary_result = user_proxy.initiate_chat(
                        summarizer,
                        message=prompt + get_history(),
                        max_turns = 2,
                    )
                    response += get_response(summary_result) + '\n\n' 
                    summarizer_time = time.time() - summary_start_time
                    print(f"Summarizer time: {summarizer_time:.2f} seconds")
    
    chat_history.append(user_prompt)
    chat_history.append(response)
    return jsonify({"role": "assistant", "content": response, "image_url": image_url})
    


if __name__ == '__main__':
    app.run(port=5000)
