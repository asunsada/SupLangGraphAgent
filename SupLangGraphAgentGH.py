################## SUPERVISOR AGENT + GRAPH + GEMINI +  WORKING CODE
#Github
#https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb
#This is a demo code that used Google Gemini as LLM, Langchain agents (Gmail Toolkit) and a langgraph with
# a supervisor node that decides how to answer the user's prompt, whether choosing an email agent 
# or choosing an LLM agent and orchestrates the flow among the nodes.
# I also make Gemini talk which is fun.

# in order to execute, add your own GOOGLE_API_KEY and the directory where you have your credentials for GMAIl.

import os
import json
from typing import Dict, TypedDict, Optional
import os
from langchain_core.prompts import ChatPromptTemplate
import ssl
# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context ### so https work

#### TEXT TO SPEECH (it is fun to have Google Gemini talking ...

from gtts import gTTS

def text_to_speech(text_to_speech):
        # Initialize gTTS with the text to convert
        speech = gTTS(text_to_speech)
        # Save the audio file to a temporary file
        speech_file = 'speech.mp3'
        speech.save(speech_file)
        # Play the audio file
        os.system('afplay ' + speech_file)

# Add your Google Gemini API Key
os.environ["GOOGLE_API_KEY"] =

generation_config = {
    "temperature": 0,
    "top_p": 0.9,
    "top_k": 2,
    "max_output_tokens": 512,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]


import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",generation_config=generation_config, safety_settings=safety_settings)


from langchain_google_genai import (ChatGoogleGenerativeAI, HarmBlockThreshold,HarmCategory)
llm1 = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", convert_system_message_to_human=True, handle_parsing_errors=True, temperature=0, max_tokens=200, safety_settings={HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH})


#********************** PART 1 *********************************#
############ ****** DEFINE TOOLS FOR THE AGENTS #################
from langchain.agents import Tool

## Gmail searcher TOOLS-
###### GMAIL API USINg LANGCHAIN
output_directory= <Directory where the API gmail credentials is stored>
path1 = os.path.dirname(output_directory)

from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file=path1 +"credentials.json", ### location of the credentials.json file downloaded from Google API site
)

from langchain_community.agent_toolkits import GmailToolkit
toolkit = GmailToolkit()

from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)
gmail_tool = toolkit.get_tools()


tools_dict = {}
for tool in gmail_tool:
    print(f"name: {tool.name}")
    print(f"Description: {tool.description}")
    #print(f"Function: {tool.func}")



################ AGENT  CODE   ############################
## CREATE AGENTS

from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
#from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain import hub

### Create agent to search in the LLM

input1=['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools']
# Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
# create the prompt
template1: str = """
Your name is “Secretary”. You are world class ai personal assistant computer that created by me(A independent AI engineer). 
You are highly advanced intelligent. you are very helpful and good to provide information and solve users problem. Secretary is a AELM 
model (Auto Execution Language Model).)
Secretary” is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and 
discussions on a wide range of topics. As a language model, “ Secretary” is able to generate human-like text based on the input it receives, 
allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Secretary” is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand 
large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. 
Additionally, “ Secretary” is able to generate its own text based on the input it receives, allowing it to engage in discussions
and provide explanations and descriptions on a wide range of topics.
Overall, “ Secretary” is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on 
a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, 
“ Secretary” is here to assist. You will access the email of the user. 



TOOLS:
-----
"Secretary" Assistant has access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
To use a tool, please use the following format:
Thought: you should always think about what to do. Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
Thought: Do I need to use a tool? No. I now know the final answer
Final Answer: the final answer to the original input question [your response here]

Begin!    Previous conversation history:
{chat_history} Question: {input} {agent_scratchpad})
"""




# Create the PromptTemplate instance using the correct constructor
sys_prompt = PromptTemplate(
    input_variables=input1,
    template=template1
)

# If you want to create a new PromptTemplate instance from another instance or template, do it like this:
prompt = PromptTemplate(
    input_variables=sys_prompt.input_variables,
    template=sys_prompt.template
)



prompt = prompt.partial(
    #tools=render_text_description(tools),
    tools=gmail_tool,
    tool_names=", ".join([t.name for t in gmail_tool]),
)



llm_with_stop = llm1.bind(stop=["\nObservation"])
memory = ConversationBufferMemory(memory_key="chat_history", k=5) ## remeber the last 5 interactions



agent_email = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: x["intermediate_steps"],
        "chat_history": lambda x: x["chat_history"],
        "tools":lambda x: gmail_tool
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)



############ GRAPH CODE ###############
################# INITIALIZE GRAPH AND DEFINE STATE  VARIABLES
from langgraph.graph import StateGraph, END
class GraphState(TypedDict):  #### Holding state variables
    question: Optional[str] = None  #### the user input to the LLM
    decision: Optional[str] = None   #### the Supervisor decision for which agent to call,  email or LLM or finish
    response: Optional[str] = None #### LLM response

workFlow = StateGraph(GraphState)

############FUNCTIONS FOR THE GRAPH
def sup_decides(question):  ### decides what agent to use, LLM or email
    print(question)
    prompt= ("classify searching method of "
             +question+
             " as email or LLM. Output just the class, either email or LLM. ")
    llm_response= llm.generate_content(prompt).text
    print(f'llm_response: {llm_response}')
    return llm_response.strip()

def sup_decides_node(state):
    # If decision is "LLM agent done" or "email agent done", then go to "bye"
    if state.get('decision') in ["LLM agent done", "email agent done"]:
        state['decision'] = "go to bye"
    else:
        question = state.get('question', '').strip()
        decision = sup_decides(question)
        state['decision'] = decision
        print(f'decision_input_node: {state}')
    return state


def email_agent_node(state):
    print(f'I am in email agent')
    question = state.get('question', '').strip()
    print(f'question:{question}')

    agent_executor = AgentExecutor(agent=agent_email, tools=gmail_tool, verbose=True, memory=memory)
    state['response'] = agent_executor.invoke({"input": task})["output"]
    text_to_speech( state['response'])
    # if above is question, it does not work. it needs to be task.
    #state['response'] = agent_executor.invoke({"input": question})["output"]
    state['decision'] = "email agent done"
    print(state['decision'])
    return state
def LLM_agent_node(state):
    print(f'I am in  LLM agent')
    question = state.get('question', '').strip()
    state['response'] = llm.generate_content(question).text
    print(llm.generate_content(question).text)
    state['decision'] = "LLM agent done"
    print(state['decision'])
    return state

def bye(state):
    #current_response = state.get('response')
    print(f'I am in bye function')
    state['response']= "#########  The graph has finished   #######"
    return state

######### CREATE GRAPH
## Nodes: Functions. They do sth and update the value of vble State
workFlow.add_node("sup_decides", sup_decides_node)
workFlow.add_node("email_agent", email_agent_node)
workFlow.add_node("web_agent", LLM_agent_node)
workFlow.add_node("bye", bye)


### Add edges: what node connects to what node.
workFlow.set_entry_point("sup_decides") #entry
workFlow.add_edge("email_agent", "sup_decides") # from email always go back to sup
workFlow.add_edge("LLM_agent", "sup_decides") # from LLM always go back to sup
#workFlow.add_edge("sup_decides", "bye")
workFlow.add_edge("bye", END) #from email always go to end


def decide_next_node(state):
    sd=state.get('decision')
    print(f'state in decide_next_node: {sd}')  #ie.: {'question': "Tell me about what is today's top news", 'decision': 'LLM', 'response': None}
    # Supervisor decides which node to go to next. If email, email, if LLM agent then LLM or bye
    return "email_agent" if state.get('decision') == "email" \
        else \
        ("LLM_agent" if state.get('decision') == "LLM"
         else "bye")

### Conditional edges. From node decide next node we go to either email or LLM or bye node depending on the result of fucntion
#decide_next_node which return go to email if supervisor thinks the email agent is better suited to take care of the task or LLM agent or finish
workFlow.add_conditional_edges(   ## sup choice to go to email, or LLM or bye based on result of function decide_next_node
    "sup_decides",
    decide_next_node,
    {
        "email_agent": "email_agent",
        "LLM_agent": "LLM_agent",
        "bye": "bye"
    }
)
## we have completed the graph
app = workFlow.compile()

######## print grpah
# Assuming you have an instance of CompiledStateGraph
app_graph = app.get_graph(xray=True)  # Adjust this line according to your app's method



# Define a function to print the graph
def print_graph(graph):
    # Assuming graph has methods or attributes to access nodes and edges
    nodes = graph.nodes  # Adjust according to actual method/attribute to access nodes
    edges = graph.edges  # Adjust according to actual method/attribute to access edges

    print("Nodes:")
    for node in nodes:
        print(f"  {node}")

    print("\nEdges:")
    for edge in edges:
        print(f"  {edge[0]} -> {edge[1]}")


# Print the graph
print_graph(app_graph)
app_graph.print_ascii()


### We invoke the code # Depending on the question, the graph will follow a path or another.
#app.invoke({"question": "Hey dude, how is everything?"})
#app.invoke({"question": "Find the email from NJ Devils"})
#app.invoke({"question": "Find an email about the school in my gmail"})
#task="There is a recent email from the NJ Devils Hockey with a list of tasks for me to do. Can you please check and let me know what the email is asking me to do?"
#task="Check my email and make a list of all the things the senders are asking me to do."
task="What is good about Paris?"


#task="List all the places and the dates. You have permission to access my email"
#task="Search the LLM using DuckDuck function for the meaning of resting"
app.invoke({"question": (task)})

