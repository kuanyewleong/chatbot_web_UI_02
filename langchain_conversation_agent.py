from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent

with open('GOOGLE_API_KEY/GOOGLE_API_KEY.txt') as f:
    google_key = f.readlines()
google_api_key = str(google_key[0])

with open('OPENAI_API_KEY/OPENAI_API_KEY.txt') as f:
    openai_key = f.readlines()
openai_api_key = str(openai_key[0])

search = GoogleSearchAPIWrapper(google_api_key=str(google_key[0]), google_cse_id=str(google_key[1]))
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

llm=ChatOpenAI(openai_api_key=openai_api_key, 
           model_name='gpt-3.5-turbo',
           temperature=0)
agent_chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)


while (True):
    user_input = input()
    output = agent_chain.run(user_input)
    print(output)




# input1 = "Hi, I am Leong. What is the time in New York now?"
# output1 = agent_chain.run(input1)
# print(output1)

# input2 = "What is my name?"
# output2 = agent_chain.run(input2)
# print(output2)