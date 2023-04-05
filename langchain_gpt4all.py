from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# You'll need to download a compatible model and convert it to ggml.
# See: https://github.com/nomic-ai/gpt4all for more information.
llm = GPT4All(model_path="./models/gpt4all-model.bin")

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)