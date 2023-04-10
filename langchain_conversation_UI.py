import os
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock

with open('OPENAI_API_KEY/OPENAI_API_KEY.txt') as f:
    openai_key = f.readlines()
openai_api_key = str(openai_key[0])

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(openai_api_key=openai_api_key,temperature=0)
    chain = ConversationChain(llm=llm)
    return chain


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key # api_key
        chain = load_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use. You can find your key in your User settings --> View API keys."))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain.run(input=inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: #9cc4d7}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Children Encyclopedia Demo</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key in the top-right box. You can find your key in your OpenAI User Settings --> View API keys.",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a wide range of topics for primary school children - from Science and History to Geography, Culture, and Society",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "Tell me more about the ocean?",
            "What is the time difference between Japan and Australia?",
            "How do I improve my presentation skill?",
        ],
        inputs=message,
    )

    gr.HTML("<H3>Introducing the simplified encyclopedia for kids! (demo version)</H3>")
    gr.HTML("<b>Say goodbye to unanswered questions with our customized ChatGPT model, ready to assist your child's curiosity with accurate and engaging responses covering a vast array of subjects, from History and Geography to Science, Nature, Society, and beyond.</b>")
    gr.HTML("<b>The information is presented in a comprehensible manner, which allows for easy understanding by children. Give your child the gift of knowledge today!</b>")

    gr.HTML(
        "<center>Developed by Dr Leong Kuan Yew</a></center>")
    gr.HTML("<center>Powered by <a href='https://platform.openai.com/docs/models/overview'>ChatGPT</a> and <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>")
    
    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

block.launch(debug=True)