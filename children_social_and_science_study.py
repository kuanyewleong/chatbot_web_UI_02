import openai
import gradio as gr
import json

with open('OPENAI_API_KEY/OPENAI_API_KEY.txt') as f:
    key = f.readlines()
openai.api_key = str(key[0])

def json_obj_maker(head, content):
  asst_prompt = '"' + content + '"}'
  merged_dict = head + asst_prompt
  json_obj = json.loads(merged_dict)
  return json_obj

def openai_chat(customer_input):    
    # Greet and get prompt from customer
    system_start = "You are an AI assistant. The assistant is helpful, creative, polite, and very friendly."
    assistant_start = "Hi there, may I help you?"        
    # have to remove " from the text to avoid issue with json object structure
    # replace with ' instead
    translate_table = str.maketrans('"', "'")
    customer_input = customer_input.translate(translate_table)
        
    system_head = '{"role": "system", "content": '
    asst_head =  '{"role": "assistant", "content": '
    customer_head =  '{"role": "user", "content": '

    all_prompt = [
            json_obj_maker(system_head, system_start),
            json_obj_maker(asst_head, assistant_start),
            json_obj_maker(customer_head, customer_input)       
        ]
    
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.9,
    max_tokens=128,
    messages=all_prompt,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["That's all", "Thank you", "bye"]
    )
    message = response['choices'][0]['message']['content']
    return message.strip()

def get_bot(input, history=[]):    
    history += [input] # context += [query]
    # print("History+=input: ", history)
    prompt = '\n\n'.join(history)[:4096]
    # print("prompt: ", prompt)
    output = openai_chat(prompt)
    # print("output: ", output)
    history += [output]
    # print("history+=[output]: ", history)
    responses = [(u,b) for u,b in zip(history[::2], history[1::2])]    

    return responses, history

with gr.Blocks(css=".gradio-container {background-color: powderblue}" "#chatbot .overflow-y-auto{height:500px}") as demo:
    with open('welcome_text.txt') as f:
        welcome_note = f.readlines()
    chatbot = gr.Chatbot([(" ==== Social and Science Studies for Children ==== \n", None), 
                          (str(welcome_note[0]), None)], elem_id="chatbot")
    state = gr.State([])

    with gr.Row(variant='compact'):
        with gr.Column(scale=0.85):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter ... ").style(container=False)
        
    txt.submit(get_bot, [txt, state], [chatbot, state])
    txt.submit(lambda :"", None, txt)        
    
demo.launch()
# demo.launch(debug=True)