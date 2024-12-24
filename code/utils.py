import os
from dotenv import load_dotenv, find_dotenv
from together import Together
import warnings
import requests
import json
import time

# Initailize global variables
_ = load_dotenv(find_dotenv())
# warnings.filterwarnings('ignore')
url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/inference"
headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }


import time

def llama(prompt, 
          add_inst=True,
          model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
          temperature=0.7,
          max_tokens=1024,
          verbose=False,
          stream=False):
    
    client = Together()
    
    # Clean up the prompt to remove any previous responses
    prompt = prompt.split('[/INST]')[0] + '[/INST]' if '[/INST]' in prompt else prompt
    
    if add_inst and '[INST]' not in prompt:
        prompt = f"[INST]{prompt}[/INST]"

    if verbose:
        print(f"Prompt:\n{prompt}\n")
        print(f"model: {model}")

    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>", "[/BLK]", "[INST]"],  # Added more stop tokens
            stream=stream
        )
        
        if stream:
            full_response = ""
            for token in response:
                if hasattr(token, 'choices'):
                    content = token.choices[0].delta.content
                    print(content, end='', flush=True)
                    full_response += content
            return full_response
        else:
            return response.choices[0].message.content.split('[/BLK]')[0]  # Only return first response
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    
def llama_chat(prompts, responses, verbose=False):
    """
    Handle multi-turn conversations by maintaining context
    """
    # Create the conversation history
    messages = []
    
    # Add all previous turns to the conversation
    for i in range(len(responses)):
        messages.append({"role": "user", "content": prompts[i]})
        messages.append({"role": "assistant", "content": responses[i]})
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompts[-1]})
    
    client = Together()
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"]
        )
        
        if verbose:
            print("Full conversation:")
            for msg in messages:
                print(f"{msg['role']}: {msg['content']}\n")
            
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def get_prompt_chat(prompts, responses):
    prompt_chat = f"<s>[INST] {prompts[0]} [/INST]"
    for n, response in enumerate(responses):
        prompt = prompts[n + 1]
        prompt_chat += f"\n{response}\n </s><s>[INST] \n{prompt}\n [/INST]"

    return prompt_chat