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


def llama_guard(query, 
               model="meta-llama/Meta-Llama-Guard-3-8B", 
               temperature=0.0, 
               max_token=1024,
               verbose=False,
               base=2,
               max_tries=3):
    
    prompt = f"[INST]{query}[/INST]"
    
    data = {
      "model": model,
      "prompt": prompt,
      "temperature": temperature,
      "max_tokens": max_token,
      "stop": ["</BEGIN CONVERSATION>"]  # Add stop token to prevent repetition
    }
    if verbose:
        print(f"model: {model}")
        print("Input is wrapped in [INST] [/INST] tags")

    url = "https://api.together.xyz/inference"

    for num_tries in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                # Clean up the response by taking only the first assessment
                text = response.json()['output']['choices'][0]['text']
                # Take only the first assessment (before any repetition)
                if "safe\n</BEGIN CONVERSATION>" in text:
                    text = text.split("safe\n</BEGIN CONVERSATION>")[0] + "safe"
                elif "unsafe\n" in text:
                    text = text.split("\n</BEGIN CONVERSATION>")[0]
                return text
            else:
                print(f"API call failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error on attempt {num_tries + 1}: {str(e)}")
            if num_tries < max_tries - 1:
                print(f"Waiting {wait_seconds[num_tries]} seconds before retrying...")
                time.sleep(wait_seconds[num_tries])

    print(f"Failed to get valid response after {max_tries} attempts")
    return None


def safe_llama(query, add_inst=True, 
               model="meta-llama/Meta-Llama-Guard-3-8B",
               safety_model="meta-llama/Meta-Llama-Guard-3-8B",
               temperature=0.0, max_token=1024,
               verbose=False,
               base=2,
               max_tries=3):
    if add_inst:
        prompt = f"[INST]{query}[/INST]"
    else:
        prompt = query
    
    if verbose:
        print(f"model: {model}")
        print(f"safety_model:{safety_model}")
    
    data = {
      "model": model,
      "prompt": prompt,
      "temperature": temperature,
      "max_tokens": max_token,
      "safety_model": safety_model
    }

    # Allow multiple attempts to call the API incase of downtime.
    # Return provided response to user after 3 failed attempts.
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.json()['output']['choices'][0]['text']
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])
 
    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response  

def code_llama(prompt, 
               model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
               temperature=0.0, 
               max_tokens=1024,
               verbose=False,
               base=2,
               max_tries=3):

    url = "https://api.together.xyz/inference"  # Ensure this is the correct API endpoint

    if model.endswith("Instruct"):
        prompt = f"[INST]{prompt}[/INST]"

    if verbose:
        print(f"Prompt:\n{prompt}\n")
        print(f"model: {model}")

    data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }

    wait_seconds = [base**i for i in range(max_tries)]
    response = None
    last_exception = None

    for wait in wait_seconds:
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()['output']['choices'][0]['text']
        except Exception as e:
            last_exception = e
            print(f"Attempt failed, retrying in {wait} seconds...")
            time.sleep(wait)
    
    if response is not None and response.status_code != 500:
        return response.json()
    else:
        print(f"error message: {last_exception}")
        return None
    
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

policy = """
<BEGIN UNSAFE CONTENT CATEGORIES>
O1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
<END UNSAFE CONTENT CATEGORIES>
"""

