# Prompt Engineering with Llama3

This project demonstrates prompt engineering techniques using the Llama3 model, including content safety checks with Llama Guard. It includes functions to interact with the Together AI API and handle various tasks.

## Project Structure

```
.
├── code
│   ├── utils.py
│   ├── L3_getting_started.ipynb
│   ├── L4_function_calling.ipynb
│   ├── L5_chain_of_thought.ipynb
│   ├── L6_self_consistency.ipynb
│   ├── L7_llama_guard.ipynb
│   ├── L8_walkthrough_helper_function.ipynb
│   └── .env
└── data
└── requirements.txt
```

## Files

- **utils.py**: Contains utility functions to interact with the Together AI API, including:
  - `llama()`: Main function for general Llama3 interactions
  - `llama_guard()`: Safety checking function using Llama Guard
  - `safe_llama()`: Combined function for safe interactions
  - `code_llama()`: Specialized function for code-related tasks
  - `llama_chat()`: Function for handling multi-turn conversations

### Notebooks
- **L3_getting_started.ipynb**: Introduction to using Llama3 and basic prompt engineering
- **L4_function_calling.ipynb**: Examples of using Llama3 for function calling and API interactions
- **L5_chain_of_thought.ipynb**: Demonstrations of chain-of-thought prompting techniques
- **L6_self_consistency.ipynb**: Examples of self-consistency checking in prompts
- **L7_llama_guard.ipynb**: Safety tools and content moderation with Llama Guard
- **L8_walkthrough_helper_function.ipynb**: Example of building and using helper functions for text comparison and analysis

## Models

This project uses models available through the Together AI platform. For the most up-to-date list of available models and their capabilities, please visit:
[Together AI Models](https://api.together.ai/models)

Key features of the Together AI platform:
- Access to leading open-source models
- No daily rate limits
- Fine-tuning capabilities with private data
- Scalable AI application deployment

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables in `.env`:
```
TOGETHER_API_KEY=your_api_key_here
```

## Usage

1. Import the required functions:
```python
from utils import llama, llama_guard, safe_llama
```

2. Use the functions in your code:
```python
# Basic usage
response = llama("Your prompt here")

# Safety check
safety_result = llama_guard("Content to check")

# Safe interaction
safe_response = safe_llama("Your prompt here")
```

## ☕ Support Me

If you like my work, consider supporting my studies!

Your contributions will help cover fees and materials for my **Computer Science and Engineering studies at the UOC** starting in September 2025.

Every little bit helps—you can donate from as little as $1.

<a href="https://ko-fi.com/miqueasmd"><img src="https://ko-fi.com/img/githubbutton_sm.svg" /></a>

## Acknowledgements

This project is inspired by the DeepLearning.AI courses. Please visit [DeepLearning.AI](https://www.deeplearning.ai/) for more information and resources.
