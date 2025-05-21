import openai
import os

# Load the OpenAI API key from an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    print("OpenAI API key not found. Ensure the OPENAI_API_KEY environment variable is set.")
    exit(1)

try:

    response = openai.chat.completions.create(
        model="gpt-4",  # Use the appropriate model identifier, assuming "gpt-4" is available
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can the baleen of a whale be eaten?"}
        ],
        stream=False
    )

    # Extracting and printing the text response
    print("API Key is working. Response:")
    print(response.choices[0].message.content)
    #print(response.choices[0].text.strip())
except Exception as e:
    print("Failed to access OpenAI API:")
    print(e)
