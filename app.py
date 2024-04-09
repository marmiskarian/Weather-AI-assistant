import openai
import requests
import json
import warnings
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from pathlib import Path

with open('apikey.txt', 'r') as f:
    openai.api_key = f.read()

with open('weatherapikey.txt', 'r') as f:
    WEATHER_API_KEY = f.read()
    
# Step 1: Use OpenAI Chat Completions API to get the location of the user if it is not given. If it is given, use function
# calling for getting weather api parameter(s).
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
]

def chat_completion_request(messages, tools=None, tool_choice=None, model="gpt-4-1106-preview"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key, 
    }
    json_data = {"model": model, "messages": messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        response_json = response.json()
        # Extracting the translated text from the response
        translated_text = response_json["choices"][0]["message"]["content"]
        return translated_text
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return None

def get_location(input_text):
    # Define a prompt that instructs the AI to extract the city name
    prompt = [
        {"role": "system", "content": "Extract the city name from the user's input."},
        {"role": "user", "content": f"{input_text}"}
    ]

    # Use the chat_completion_request function to send the prompt
    location = chat_completion_request(prompt)
    return location

# Step 2: Call weather api for the given location.
# Step 3: Call Chat Completions API again for processing the response of weather api. Make it to provide short answer like this:
# "The temperature in Yerevan is -1.91 degrees Celsius".
    
def get_weather(location):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data['main']['temp']

        weather_message = [
            {"role": "system", "content": "You are weather assistant"},
            {"role": "user", "content": f"In location {location} the current temperature is {temperature} degrees Celsius. Provide short answer stating this fact to user."},
        ]

        return chat_completion_request(weather_message, model="gpt-4-1106-preview")
    else:
        print(f"Error fetching weather data: {response.status_code}")
        return None

# Step 4: Call Chat Completions API again to translate the output of Step 3 into Armenian.
def translate_text(text, target_language="Armenian"):
    messages = [
        {"role": "system", "content": "You are a highly intelligent translator."},
        {"role": "user", "content": f"Translate the given English text to {target_language}: {text}"}
    ]
    translated_text = chat_completion_request(messages,tools=tools)
    return translated_text

# Step 5: Use OpenAI Text to Speech API to create an audio version (mp3) of the output of Step 4.
warnings.filterwarnings("ignore", category=DeprecationWarning)
def text_to_audio(text, file_path="translation.mp3"):
    try:
        response = openai.audio.speech.create(
            model="tts-1-hd",
            voice="alloy",
            input=text
        )
        response.stream_to_file(file_path)
        print(f"Audio file created successfully at: {file_path}")
    except Exception as e:
        print(f"Failed to create audio file: {e}")

# Step 6: Use one of OpenAIs APIs to create an image based on the output of Step 3 (text to image).
def text_to_image(city, weather):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=f"create nice image of {city} that has {weather} weather, dont include any text",
        size="1024x1024",
        quality="hd",
        n=1
    )
    return response.data[0].url

# Step 7: Use one of OpenAIs APIs to extract text (if any) from the output of Step 6.
def image_to_text(image_url):
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
          {
            "role": "user",
            "content": [
              {"type": "text", "text": "What’s in this image?"},
              {
                "type": "image_url",
                "image_url": {
                  "url": image_url,
                  "detail": "high"
                },
              },
            ],
          }
        ],
        max_tokens=300
    )

    return response.choices[0].message.content

# Step 8: Create a Chainlit app that will answer the questions mentioned at the beginning and will output the
# outputs of Steps 3, 4, 5, 6, and 7.
import chainlit as cl

# Define the function to handle incoming messages
@cl.on_message
async def main(message: cl.Message):
    
    user_input = message.content

    # STEP 1: Extract location from user input
    location = get_location(user_input)
    if location:
        # STEP 2, 3: Get weather information for the extracted location
        weather = get_weather(location)
        if weather:
            # Print the original weather response
            await cl.Message(content=weather).send()

            # STEP 4: Translate the weather response
            translated_response = translate_text(weather)
            if translated_response:
                # Print the translated response
                await cl.Message(content=translated_response).send()

                # STEP 5: Convert translated text to audio
                text_to_audio(translated_response, "weather_audio.mp3")

                # STEP 6: Generate an image based on location and weather
                image_url = text_to_image(location, weather)
                # Print the image URL
                await cl.Message(content=f'Image URL: {image_url}').send()

                # STEP 7: Extract text from the generated image
                extracted_text = image_to_text(image_url)
                # Print the extracted text from the image
                await cl.Message(content=f'Extracted text from image: {extracted_text}').send()
            else:
                await cl.Message(content="Translation failed.").send()
        else:
            await cl.Message(content="Failed to fetch weather data.").send()
    else:
        await cl.Message(content="Failed to extract location from user input.").send()
