import cv2
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path
import picamera2
from libcamera import controls
import time
import os
import openai
import base64
from PIL import Image
from io import BytesIO
import requests
from requests.exceptions import (
    HTTPError,
    ConnectionError,
    Timeout,
    TooManyRedirects,
    RequestException,
)

import sys


def too_dark(gray_image):
    # Calculate the 25th percentile of brightness
    p25_brightness = np.percentile(gray_image, 25)

    # Example threshold based on 25th percentile
    if p25_brightness < 64:  # Adjust this value based on your requirements
        return True
    else:
        return False

def brighten(frame):
    # Assuming prev_frame is your image loaded previously
    # Define the gamma value, >1 to darken, <1 to brighten
    gamma = 0.5  # Use a value less than 1 to brighten the image

    # First, normalize the image to the range 0 to 1
    normalized_image = frame / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(normalized_image, gamma)

    # Convert back to 0-255 range
    brightened_image = np.uint8(gamma_corrected * 255)

    return brightened_image

def image_to_base64(in_image):
    # Assuming 'capture_array' is the numpy array from your camera
    image = Image.fromarray(np.uint8(in_image))

    # Save the image to a buffer
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    # Encode the image as base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str

def llm_vision_analysis(image):

    openai_api_key = os.getenv('OPENAI_API_KEY')

    base64_image = image_to_base64(image)

    prompt1 = """Produce a JSON structured description for each object you see in the image. 
    Use the following fields as described here.  
    type:  Type of object (e.g. vehicle, person, animal, house, ...)
    details: Details about the object (car, garbage truck, dog, cat, )
    description: Free form description
    action: What is the object doing if that can be determined. (walking, driving, talking, jumping)

    Also include a JSON scene description that provides an overall description of the scene in 
    the image.
    
    Note: Sometimes an object will be partially occluded by a tree.  Please report on these objects.  If a
    vehicle appears to be in the middle of the street and not against a curb it is likely driving and not parked."""

    prompt = "Produce a JSON structured description for each object you see in the image.  Use the following fields as described here.  type:  Type of object examples: vehicle, person, animal, house, ... details: Details about the object examples: car, garbage truck, dog, cat, ... description: Free form description action: What is the object doing if that can be determined. examples: walking, driving, talking, jumping Also include a JSON scene description that provides an overall description of the scene in the image"

    prompt = "Produce a structured description for each object you see in the image."
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt1}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 1000,
    }

    message = "" 

    try:

        response = requests.post(
         "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        json_response = response.json()

        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)

        message = json_response['choices'][0]['message']['content']

    except HTTPError as http_err:
        message = f'HTTP error occurred: {http_err}'  # Specific HTTP related errors (e.g., 404, 500)
    except ConnectionError as conn_err:
        message = f'Connection error occurred: {conn_err}'  # Problems with the network connection
    except Timeout as timeout_err:
        message = f'Timeout error occurred: {timeout_err}'  # Request timed out
    except TooManyRedirects as redirects_err:
        message = f'Too many redirects: {redirects_err}'  # Request exceeded the configured number of maximum redirections
    except RequestException as req_err:
        message = f'An error occurred: {req_err}'  # Catch-all for any other requests-related exceptions
    except Exception as err:
    # Optional: catch any other exceptions that are not related to requests
        message = f'An unexpected error occurred: {err}'

    return message


CaptureState = Enum('CaptureState', ['FIRST', 'SECOND', 'WAITING'])

def main():

    # Check if the user has provided an output file name
    if len(sys.argv) < 2:
        print("Usage: python script.py <output_file_name>")
        print("Proceeding with no LLM output")
        use_llm = False
    else:
        use_llm = True
        output_file_name = sys.argv[1]

    max_llm = 100 # maximum number of llm request to make
    root_dir = Path("./pictures")

    width = 1920  # Desired width in pixels
    height = 1080  # Desired height in pixels
    #tuning = picamera2.Picamera2.load_tuning_file("imx477_scientific.json")
    #tuning = picamera2.Picamera2.load_tuning_file("ov5647_noir.json")
    picam2 = picamera2.Picamera2()
    still_config = picam2.create_still_configuration({"size": (width, height)})

    picam2.set_controls({"AwbMode": controls.AwbModeEnum.Fluorescent})
    #    {"ExposureTime": 10000, "AnalogueGain": 1.0, "ColourGains": (2.522, 1.897)}
    picam2.configure(still_config)
    # picam2.set_controls(
    #    {"AnalogueGain": 1.0, "ColourGains": (2.822, 2.0)}
    # )
    picam2.start()

    time.sleep(2)
    prev_frame = picam2.capture_array()

    # _, prev_frame = cap.read()  # Read the first frame
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    capture_state = CaptureState.FIRST # We ignore the first movement.

    while True:
        frame = picam2.capture_array()
        # _, frame = cap.read()  # Read the next frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        if too_dark(gray_frame):
            gray_frame = brighten(gray_frame)
            threshold = 50 # Get rid of noise
        else:
            threshold = 30 

        # Calculate the absolute difference between the current frame and the previous frame
        diff = cv2.absdiff(prev_frame, gray_frame)

        # Threshold the difference to get a binary image
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        # thresh = cv2.adaptiveThreshold(
        #    diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)

        # Find the percentage of changed pixels
        change_percentage = np.sum(thresh) / (thresh.size * 255)

        if change_percentage > 0.01:  # Threshold for significant change, adjust as needed
            print("changed: {}".format(change_percentage))

            if capture_state == CaptureState.FIRST:

                # If this is the first detection we do nothing a set up for the next capture
                # We don't want to capture the first difference.
                capture_state = CaptureState.SECOND
                print("In Capture: Capture state first.  Just set to second")
                prev_frame = gray_frame  # Update the previous frame
                time.sleep(1.5) # Give a little more time for the scene to evolve.
                continue
            elif capture_state == CaptureState.WAITING:
                print("In Capture: Capture Waiting")
                # If we are waiting for no changes then we just continue
                prev_frame = gray_frame  # Update the previous frame
                time.sleep(1)
                continue
            else:
                # We must be in the SECOND state so we set our state
                # to waiting and make the capture.
                print("In Capture: Capture State second.  Setting to Waiting")
                capture_state = CaptureState.WAITING

            if too_dark(gray_frame):
                new_frame = brighten(frame)
            else:
                new_frame = frame

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
            file_name = f"picture_{timestamp}.jpg"
            cv2.imwrite(str(root_dir / file_name), new_frame)

            if use_llm and max_llm > 0:
                print("Calling LLM")
                llm_response = llm_vision_analysis(new_frame) # Get an analysis
                print("Completed calling LLM")
                with open(output_file_name, 'a') as file:
                    file.write("[{Timestamp: " + timestamp + '},  ')
                    file.write(llm_response + ']\n')

                max_llm -= 1 # Decrement the count
        else:
            # If there was no difference detected then we go back to the initial state
            # print("In Not Capture: Setting to First")
            # print("Not changed: {}".format(change_percentage))
            capture_state = CaptureState.FIRST

        prev_frame = gray_frame  # Update the previous frame

        time.sleep(1)

        # Press 'q' to quit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
