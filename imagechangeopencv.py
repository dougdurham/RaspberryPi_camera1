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


# In this version we use OpenCV to find the boundingRectangles of changes


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

    prompt1 = """This image was captured bacause motion was detected.  That means in almost all cases something is moving in
    the image.  Use this information to inform your description of the image.
    
    Produce a JSON structured description for each object you see in the image.
    Use the following fields as described here.  
    type:  Type of object (e.g. vehicle, person, animal, house, ...)
    details: Details about the object (car, garbage truck, dog, cat, )
    description: Free form description
    action: What is the object doing if that can be determined. (walking, driving, talking, jumping)

    Also include a JSON scene description that provides an overall description of the scene in 
    the image.
    
    Note: Sometimes an object will be partially occluded by a tree.  Please report on these objects.  If a
    vehicle appears to be in the middle of the street and not against a curb it is likely driving and not parked."""

    prompt2 = """This image was captured bacause motion was detected. Use this information to inform your description of the image.
    This image is of a suburban backyard. We expect most of the changes you will see will be animals moving. Please identify the specific types of animals to the best of your ability.
    Produce a JSON structured description for each object you see in the image.
    Use the following fields as described here.  
    type:  Type of object (e.g. vehicle, person, bird, dog, squirrel,  ...)
    details: Details about the object (hummingbird, poodle,  gray squirrel, ... )
    description: Free form description
    action: What is the object doing if that can be determined. (walking, flying, eating, jumping)

    Also include a JSON scene description that provides an overall description of the scene in 
    the image.
    
    Note: Sometimes an object will be partially occluded by a tree.  Please report on these objects.  """

    prompt3 = """This image was captured bacause motion was detected. Use this information to inform your description of the image.
    This image is of the interior of a home . We expect most of the changes you will see will be animals and humans moving. Please identify the specific types of animals to the best of your ability.
    Produce a JSON structured description for each object you see in the image.
    Use the following fields as described here.  
    type:  Type of object (e.g. vehicle, person, bird, dog, squirrel,  ...)
    details: Details about the object (hummingbird, poodle,  gray squirrel, ... )
    description: Free form description
    action: What is the object doing if that can be determined. (walking, flying, eating, jumping)

    Also include a JSON scene description that provides an overall description of the scene in 
    the image.
    
    Note: Sometimes an object will be partially occluded by a tree.  Please report on these objects.  """

    prompt4 = """This image was captured bacause motion was detected. Use this information to inform your description of the image.
    This image is an area where dogs feed. There are three bowls:
     * A stainless steel bowl containing water on the left
     * A food bowl sitting on the blue mat which contains high protein kibble in the center
     * A food bowl with a black rim and blue interior sitting on the tile floor which contains low protein kibble on the right.

    There are two dogs.  Cinder is a white mutt with black and brown patches.  Wallace is a black poodle.

    Cinder should only eat from the low protein bowl.  Wallace should only eat from the high protein bowl. Either can drink from the water bowl.

    Make observations in JSON format.  We want to know: dog name, dog action, and whether the dog is being good or bad.  Meaning they are eating from the correct bowl.
    Add an additional field with any general observations you may have.
    Sometimes you will see other activity.  Please report on that in JSON format using the format given above. 
    Please take care to correctly attribute bowl usage and actions.  In the past you have gotten the use of the
    low protein bowl wrong.  You have also mistaken drinking water for eating.  Please don't make those mistakes. """

    prompt = "Produce a JSON structured description for each object you see in the image.  Use the following fields as described here.  type:  Type of object examples: vehicle, person, animal, house, ... details: Details about the object examples: car, garbage truck, dog, cat, ... description: Free form description action: What is the object doing if that can be determined. examples: walking, driving, talking, jumping Also include a JSON scene description that provides an overall description of the scene in the image"

    prompt = "Produce a structured description for each object you see in the image."
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}

    # "response_format" : { "type": "json_object" },
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt4}",
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


# Use OpenCV to detect changes in the image
# Presumes both images have been converted to gray scale.
def image_changed(image1_in, image2_in):

    image1 = image1_in.copy()
    image2 = image2_in.copy()

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gaussian_parameter = (21, 21)
    gaussian_parameter = (51, 51)
    gaussian_parameter = (41, 41)
    #gaussian_parameter = (101, 101)
    blur1 = cv2.GaussianBlur(gray1, gaussian_parameter, sigmaX=0)
    blur2 = cv2.GaussianBlur(gray2, gaussian_parameter, sigmaX=0)

    #### NEW
    # Otsu's thresholding
    """
    _, otsu_thresh = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Otsu's threshold as a reference to set Canny thresholds
    edges1 = cv2.Canny(blur1, otsu_thresh * 0.5, otsu_thresh)
    edges2 = cv2.Canny(blur2, otsu_thresh * 0.5, otsu_thresh)
    diff = cv2.absdiff(edges1, edges2)
    """

    diff = cv2.absdiff(blur1, blur2)
#    _, thresh = cv2.threshold(diff, 6, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 55, 5)
    dilated = cv2.dilate(thresh, None, iterations=2)

    #cv2.imwrite("pictures/zzzblur1.jpg", blur1)
    #cv2.imwrite("pictures/zzzblur2.jpg", blur2)
    #cv2.imwrite("pictures/zzzdiff.jpg", diff)
    #cv2.imwrite("pictures/zzzthresh.jpg", thresh)
    #cv2.imwrite("pictures/zzzdilated.jpg", dilated)
    # dilated = thresh

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    AREA_THRESHOLD = 1000
    LIMIT_DISTANCE = 550
    LIMIT_DISTANCE = 650
    LIMIT_DISTANCE = 0

    found = False
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < AREA_THRESHOLD or (y + h) < LIMIT_DISTANCE:  # Define AREA_THRESHOLD as needed
            continue
        print(f"x:{x} y:{y} w:{w} z:{h}")

        cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), 2)
        found = True

    # Mark the limit box
    #print(f"Marking limit box: x:0 y:0 w:1920 z:{LIMIT_DISTANCE}")
    cv2.rectangle(image2, (0, 0), (1920, LIMIT_DISTANCE), (255, 255, 255), 2)

    return found, image2

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

    # image1 = cv2.imread("./pictures/picture_2024-03-02_12-11-47.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-02_12-13-09.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-02_11-42-52.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-02_11-59-03.jpg")

    # image1 = cv2.imread("./pictures/picture_2024-03-02_10-37-40.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-02_10-32-47.jpg")

    # image1 = cv2.imread("./pictures/picture_2024-03-02_10-19-52.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-02_10-24-50.jpg")

    # image1 = cv2.imread("./pictures/picture_2024-03-02_20-15-19.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-02_21-23-55.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-02_23-09-44.jpg")

    # image1 = cv2.imread("./pictures/picture_2024-03-04_18-13-39.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-04_18-13-54.jpg")
    # image1 = cv2.imread("./pictures/picture_2024-03-04_18-55-32.jpg")
    # image2 = cv2.imread("./pictures/picture_2024-03-04_19-00-36.jpg")
    #image1 = cv2.imread("./pictures/zzzdark1.jpg")
    #image2 = cv2.imread("./pictures/zzzdark2.jpg")

    #changed, new_image = image_changed(image1, image2)

    #if changed:
    #   cv2.imwrite("pictures/contour.jpg", new_image)
    #else:
    #   print("No change")
    #sys.exit()

    width = 1920  # Desired width in pixels
    height = 1080  # Desired height in pixels
    # tuning = picamera2.Picamera2.load_tuning_file("imx477_scientific.json")
    # tuning = picamera2.Picamera2.load_tuning_file("ov5647_noir.json")
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
    #cv2.imwrite("pictures/zzzdark1.jpg", brighten(prev_frame))
    #cv2.imwrite("pictures/zzzdark1.jpg", prev_frame)
    #print("write 1")
    #time.sleep(20)
    #prev_frame = picam2.capture_array()
    #cv2.imwrite("pictures/zzzdark2.jpg", brighten(prev_frame))
    #cv2.imwrite("pictures/zzzdark2.jpg", prev_frame)
    #print("write 2")
    #sys.exit(1)

    # _, prev_frame = cap.read()  # Read the first frame
    # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    capture_state = CaptureState.FIRST # We ignore the first movement.

    changed = False

    while True:
        frame = picam2.capture_array()
        # _, frame = cap.read()  # Read the next frame
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        #if too_dark(frame):
        if False:
            frame = brighten(frame)
            threshold = 50 # Get rid of noise
        else:
            threshold = 30 

        changed, new_frame = image_changed(prev_frame, frame)

        if changed:  # Threshold for significant change, adjust as needed
            # print("changed: {}".format(change_percentage))

            if capture_state == CaptureState.FIRST:

                # If this is the first detection we do nothing a set up for the next capture
                # We don't want to capture the first difference.
                capture_state = CaptureState.SECOND
                print("In Capture: Capture state first.  Just set to second")
                prev_frame = frame  # Update the previous frame
                time.sleep(1.5) # Give a little more time for the scene to evolve.
                continue
            elif capture_state == CaptureState.WAITING:
                print("In Capture: Capture Waiting")
                # If we are waiting for no changes then we just continue
                prev_frame = frame  # Update the previous frame
                time.sleep(1)
                continue
            else:
                # We must be in the SECOND state so we set our state
                # to waiting and make the capture.
                print("In Capture: Capture State second.  Setting to Waiting")
                capture_state = CaptureState.WAITING

            # if too_dark(gray_frame):
            #    new_frame = brighten(frame)
            # else:
            #    new_frame = frame

            now = datetime.now()
            timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
            file_name = f"picture_{timestamp}.jpg"
            cv2.imwrite(str(root_dir / file_name), cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            #cv2.imwrite(str(root_dir / file_name), new_frame)
            cv2.imwrite(str(root_dir / "prev_frame.jpg"), prev_frame)

            if use_llm and max_llm > 0:
                print("Calling LLM")
                llm_response = llm_vision_analysis(frame) # Get an analysis
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

        prev_frame = frame  # Update the previous frame

        time.sleep(1)

        # Press 'q' to quit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
