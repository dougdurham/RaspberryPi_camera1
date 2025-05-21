# RaspberryPi_camera1

This project utilizes a Raspberry Pi with a camera module to perform motion detection in a video stream. It includes Python scripts that can capture images upon detecting motion and optionally send these images to OpenAI's GPT-4 Vision API for analysis and description. The project provides different approaches to motion detection using the OpenCV library.

## `imagechangeopencv.py`

This script implements motion detection using a Raspberry Pi camera (`picamera2`) and OpenCV (`cv2`).

**Core Functionality:**
- Captures images from the `picamera2` module.
- Preprocesses images for motion detection:
    - Converts images to grayscale.
    - Applies Gaussian blur to reduce noise.
    - Uses `cv2.absdiff` to find differences between consecutive frames.
    - Applies adaptive thresholding to the difference image.
    - Dilates the thresholded image to fill in gaps in detected regions.
- Finds contours in the processed image to identify areas of significant change. Rectangles are drawn around these areas in the output image.
- Saves images when motion is detected, subject to certain conditions (area threshold, distance limit).
- Includes a function to brighten images if they are too dark.
- Implements a state machine (`CaptureState`) to manage image capture logic:
    - `FIRST`: Initial state, ignores the first motion detected.
    - `SECOND`: After first motion, ready to capture on next motion.
    - `WAITING`: After a capture, waits for a period of no motion before returning to `FIRST`.

**OpenAI GPT-4 Vision Analysis (Optional):**
- If a filename is provided as a command-line argument, the script will:
    - Convert the captured image (where motion was detected) to a base64 string.
    - Send the image to the OpenAI GPT-4 Vision API for analysis.
    - The script contains several specialized prompts tailored for different scenes (e.g., general motion, backyard, home interior, specific dog feeding scenarios).
    - The JSON response from the API, along with a timestamp, is appended to the specified output file.
- This feature is limited by the `max_llm` variable in the script.

**Dependencies:**
- `opencv-python` (cv2)
- `picamera2`
- `numpy`
- `Pillow` (PIL)
- `requests`
- `openai` (if using the vision analysis feature)

## `imagechangetest.py`

This script also performs motion detection using `picamera2` and `cv2`, offering a slightly different approach compared to `imagechangeopencv.py`.

**Core Functionality:**
- Captures images from the `picamera2` module.
- Preprocesses images for motion detection:
    - Converts images to grayscale.
    - Calculates the absolute difference (`cv2.absdiff`) between the current grayscale frame and the previous one.
    - Applies a binary threshold to the difference image.
    - Calculates the percentage of changed pixels in the thresholded image.
- If the `change_percentage` exceeds a predefined threshold, it signifies motion.
- Saves the original color image when significant motion is detected.
- Includes a function to brighten images if they are too dark.
- Implements the same `CaptureState` logic as `imagechangeopencv.py` (`FIRST`, `SECOND`, `WAITING`) to manage when images are saved.

**OpenAI GPT-4 Vision Analysis (Optional):**
- Similar to `imagechangeopencv.py`, if a filename is provided as a command-line argument:
    - The captured color image (where motion was detected) is converted to base64.
    - The image is sent to the OpenAI GPT-4 Vision API for analysis using a general prompt.
    - The JSON response from the API, along with a timestamp, is appended to the specified output file.
- This feature is also limited by the `max_llm` variable.

**Dependencies:**
- `opencv-python` (cv2)
- `picamera2`
- `numpy`
- `Pillow` (PIL)
- `requests`
- `openai` (if using the vision analysis feature)

## `openaitest.py`

This is a utility script designed to test the connection and functionality of the OpenAI API.

**Functionality:**
- Loads the OpenAI API key from the `OPENAI_API_KEY` environment variable.
- Sends a hardcoded sample query ("Can the baleen of a whale be eaten?") to the specified OpenAI model (e.g., "gpt-4").
- Prints the API's response to the console.
- This helps verify that the API key is correctly configured and that the OpenAI service is accessible.

**Dependencies:**
- `openai`

## Setup/Prerequisites

1.  **Hardware:**
    *   A Raspberry Pi (e.g., Raspberry Pi 3B+, 4, Zero 2 W) with a compatible camera module (e.g., Camera Module V1, V2, HQ Camera) correctly connected.

2.  **Environment Variable:**
    *   For scripts that use the OpenAI API (`imagechangeopencv.py`, `imagechangetest.py`, `openaitest.py`), you need to set the `OPENAI_API_KEY` environment variable.
        ```bash
        export OPENAI_API_KEY='your_openai_api_key_here'
        ```
        You can add this line to your shell's configuration file (e.g., `~/.bashrc` or `~/.zshrc`) for persistence.

3.  **Python Packages:**
    *   Install the necessary Python libraries. You can typically install them using pip:
        ```bash
        pip install opencv-python picamera2 numpy Pillow requests openai
        ```
    *   Note: `picamera2` might have specific installation instructions or dependencies depending on your Raspberry Pi OS version. Refer to the official `picamera2` documentation if you encounter issues. OpenCV can also sometimes be tricky to install on Raspberry Pi; pre-compiled wheels or specific versions might be needed.

## Usage

### `imagechangeopencv.py`
This script monitors for motion and saves images with detected changes highlighted.

-   **To run without OpenAI analysis:**
    ```bash
    python imagechangeopencv.py
    ```
    Images will be saved in a `pictures` directory (created if it doesn't exist) in the same directory as the script.

-   **To run with OpenAI analysis:**
    ```bash
    python imagechangeopencv.py output_llm.txt
    ```
    Replace `output_llm.txt` with your desired file name for storing the OpenAI API responses. The script will append JSON responses to this file. Ensure your `OPENAI_API_KEY` is set.

### `imagechangetest.py`
This script provides an alternative motion detection method.

-   **To run without OpenAI analysis:**
    ```bash
    python imagechangetest.py
    ```
    Images will be saved in a `pictures` directory.

-   **To run with OpenAI analysis:**
    ```bash
    python imagechangetest.py output_llm_test.txt
    ```
    Replace `output_llm_test.txt` with your desired file name. Ensure your `OPENAI_API_KEY` is set.

### `openaitest.py`
This script tests your OpenAI API setup.

-   **To run:**
    ```bash
    python openaitest.py
    ```
    Ensure your `OPENAI_API_KEY` is set. The script will print the API response to the console.
