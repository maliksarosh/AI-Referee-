# AI-Referee-Here is a complete, detailed README file for Version 1.0 of your AI Football Referee.
You can create a new file in your project folder named README.md and paste the following content into it.
code
Markdown
# AI Football Referee - Version 1.0

This project is an automated, offline video analysis tool designed to detect and analyze player interactions in football/soccer footage. It processes multiple videos from a folder, uses a combination of computer vision models to track players and their skeletons, and applies a rule-based "brain" to identify events like ball possession, tackle attempts, and specific types of fouls.

## Core Features

-   **Batch Video Processing**: Automatically finds and processes all `.mp4` videos in a specified folder.
-   **Multi-Object Detection**: Uses a custom-trained YOLOv5 model to detect players and the ball.
-   **Stable Player Tracking**: Implements the SORT tracking algorithm to assign a unique and persistent ID to each player throughout the video.
-   **On-Demand Skeleton Analysis**: For high performance, it runs a fast object tracker on every frame but only activates the heavy MediaPipe Pose model for detailed skeletal analysis during specific "tackle events."
-   **Intelligent Tackle Detection**: The "Referee Brain" uses a multi-stage logic to identify tackle attempts, considering:
    -   **Dynamic Proximity**: The "personal space" bubble around the ball carrier adjusts based on their speed.
    -   **Intent Analysis**: A challenger is only flagged if their velocity vector indicates they are moving *towards* the player with the ball.
-   **Foul Detection**: Implements a time-based rule to detect a dangerous slide tackle by analyzing the sequence of a tackler sliding and the victim subsequently falling.
-   **Automated Snapshot Generation**: When a tackle event is detected, the system automatically saves an annotated snapshot of the frame to an output folder for review.

## System Architecture

The system uses a multi-layered, on-demand architecture to balance performance and analytical depth.

1.  **Input**: The system takes a video file as input.
2.  **Fast Pass (Every Frame)**:
    -   **YOLOv5**: Detects bounding boxes for players and the ball.
    -   **SORT Tracker**: Takes YOLO's player boxes and assigns stable track IDs.
    -   **RefereeBrain (Fast Check)**: Receives the tracked player data. It calculates player velocities and performs the "Proximity + Intent" check to see if a tackle event is happening.
3.  **Slow Pass (On-Demand)**:
    -   If the brain signals a tackle event, **MediaPipe Pose** is activated on the current frame to get detailed skeletons.
    -   **RefereeBrain (Foul Check)**: The brain receives the skeletons, reliably matches them to the tracked player IDs using IOU, and runs its advanced foul detection logic.
4.  **Output**:
    -   An annotated video stream is displayed showing player IDs, roles (Possessor, Tackler), and foul alerts.
    -   Annotated JPEG snapshots of each tackle event are saved to the `Tackle_Snapshots` folder.

## Setup and Installation

Follow these steps to set up and run the project.

### 1. Prerequisites
-   Python 3.9+
-   Git (for cloning the repository)

### 2. Project Setup
```bash
# Clone this repository (example URL)
git clone https://github.com/your-username/AI-Referee-Engine.git
cd AI-Referee-Engine

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
This project uses several libraries. A requirements.txt file is provided for easy installation.
code
Bash
# Install all required libraries
pip install -r requirements.txt
(If you do not have a requirements.txt file, create one and add the following lines, then run the command above):
code
Code
# requirements.txt
torch
torchvision
yolov5
mediapipe
opencv-python
scikit-image
filterpy
numpy
scipy
4. Download Models and Tracker
This project requires several files that must be downloaded manually.
YOLOv5 Model: Place your custom-trained best.pt file inside a subfolder. The default path is yolov5n-football\best.pt.
MediaPipe Pose Model: Download the following file and place it in the main project directory:
pose_landmarker_full.task
SORT Tracker: Download the sort.py file from the original repository. Important: You must then open the file and delete the if __name__ == '__main__': block at the end.
Download sort.py
5. Project Structure
Ensure your project directory is organized as follows:
code
Code
AI-Referee-Engine/
|
|-- Videos/
|   |-- Tackle_match.mp4
|   `-- ... (your other .mp4 videos)
|
|-- yolov5n-football/
|   `-- best.pt
|
|-- Tackle_Snapshots/   <-- This will be created automatically
|
|-- brain.py
|-- main.py
|-- sort.py
|-- pose_landmarker_full.task
|-- requirements.txt
`-- venv/
How to Run
Once the setup is complete, run the main script from your activated terminal:
code
Bash
python main.py
The script will automatically find all videos in the Videos folder, process them one by one, and save any detected tackle snapshots into the Tackle_Snapshots folder. A window will display the real-time processing.
Configuration and Tuning
All key parameters can be easily adjusted in the "Tuning Dashboard" section at the top of main.py.
PROCESSING_WIDTH: The resolution width for AI processing. Higher values improve accuracy but decrease performance. 1280 is for quality, 640 is for balance.
YOLO_CONFIDENCE_THRESHOLD: How confident YOLOv5 must be to detect a player or ball.
TACKLE_DISTANCE_THRESHOLD: The base size of the "tackle bubble" in pixels.
SPRINTING_SPEED_PIXELS: The speed (in pixels/frame) a player must exceed to be considered sprinting.
MOVEMENT_VECTOR_ANGLE_TOLERANCE: The cone of intent (in degrees). A challenger must be moving towards a possessor within this angle.
Limitations & Future Work
Version 1.0 is a powerful proof-of-concept but has key limitations that represent opportunities for future development:
No Team/Referee Identification: The system cannot distinguish between teammates, opponents, or referees. This is the primary source of "false positive" tackle detections.
Limited Foul Types: The foul detection logic is currently limited to a specific "slide causes fall" scenario. It cannot detect other fouls like pushes, high kicks, or illegal blocks.
The next major step for this project is to move towards a supervised machine learning approach by building a custom classifier model that can learn the difference between a true tackle and other interactions directly from labeled image data.