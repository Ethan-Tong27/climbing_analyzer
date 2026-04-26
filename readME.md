# BetaBot Analyzer 🧗‍♂️🤖

BetaBot Analyzer is a lightweight computer vision tool that uses YOLOv8 Pose estimation to analyze rock climbing biomechanics. It automatically processes climbing videos to track the climber's center of mass (CoM), evaluate joint efficiency, and visualize motion trails in real-time. 

## 🚀 How to Use

### 1. Setup Folder Structure
Place the script in your working directory. When run, the script will automatically look for (or create) a folder named `video_data`.

```text
BetaBot-Project/
├── betabot.py
└── video_data/
    ├── my_climb.mp4
    └── project_attempt.mov
2. Run the Analyzer
Execute the script from your terminal:

Bash
python betabot.py
3. Controls & Output
Resize Window: The analysis displays in a pop-up window. You can click and drag the corners to fit your screen.

Stop Early: Press q while the video window is focused to stop.

Saved Videos: Analyzed videos are saved in the video_data folder with the prefix analyzed_.

📊 Data Output (climbing_data.csv)
The exported CSV file contains the following columns for analysis:

Frame: The current video frame number.

CoM_X / CoM_Y: The calculated Center of Mass coordinates.

L_Elbow / R_Elbow: The interior angle of the elbows (ideal for analyzing "straight arm" technique).

Velocity: The speed of the CoM movement (measured in pixels/frame).