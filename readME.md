🚀 How to Use
Setup Folder Structure: Place the script in your working directory. When run, the script will automatically look for (or create) a folder named video_data.

Plaintext
📁 BetaBot-Project/
├── 📄 betabot.py
└── 📁 video_data/
    ├── 🧗‍♂️ my_climb.mp4
    └── 🧗‍♂️ project_attempt.mov
Add Videos:
Drop your raw climbing videos (.mp4 or .mov) into the video_data folder.

Run the Analyzer:
Execute the script from your terminal:

Bash
python betabot.py
Controls & Output:

The analysis will display in a pop-up window. You can click and drag the corners to resize the window to fit your screen perfectly.

Press q while the video window is focused to stop the analysis early.

Analyzed videos are saved back into the video_data folder with the prefix analyzed_ (e.g., analyzed_my_climb.mp4).

📊 Data Output (climbing_data.csv)
The exported CSV file contains the following columns for graphing or further analysis:

Frame: The current video frame number.

CoM_X / CoM_Y: The calculated Center of Mass coordinates.

L_Elbow / R_Elbow: The interior angle of the elbows (useful for analyzing "straight arm" efficiency).

Velocity: The speed of the CoM movement between frames (measured in pixels/frame).
"""