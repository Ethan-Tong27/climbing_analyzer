import cv2
import numpy as np
import time
import csv
import os
from pathlib import Path
from ultralytics import YOLO
import subprocess

class BetaBotAnalyzer:
    def __init__(self, smoothing_factor=0.15):
        # Use YOLOv8 pose model which is more reliable and easier to use
        self.model = YOLO("yolov8n-pose.pt")  # nano pose model
        
        # Smoothing (Exponential Moving Average)
        self.smoothing_factor = smoothing_factor
        self.prev_com = None
        self.velocity = 0
        
        # Motion trail: stores all COM positions
        self.com_trail = []
        self.max_trail_length = 500  # Keep last 500 frames worth of trail
        
        # Data Logging
        self.logs = []

    def smooth_value(self, current, previous):
        """Applies an Exponential Moving Average (EMA) low-pass filter."""
        if previous is None:
            return current
        return (self.smoothing_factor * current) + (1 - self.smoothing_factor) * previous

    def calculate_angle(self, a, b, c):
        """Calculates the interior angle between three coordinates."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180.0 else 360 - angle

    def get_com(self, keypoints):
        """Calculates Center of Mass using weighted centroid of key body points."""
        # YOLO keypoint indices: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
        if len(keypoints) < 13:
            return None
        
        valid_points = []
        for idx in [5, 6, 11, 12]:
            if idx < len(keypoints) and keypoints[idx] is not None:
                valid_points.append(keypoints[idx][:2])
        
        if not valid_points:
            return None
            
        x = np.mean([p[0] for p in valid_points])
        y = np.mean([p[1] for p in valid_points])
        return np.array([x, y])

    def get_bounding_box(self, keypoints):
        """Calculate bounding box from all keypoints."""
        valid_points = [p[:2] for p in keypoints if p is not None and len(p) >= 2]
        if not valid_points:
            return None
        
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]
        
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(1920, x_max + padding)  # Assume max width 1920
        y_max = min(1080, y_max + padding)  # Assume max height 1080
        
        return (x_min, y_min, x_max, y_max)

    def draw_motion_trail(self, frame, com_position, color=(0, 255, 255)):
        """Draw yellow motion trail on the frame."""
        if com_position is None:
            return frame
        
        # Add current position to trail
        self.com_trail.append(tuple(map(int, com_position)))
        
        # Keep trail length limited
        if len(self.com_trail) > self.max_trail_length:
            self.com_trail.pop(0)
        
        # Draw the trail
        if len(self.com_trail) > 1:
            for i in range(1, len(self.com_trail)):
                pt1 = self.com_trail[i - 1]
                pt2 = self.com_trail[i]
                # Fade effect: older points are dimmer
                alpha = i / len(self.com_trail)
                thickness = max(2, int(8 * alpha))  # Thicker trail (was 3, now 8)
                cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame

    def draw_skeleton(self, frame, keypoints):
        """Draw skeleton keypoints and connections without bounding box."""
        # YOLO pose keypoint connections (17 keypoints)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw connections (skeleton)
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                pt1 = tuple(map(int, keypoints[start][:2]))
                pt2 = tuple(map(int, keypoints[end][:2]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green skeleton
        
        # Draw keypoints as circles
        for kpt in keypoints:
            if kpt is not None:
                pt = tuple(map(int, kpt[:2]))
                cv2.circle(frame, pt, 4, (0, 255, 255), -1)  # Cyan joints
        
        return frame

    def process_video(self, input_video_path, output_video_path="analyzed_climb.mp4"):
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open {input_video_path}. Check if the file name is correct.")
            return

        # Setup VideoWriter to save the output
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video dimensions: {width}x{height} (aspect ratio: {width/height:.2f})")
        
        # Increase output FPS to 120 max (supports both vertical and horizontal)
        output_fps = min(120, max(fps, 30))
        
        # Use mp4v codec for MP4 format - properly handles vertical videos
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Failed to open VideoWriter for {output_video_path}")
            print(f"Attempted dimensions: {width}x{height} at {output_fps} FPS")
            return
        
        # Create resizable window
        window_name = 'Climbing Analysis'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)
        
        print("Processing video... Press 'q' to stop early.")
        frame_count = 0
        first_frame = True

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break # End of video

            frame_count += 1
            h, w, _ = frame.shape
            
            # Debug: Check frame dimensions on first frame
            if first_frame:
                print(f"First frame dimensions: {w}x{h}")
                first_frame = False
            
            # Run YOLOv8 pose detection
            results = self.model(frame, verbose=False)
            
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0].cpu().numpy()  # Get first person's keypoints
                
                # Kinematics: Center of Mass & Smoothing
                raw_com = self.get_com(keypoints)
                if raw_com is not None:
                    if self.prev_com is not None:
                        # Calculate Velocity (pixels per frame)
                        self.velocity = np.linalg.norm(raw_com - self.prev_com)
                    
                    self.prev_com = self.smooth_value(raw_com, self.prev_com)
                    com_px = (int(self.prev_com[0]), int(self.prev_com[1]))
                    color = (0, 255, 0) if self.velocity < 10 else (0, 0, 255)  # Green if stable, Red if dynamic

                    # 2. Joint Efficiency (Left & Right Elbows)
                    try:
                        l_angle = self.calculate_angle(keypoints[5][:2], keypoints[7][:2], keypoints[9][:2])
                        r_angle = self.calculate_angle(keypoints[6][:2], keypoints[8][:2], keypoints[10][:2])
                    except:
                        l_angle = 0
                        r_angle = 0

                    # 3. Visuals & Rendering - Draw everything once
                    frame = self.draw_skeleton(frame, keypoints)
                    
                    # Draw motion trail
                    frame = self.draw_motion_trail(frame, self.prev_com, color=(0, 255, 255))
                    
                    # Draw CoM indicator
                    cv2.circle(frame, com_px, 8, color, -1)
                    
                    cv2.putText(frame, f"Stability: {'Stable' if self.velocity < 10 else 'Dynamic'}", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                    
                    # Display Position & Velocity in Black
                    cv2.putText(frame, f"X: {self.prev_com[0]:.1f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(frame, f"Y: {self.prev_com[1]:.1f}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(frame, f"Velocity: {self.velocity:.1f} px/f", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Display Angles
                    cv2.putText(frame, f"L-Arm: {int(l_angle)}deg", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(frame, f"R-Arm: {int(r_angle)}deg", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # 4. Log Data for Engineering Analysis
                    self.logs.append([frame_count, self.prev_com[0], self.prev_com[1], l_angle, r_angle, self.velocity])

            # Write the modified frame to the output file
            # Ensure frame dimensions exactly match output dimensions
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            
            out.write(frame)
            
            # Display on screen (resizable window)
            cv2.imshow(window_name, frame)
            
            # Check for 'q' key press (check every frame for responsiveness)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Early exit requested by user")
                break

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)  # Give window time to fully close
        self.save_logs()
        self.fix_video_metadata(output_video_path)
        print(f"Finished! Output video saved as '{output_video_path}'")

    def save_logs(self):
        """Exports the kinematic data to a CSV file."""
        if not self.logs:
            print("No data to save.")
            return
            
        with open('climbing_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'CoM_X', 'CoM_Y', 'L_Elbow', 'R_Elbow', 'Velocity'])
            writer.writerows(self.logs)
        print("Telemetry data saved to 'climbing_data.csv'")

    def fix_video_metadata(self, video_path):
        """Use ffmpeg to properly encode MP4 with correct aspect ratio metadata."""
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("FFmpeg not found - video will work but may display incorrectly in some players")
            return
        
        # Create temporary output file
        temp_path = video_path.replace('.mp4', '_temp.mp4')
        
        try:
            # Re-encode with ffmpeg to ensure proper metadata
            cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264', '-crf', '18',  # Good quality
                '-c:a', 'aac',
                '-y',  # Overwrite without asking
                temp_path
            ]
            
            print("Optimizing video metadata with ffmpeg...")
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Replace original with optimized version
            os.remove(video_path)
            os.rename(temp_path, video_path)
            print("Video metadata optimized!")
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = BetaBotAnalyzer(smoothing_factor=0.15)
    
    # Define video folder
    video_folder = Path("video_data")
    
    # Create video_data folder if it doesn't exist
    video_folder.mkdir(exist_ok=True)
    
    # Find all MOV and MP4 files in the video_data folder (exclude already analyzed videos)
    video_extensions = ['.mov', '.mp4', '.MOV', '.MP4']
    video_files = [f for f in video_folder.iterdir() 
                   if f.is_file() and f.suffix in video_extensions and not f.name.startswith('analyzed_')]
    
    if not video_files:
        print(f"No video files found in '{video_folder}' folder.")
        print(f"Please add MOV or MP4 files to the '{video_folder}' folder.")
    else:
        print(f"Found {len(video_files)} video file(s) to analyze:\n")
        
        for i, video_file in enumerate(video_files, 1):
            print(f"{i}. {video_file.name}")
        
        print("\nProcessing all videos...")
        
        for video_file in video_files:
            print(f"\n{'='*60}")
            print(f"Analyzing: {video_file.name}")
            print(f"{'='*60}")
            
            # Create output filename based on input filename
            output_name = f"analyzed_{video_file.stem}.mp4"
            output_path = str(video_file.parent / output_name)
            
            # Process the video
            analyzer.process_video(str(video_file), output_path)
            
            print(f"✓ Saved to: {output_path}")
            
            # Reset logs and state for next video
            analyzer.logs = []
            analyzer.prev_com = None
            analyzer.com_trail = []
            analyzer.velocity = 0