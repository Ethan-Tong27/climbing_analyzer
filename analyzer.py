import cv2
import numpy as np
import time
import csv
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Any
from ultralytics import YOLO

class BetaBotAnalyzer:
    # --- YOLO Pose Keypoint Constants ---
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    
    # --- Drawing Constants ---
    COLOR_TRAIL = (0, 255, 255)      # Yellow
    COLOR_SKELETON = (0, 255, 0)     # Green
    COLOR_JOINTS = (0, 255, 255)     # Cyan
    COLOR_STABLE = (0, 255, 0)       # Green
    COLOR_DYNAMIC = (0, 0, 255)      # Red
    COLOR_TEXT = (0, 0, 0)           # Black

    def __init__(self, smoothing_factor: float = 0.15):
        self.model = YOLO("yolov8n-pose.pt")
        self.smoothing_factor = smoothing_factor
        self.max_trail_length = 500
        self.reset()

    def reset(self) -> None:
        """Resets the state for processing a new video."""
        self.prev_com: Optional[np.ndarray] = None
        self.velocity: float = 0.0
        self.com_trail: List[Tuple[int, int]] = []
        self.logs: List[List[Any]] = []

    def smooth_value(self, current: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
        """Applies an Exponential Moving Average (EMA) low-pass filter."""
        if previous is None:
            return current
        return (self.smoothing_factor * current) + (1 - self.smoothing_factor) * previous

    def calculate_angle(self, a: Any, b: Any, c: Any) -> float:
        """Calculates the interior angle between three coordinate points."""
        if a is None or b is None or c is None:
            return 0.0
            
        a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180.0 else 360 - angle

    def get_com(self, keypoints: List[Any]) -> Optional[np.ndarray]:
        """Calculates Center of Mass using weighted centroid of key body points."""
        if len(keypoints) < 13:
            return None
        
        target_indices = [self.L_SHOULDER, self.R_SHOULDER, self.L_HIP, self.R_HIP]
        valid_points = [
            keypoints[idx][:2] for idx in target_indices 
            if idx < len(keypoints) and keypoints[idx] is not None
        ]
        
        if not valid_points:
            return None
            
        return np.mean(valid_points, axis=0)

    def draw_motion_trail(self, frame: np.ndarray, com_position: Optional[np.ndarray]) -> np.ndarray:
        """Draws a fading motion trail representing the Center of Mass path."""
        if com_position is None:
            return frame
        
        self.com_trail.append(tuple(map(int, com_position)))
        
        if len(self.com_trail) > self.max_trail_length:
            self.com_trail.pop(0)
        
        for i in range(1, len(self.com_trail)):
            pt1 = self.com_trail[i - 1]
            pt2 = self.com_trail[i]
            alpha = i / len(self.com_trail)
            thickness = max(2, int(8 * alpha))
            cv2.line(frame, pt1, pt2, self.COLOR_TRAIL, thickness)
        
        return frame

    def draw_skeleton(self, frame: np.ndarray, keypoints: List[Any]) -> np.ndarray:
        """Draws skeleton keypoints and connections."""
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),                     # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),            # Arms
            (5, 11), (6, 12), (11, 12),                         # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)              # Legs
        ]
        
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                if keypoints[start] is not None and keypoints[end] is not None:
                    pt1 = tuple(map(int, keypoints[start][:2]))
                    pt2 = tuple(map(int, keypoints[end][:2]))
                    cv2.line(frame, pt1, pt2, self.COLOR_SKELETON, 2)
        
        for kpt in keypoints:
            if kpt is not None:
                pt = tuple(map(int, kpt[:2]))
                cv2.circle(frame, pt, 4, self.COLOR_JOINTS, -1)
        
        return frame

    def render_overlay(self, frame: np.ndarray, keypoints: List[Any], l_angle: float, r_angle: float) -> np.ndarray:
        """Applies all visual overlays to the frame."""
        frame = self.draw_skeleton(frame, keypoints)
        frame = self.draw_motion_trail(frame, self.prev_com)
        
        if self.prev_com is not None:
            com_px = (int(self.prev_com[0]), int(self.prev_com[1]))
            state_color = self.COLOR_STABLE if self.velocity < 10 else self.COLOR_DYNAMIC
            state_text = 'Stable' if self.velocity < 10 else 'Dynamic'
            
            cv2.circle(frame, com_px, 8, state_color, -1)
            cv2.putText(frame, f"Stability: {state_text}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 3)
            
            texts = [
                f"X: {self.prev_com[0]:.1f}",
                f"Y: {self.prev_com[1]:.1f}",
                f"Velocity: {self.velocity:.1f} px/f",
                f"L-Arm: {int(l_angle)}deg",
                f"R-Arm: {int(r_angle)}deg"
            ]
            
            for i, text in enumerate(texts):
                y_offset = 90 + (i * 30)
                cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
                
        return frame

    def analyze_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        """Processes a single frame: runs YOLO, computes kinematics, and draws overlays."""
        results = self.model(frame, verbose=False)
        
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return frame

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        raw_com = self.get_com(keypoints)
        
        l_angle, r_angle = 0.0, 0.0
        
        if raw_com is not None:
            if self.prev_com is not None:
                self.velocity = float(np.linalg.norm(raw_com - self.prev_com))
            
            self.prev_com = self.smooth_value(raw_com, self.prev_com)
            
            if len(keypoints) > self.L_WRIST:
                l_angle = self.calculate_angle(keypoints[self.L_SHOULDER], keypoints[self.L_ELBOW], keypoints[self.L_WRIST])
            if len(keypoints) > self.R_WRIST:
                r_angle = self.calculate_angle(keypoints[self.R_SHOULDER], keypoints[self.R_ELBOW], keypoints[self.R_WRIST])
                
            self.logs.append([frame_count, self.prev_com[0], self.prev_com[1], l_angle, r_angle, self.velocity])

        return self.render_overlay(frame, keypoints, l_angle, r_angle)

    def process_video(self, input_video_path: Path, output_video_path: Path) -> None:
        """Reads, processes, and exports a climbing video."""
        self.reset()
        cap = cv2.VideoCapture(str(input_video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open {input_video_path}.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_fps = min(120, max(fps, 30))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (width, height))
        
        window_name = 'Climbing Analysis'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)
        
        print(f"Processing '{input_video_path.name}'... Press 'q' to stop early.")
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    break 

                frame_count += 1
                processed_frame = self.analyze_frame(frame, frame_count)

                if processed_frame.shape[1] != width or processed_frame.shape[0] != height:
                    processed_frame = cv2.resize(processed_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                
                out.write(processed_frame)
                cv2.imshow(window_name, processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Early exit requested by user")
                    break
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            time.sleep(0.5) 
            
        self.save_logs(input_video_path.stem)
        self.fix_video_metadata(output_video_path)
        print(f"Finished! Output video saved as '{output_video_path}'")

    def save_logs(self, video_stem: str) -> None:
        """Exports the kinematic data to a CSV file."""
        if not self.logs:
            print("No data to save.")
            return
            
        csv_filename = f"{video_stem}_climbing_data.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'CoM_X', 'CoM_Y', 'L_Elbow', 'R_Elbow', 'Velocity'])
            writer.writerows(self.logs)
        print(f"Telemetry data saved to '{csv_filename}'")

    def fix_video_metadata(self, video_path: Path) -> None:
        """Uses ffmpeg to properly encode MP4 with correct aspect ratio metadata."""
        if not video_path.exists():
            return
        
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("FFmpeg not found - skipping metadata optimization.")
            return
        
        temp_path = video_path.with_name(f"{video_path.stem}_temp{video_path.suffix}")
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-c:v', 'libx264', '-crf', '18',
            '-c:a', 'aac', '-y', str(temp_path)
        ]
        
        try:
            print("Optimizing video metadata with ffmpeg...")
            subprocess.run(cmd, capture_output=True, check=True)
            video_path.unlink()  # Remove original
            temp_path.rename(video_path)  # Rename temp to original
            print("Video metadata optimized!")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e}")
            if temp_path.exists():
                temp_path.unlink()

if __name__ == "__main__":
    analyzer = BetaBotAnalyzer()
    video_folder = Path("video_data")
    video_folder.mkdir(exist_ok=True)
    
    video_extensions = {'.mov', '.mp4'}
    video_files = [
        f for f in video_folder.iterdir() 
        if f.is_file() and f.suffix.lower() in video_extensions and not f.name.startswith('analyzed_')
    ]
    
    if not video_files:
        print(f"No video files found. Add MOV or MP4 files to the '{video_folder}' directory.")
    else:
        print(f"Found {len(video_files)} video file(s) to analyze:\n")
        for i, video_file in enumerate(video_files, 1):
            print(f"{i}. {video_file.name}")
        
        for video_file in video_files:
            print(f"\n{'='*60}")
            print(f"Analyzing: {video_file.name}")
            print(f"{'='*60}")
            
            output_name = f"analyzed_{video_file.stem}.mp4"
            output_path = video_file.parent / output_name
            
            analyzer.process_video(video_file, output_path)