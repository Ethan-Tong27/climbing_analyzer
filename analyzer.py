import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Any
from ultralytics import YOLO

class BetaBotAnalyzer:
    # --- YOLO Pose Keypoint Constants ---
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16
    
    # --- Drawing Constants ---
    COLOR_SKELETON = (0, 255, 0)     # Green
    COLOR_JOINTS = (0, 255, 255)     # Cyan
    COLOR_COM = (0, 0, 255)          # Red
    COLOR_TRAIL = (0, 255, 255)      # Yellow
    COLOR_STABLE = (0, 255, 0)       # Green for stable
    COLOR_UNSTABLE = (0, 0, 255)     # Red for unstable
    
    STABILITY_THRESHOLD = 5.0        # Pixels per frame
    TRAIL_LENGTH = 120               # Number of COM points to keep
    KEYPOINT_SMOOTH_FRAMES = 5       # Number of frames to smooth over

    def __init__(self):
        self.model = YOLO("yolov8m-pose.pt")
        self.prev_com: Optional[np.ndarray] = None
        self.com_trail: List[np.ndarray] = []
        self.com_displacement = 0.0
        self.keypoint_history: List[np.ndarray] = []

    def reset(self) -> None:
        """Resets the state for processing a new video."""
        self.prev_com: Optional[np.ndarray] = None
        self.com_trail: List[np.ndarray] = []
        self.com_displacement = 0.0
        self.keypoint_history: List[np.ndarray] = []

    def get_com(self, keypoints: List[Any]) -> Optional[np.ndarray]:
        """Calculates Center of Mass using weighted center of key body points."""
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

    def is_com_stable(self) -> bool:
        """Determines if COM is stable based on displacement from previous frame."""
        return self.com_displacement < self.STABILITY_THRESHOLD

    def smooth_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Applies smoothing to keypoints using a buffer of recent frames."""
        self.keypoint_history.append(keypoints.copy())
        
        # Keep only recent frames
        if len(self.keypoint_history) > self.KEYPOINT_SMOOTH_FRAMES:
            self.keypoint_history.pop(0)
        
        # Average keypoints over the buffer
        smoothed = np.mean(self.keypoint_history, axis=0)
        return smoothed

    def draw_skeleton(self, frame: np.ndarray, keypoints: List[Any]) -> np.ndarray:
        """Draws skeleton keypoints and connections"""
        connections = [
            (5, 6),           # Shoulders
            (5, 7), (7, 9),   # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12), # Shoulders to hips
            (11, 12),         # Hips
            (11, 13), (13, 15),  # Left leg (hip -> knee -> ankle)
            (12, 14), (14, 16)   # Right leg (hip -> knee -> ankle)
        ]
        
        # Draw connections
        for start, end in connections:
            if (start < len(keypoints) and end < len(keypoints) and
                keypoints[start] is not None and keypoints[end] is not None):
                # Verify confidence is above threshold
                start_conf = keypoints[start][2] if len(keypoints[start]) > 2 else 1.0
                end_conf = keypoints[end][2] if len(keypoints[end]) > 2 else 1.0
                
                if start_conf > 0.1 and end_conf > 0.1:
                    pt1 = tuple(map(int, keypoints[start][:2]))
                    pt2 = tuple(map(int, keypoints[end][:2]))
                    cv2.line(frame, pt1, pt2, self.COLOR_SKELETON, 2)
        
        # Draw joints
        for kpt in keypoints:
            if kpt is not None:
                conf = kpt[2] if len(kpt) > 2 else 1.0
                if conf > 0.1:
                    pt = tuple(map(int, kpt[:2]))
                    cv2.circle(frame, pt, 4, self.COLOR_JOINTS, -1)
        
        return frame

    def render_overlay(self, frame: np.ndarray, keypoints: List[Any]) -> np.ndarray:
        """Applies visual overlays to the frame."""
        frame = self.draw_skeleton(frame, keypoints)
        
        # Draw COM trail
        if len(self.com_trail) > 1:
            for i in range(len(self.com_trail) - 1):
                pt1 = tuple(map(int, self.com_trail[i]))
                pt2 = tuple(map(int, self.com_trail[i + 1]))
                # Fade the trail by varying alpha
                alpha = (i + 1) / len(self.com_trail)
                color_alpha = tuple(int(c * alpha) for c in self.COLOR_TRAIL)
                cv2.line(frame, pt1, pt2, color_alpha, 4)
        
        # Draw COM circle
        if self.prev_com is not None:
            com_px = (int(self.prev_com[0]), int(self.prev_com[1]))
            com_color = self.COLOR_STABLE if self.is_com_stable() else self.COLOR_UNSTABLE
            cv2.circle(frame, com_px, 6, com_color, -1)
            cv2.circle(frame, com_px, 8, com_color, 2)
        
        # Draw stability status text
        stability_text = "STABLE" if self.is_com_stable() else "UNSTABLE"
        stability_color = self.COLOR_STABLE if self.is_com_stable() else self.COLOR_UNSTABLE
        cv2.putText(frame, f"COM: {stability_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, stability_color, 2)
        cv2.putText(frame, f"Disp: {self.com_displacement:.1f}px", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
        return frame
    #Processes a single frame: runs YOLO, computes kinematics, and draws overlays.
    def analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame, verbose=False)
        
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return frame

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # Apply smoothing
        keypoints = self.smooth_keypoints(keypoints)
        
        raw_com = self.get_com(keypoints)
        
        if raw_com is not None:
            # Calculate displacement from previous COM
            if self.prev_com is not None:
                self.com_displacement = np.linalg.norm(raw_com - self.prev_com)
            else:
                self.com_displacement = 0.0
            
            self.prev_com = raw_com
            
            # Update trail
            self.com_trail.append(raw_com.copy())
            if len(self.com_trail) > self.TRAIL_LENGTH:
                self.com_trail.pop(0)

        return self.render_overlay(frame, keypoints)

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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        window_name = 'Climbing Analysis'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 540, 960)  # iPhone portrait resolution (9:16 aspect ratio)
        
        print(f"Processing '{input_video_path.name}'... Press 'q' to stop early.")
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    break 

                frame_count += 1
                processed_frame = self.analyze_frame(frame)

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
            
        print(f"Finished! Output video saved as '{output_video_path}'")

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