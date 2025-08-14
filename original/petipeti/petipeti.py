import cv2
import numpy as np
import mediapipe as mp
import json
import random

class Config:
    def __init__(self):
        self.max_pupil_offset = 150        # Max horizontal pupil movement offset
        self.eye_move_ratio = 0.75         # Ratio of eye movement to pupil movement
        self.smoothing = 1.0               # Steady-state tracking responsiveness
        self.initial_smoothing = 0.05      # Slower tracking when first detecting a face
        self.max_vertical_offset = 59      # Max vertical pupil movement offset
        self.vertical_smoothing = 1.0      # Vertical tracking smoothing
        self.pupil_max_move = 100          # Max pupil movement within the eye
        self.decay_speed = 0.5             # How quickly eyes return to center (when no face)

        # Pulsing motion parameters
        self.pulse_amplitude = 0.1        # Amplitude of the pulse (as a fraction of bg height)
        self.pulse_period = 4.0            # Time (in seconds) for one complete pulse cycle
        self.pulse_speed = 0.02            # Speed of the pulse animation
        self.pulse_smoothing = 0.1         # Smoothing factor for pulse motion

        self.face_detection_start_time = None
        self.face_detection_frames = 0
        self.typing_index = 0
        self.hello_text_options = [
            "Welcome to the Dive Inn!",
            "How can we drink you?",
            "Dive right Inn!",
            "Most effishient bar in town!",
            "You fishing for compliments?",
            "Ahoy, come sea what we have instore",
            "Come in and drop anchor!",
            "We got deeep drinks in here",
            "Water you waiting for, come in!",
            "We are offishelly OPEN!"]
        self.required_face_frames = 20
        self.frames_per_character = 1
        self.last_frame_time = 0
        self.fps = 60
        self.text_size = 1.0
        self.text_x_percent = 0.5
        self.text_y_percent = 0.1
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.center_text = True

        self.face_switch_threshold = 0.8        # Only switch faces when new face is X times more centered
        self.primary_face_lock_frames = 30      # Minimum frames to keep tracking the same face
        self.current_primary_lock = 0           # Counter for face lock
        self.min_face_confidence = 0.7      # Minimum detection confidence to consider
        self.face_center_bias = 0.5        # 0-1, how much to prefer centered faces (1.0 = strong preference)
        self.max_face_distance = 500       # Pixel distance beyond which faces are ignored
        self.face_size_weight = 0.3        # How much to prioritize larger faces (0-1)
        self.min_face_size = 0.1           # Relative minimum face size (0-1 of frame width)

        # Smoothing adjustments
        self.smoothing_fast = 0.2          # Used when making large movements
        self.smoothing_slow = 0.05         # Used for fine adjustments
        self.movement_threshold = 10       # Pixels to switch between fast/slow smoothing

config = Config()

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def overlay(bg, fg):
    """Overlays fg onto bg, handling alpha transparency."""
    # Check if foreground has alpha (4 channels)
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
        alpha_inv = 1.0 - alpha
        
        # Crop foreground if it's larger than background
        h, w = fg.shape[0], fg.shape[1]
        bh, bw = bg.shape[0], bg.shape[1]
        if h > bh or w > bw:
            fg = fg[:bh, :bw]
            alpha = alpha[:bh, :bw]
            alpha_inv = alpha_inv[:bh, :bw]
            
        for c in range(0, 3):  # Loop through BGR channels
            bg[:h, :w, c] = (alpha * fg[:, :, c] + alpha_inv * bg[:h, :w, c])
    else:
        # No alpha channel (simple overlay)
        h, w = fg.shape[0], fg.shape[1]
        bg[:h, :w] = fg  # Direct assignment if no transparency
        
    return bg
    
def safe_resize(img, target_width, target_height):
    """Resize image while maintaining aspect ratio and alpha channel."""
    if img is None:
        return None
    
    # Preserve alpha channel if it exists
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        rgb_resized = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(alpha, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2BGRA)
        resized[:, :, 3] = alpha_resized
        return resized
    else:
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

def main():
    # Load background
    bg = cv2.imread('assets/background.png', cv2.IMREAD_UNCHANGED)
    if bg is None:
        print("Error: Background image not found.")
        return
    
    bg_height, bg_width = bg.shape[:2]
    
    # Load other assets and resize them proportionally
    face_img = cv2.imread('assets/face.png', cv2.IMREAD_UNCHANGED)
    eyes = cv2.imread('assets/eyes.png', cv2.IMREAD_UNCHANGED)
    pupils = cv2.imread('assets/pupils.png', cv2.IMREAD_UNCHANGED)
    
    # Define scaling factors if desired (e.g., 0.5 for half size)
    scale_factor = 1.0
    
    # Resize assets proportionally
    face_img = safe_resize(face_img, int(bg_width * scale_factor), int(bg_height * scale_factor))
    eyes = safe_resize(eyes, int(bg_width * scale_factor), int(bg_height * scale_factor))
    pupils = safe_resize(pupils, int(bg_width * scale_factor), int(bg_height * scale_factor))
    
    if any(x is None for x in [bg, face_img, eyes, pupils]):
        print("Error: Asset loading failed.")
        return

    # Use dynamic max offsets based on background size
    config.max_pupil_offset = int(bg_width * 0.2)  # 20% of width
    config.max_vertical_offset = int(bg_height * 0.1)  # 10% of height
    config.pupil_max_move = int(bg_width * 0.15)  # 15% of width

    # Pre-compose static parts
    base_image = np.zeros_like(bg)
    base_image = overlay(base_image, bg)

    # Camera and windows
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Petipeti', cv2.WINDOW_NORMAL)

    # Tracking variables
    current_pupil_x = 0
    current_pupil_y = 0
    current_eye_x = 0
    current_eye_y = 0

    # Pulsing motion
    time_counter = 0.0
    current_pulse_offset = 0
    target_pulse_offset = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Update time counter for pulsing
        time_counter += config.pulse_speed
        target_pulse_offset = int(config.pulse_amplitude * bg.shape[0] * np.sin(2 * np.pi * time_counter / config.pulse_period))
        
        # Smooth the pulse offset
        current_pulse_offset += config.pulse_smoothing * (target_pulse_offset - current_pulse_offset)

        # Face detection
        primary_face_x, primary_face_y, face_count = 0, 0, 0
        best_face_score = 0
        
        # Convert frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results and results.detections:
            face_count = len(results.detections)
            ih, iw = frame.shape[:2]  # Get frame dimensions
            
            for detection in results.detections:
                if not hasattr(detection, 'location_data'):
                    continue
                    
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = max(10, int(bbox.width * iw))  # Ensure minimum width
                h = max(10, int(bbox.height * ih))  # Ensure minimum height
                
                # Calculate face center and score
                center_x = x + w // 2
                center_y = y + h // 2
                dist_to_center = np.sqrt((center_x - iw/2)**2 + (center_y - ih/2)**2)
                score = (1 - (dist_to_center / max(iw,ih))) * bbox.width
                
                # Keep tracking same face unless significantly better option exists
                if config.current_primary_lock > 0:
                    if score > best_face_score * config.face_switch_threshold:
                        best_face_score = score
                        primary_face_x, primary_face_y = center_x, center_y
                else:
                    if score > best_face_score:
                        best_face_score = score
                        primary_face_x, primary_face_y = center_x, center_y
                
            # Update face lock counter
            if face_count > 0:
                # Reset face lock if we have a good detection
                if best_face_score > 0.5:
                    config.current_primary_lock = config.primary_face_lock_frames
                else:
                    config.current_primary_lock = max(0, config.current_primary_lock - 1)
            else:
                config.current_primary_lock = 0
            
            # Visualization (optional)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (primary_face_x, primary_face_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Face timer logic
        if face_count > 0:
            config.face_detection_frames += 1
            if config.face_detection_frames == config.required_face_frames:
                config.hello_text = random.choice(config.hello_text_options)
                config.typing_index = 0
            elif config.face_detection_frames > config.required_face_frames:
                extra_frames = config.face_detection_frames - config.required_face_frames
                config.typing_index = min(extra_frames // config.frames_per_character, len(config.hello_text))
        else:
            config.face_detection_frames = 0
            config.typing_index = 0

        # Eye movement calculation
        if face_count > 0:
            # Reset face lock if we have a good detection
            if best_face_score > 0.5:
                config.current_primary_lock = config.primary_face_lock_frames
                
            target_x = np.interp(primary_face_x, [0, frame.shape[1]], [-config.max_pupil_offset, config.max_pupil_offset])
            target_y = np.interp(primary_face_y, [0, frame.shape[0]], [-config.max_vertical_offset, config.max_vertical_offset])
            
            # Dynamic smoothing based on face score (more stable for high-confidence detections)
            dyn_smoothing = config.smoothing * (0.5 + 0.5 * best_face_score)
            
            current_pupil_x += dyn_smoothing * (target_x - current_pupil_x)
            current_eye_x += dyn_smoothing * (target_x * config.eye_move_ratio - current_eye_x)
            current_pupil_y += dyn_smoothing * config.vertical_smoothing * (target_y - current_pupil_y)
            current_eye_y += dyn_smoothing * config.vertical_smoothing * (target_y * config.eye_move_ratio - current_eye_y)

        # --- Image Composition ---
        try:
            display_array = base_image.copy()

            # Face (with smoothed pulsing)
            moved_face = cv2.warpAffine(
                face_img,
                np.float32([[1, 0, 0], [0, 1, current_pulse_offset]]),
                (face_img.shape[1], face_img.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            display_array = overlay(display_array, moved_face)

            # Eyes (with tracking and pulsing)
            eyes_x_offset = int(current_eye_x)
            eyes_y_offset = int(current_eye_y) + current_pulse_offset
            moved_eyes = cv2.warpAffine(
                eyes,
                np.float32([[1, 0, eyes_x_offset], [0, 1, eyes_y_offset]]),
                (eyes.shape[1], eyes.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            display_array = overlay(display_array, moved_eyes)

            # Pupils (relative to eyes)
            pupil_x_offset = int(np.clip(current_pupil_x - current_eye_x, -config.pupil_max_move, config.pupil_max_move))
            pupil_y_offset = int(np.clip(current_pupil_y - current_eye_y, -config.pupil_max_move // 2, config.pupil_max_move // 2))
            final_pupil_x = eyes_x_offset + pupil_x_offset
            final_pupil_y = eyes_y_offset + pupil_y_offset
            
            moved_pupils = cv2.warpAffine(
                pupils,
                np.float32([[1, 0, final_pupil_x], [0, 1, final_pupil_y]]),
                (pupils.shape[1], pupils.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            display_array = overlay(display_array, moved_pupils)

            # Dynamic greeting text
            if config.typing_index > 0:
                partial_text = config.hello_text[:config.typing_index]
                (text_width, text_height), _ = cv2.getTextSize(partial_text, config.font_face, config.text_size, 2)
                text_x = (display_array.shape[1] - text_width) // 2
                text_y = int(display_array.shape[0] * config.text_y_percent)
                
                # Text with outline
                cv2.putText(display_array, partial_text, (text_x, text_y), config.font_face, 
                            config.text_size, (0, 0, 0), 4)
                cv2.putText(display_array, partial_text, (text_x, text_y), config.font_face, 
                            config.text_size, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error processing face detection: {e}")
            continue
                        
        cv2.imshow('Camera Feed', frame)
        cv2.imshow('Petipeti', cv2.cvtColor(display_array, cv2.COLOR_BGRA2BGR))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    face_detection.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to close...")
