# main.py - COMPLETE version for the "Good" working brain

import cv2
import mediapipe as mp
import numpy as np
import time
import torch
import os
import glob

from brain import RefereeBrain
from sort import Sort
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================================================================================
YOLO_MODEL_PATH = 'yolov5n-football\\best.pt'
PLAYER_CLASS_NAME = 'player'
BALL_CLASS_NAME = 'football'
PROCESSING_WIDTH = 1280
POSE_MODEL_PATH = 'pose_landmarker_full.task'
YOLO_CONFIDENCE_THRESHOLD = 0.5
POSE_NUM_POSES = 10

# --- TUNING FOR THE "GOOD" BRAIN LOGIC ---
TACKLE_DISTANCE_THRESHOLD = 200
SPRINTING_SPEED_PIXELS = 10
WALKING_SPEED_PIXELS = 3
MOVEMENT_VECTOR_ANGLE_TOLERANCE = 65.0
SLIDE_LOW_POSTURE_THRESHOLD = 0.7
JUMP_FALL_VELOCITY_THRESHOLD = 0.03

# --- BATCH PROCESSING ---
VIDEO_SOURCE_FOLDER = 'Videos'
SNAPSHOT_OUTPUT_FOLDER = 'Tackle_Snapshots'
# ==================================================================================

try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=True)
    yolo_model.conf = YOLO_CONFIDENCE_THRESHOLD
    pose_options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH), running_mode=vision.RunningMode.IMAGE, num_poses=POSE_NUM_POSES)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    print(f"Error initializing models: {e}")
    exit()

if not os.path.exists(SNAPSHOT_OUTPUT_FOLDER): os.makedirs(SNAPSHOT_OUTPUT_FOLDER)
video_files = glob.glob(os.path.join(VIDEO_SOURCE_FOLDER, '*.mp4'))
print(f"Found {len(video_files)} videos to process: {video_files}")

for video_path in video_files:
    print(f"\n--- Processing video: {video_path} ---")
    
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    referee_brain = RefereeBrain(
        tackle_distance_threshold=TACKLE_DISTANCE_THRESHOLD,
        slide_threshold=SLIDE_LOW_POSTURE_THRESHOLD,
        jump_fall_threshold=JUMP_FALL_VELOCITY_THRESHOLD,
        sprinting_speed=SPRINTING_SPEED_PIXELS,
        walking_speed=WALKING_SPEED_PIXELS,
        vector_tolerance=MOVEMENT_VECTOR_ANGLE_TOLERANCE
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): continue

    frame_number = 0
    was_tackle_in_previous_frame = False
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None: break
        frame_number += 1
        
        orig_h, orig_w, _ = frame.shape
        aspect_ratio = orig_h / orig_w
        processing_height = int(PROCESSING_WIDTH * aspect_ratio)
        processing_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height))

        yolo_results = yolo_model(processing_frame)
        detections = yolo_results.pandas().xyxy[0]
        player_detections = detections[detections['name'] == PLAYER_CLASS_NAME]
        ball_detections = detections[detections['name'] == BALL_CLASS_NAME]
        detections_for_sort = player_detections[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].to_numpy()
        tracked_players = tracker.update(detections_for_sort)

        skeletons_needed = referee_brain.process_frame_for_tackle_event(tracked_players, ball_detections)

        pose_results = None
        if skeletons_needed:
            pose_image_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=pose_image_rgb)
            pose_results = pose_landmarker.detect(mp_image)
            frame_h, frame_w, _ = processing_frame.shape
            referee_brain.process_skeletons_for_foul(pose_results, frame_w, frame_h)
            
        # DRAWING
        if pose_results and pose_results.pose_landmarks:
            for person_landmarks in pose_results.pose_landmarks:
                proto_landmarks = landmark_pb2.NormalizedLandmarkList()
                for landmark in person_landmarks:
                    proto_landmarks.landmark.add(x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility, presence=landmark.presence)
                mp_drawing.draw_landmarks(processing_frame, proto_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(200,200,200), thickness=1, circle_radius=1))
        
        for player in tracked_players:
            xmin, ymin, xmax, ymax, track_id = int(player[0]), int(player[1]), int(player[2]), int(player[3]), int(player[4])
            box_color, label = (255, 0, 255), f"Player [{track_id}]"
            if track_id == referee_brain.player_with_ball_id:
                box_color, label = (0, 255, 0), f"IN POSSESSION [{track_id}]"
            elif track_id in referee_brain.tackling_player_ids:
                box_color, label = (0, 255, 255), f"TACKLER [{track_id}]"
            cv2.rectangle(processing_frame, (xmin, ymin), (xmax, ymax), box_color, 2)
            cv2.putText(processing_frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        if referee_brain.foul_details:
            foul = referee_brain.foul_details
            foul_text = f"FOUL: {foul['reaction']} by #{foul['tackler']}"
            cv2.putText(processing_frame, foul_text, (50, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
            
        if not ball_detections.empty:
            ball_row = ball_detections.iloc[0]
            cv2.rectangle(processing_frame, (int(ball_row['xmin']), int(ball_row['ymin'])), (int(ball_row['xmax']), int(ball_row['ymax'])), (0, 165, 255), 2)
            
        # SNAPSHOTS
        if referee_brain.is_tackle_event_active and not was_tackle_in_previous_frame:
            victim_id = referee_brain.player_with_ball_id
            tackler_ids = '-'.join(map(str, referee_brain.tackling_player_ids))
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            snapshot_filename = f"{video_name}_frame{frame_number}_victim{victim_id}_tacklers{tackler_ids}.jpg"
            snapshot_path = os.path.join(SNAPSHOT_OUTPUT_FOLDER, snapshot_filename)
            cv2.imwrite(snapshot_path, processing_frame)
            print(f"  >> Saved tackle snapshot: {snapshot_path}")

        was_tackle_in_previous_frame = referee_brain.is_tackle_event_active

        # DISPLAY
        DISPLAY_WIDTH = 600
        display_height = int(DISPLAY_WIDTH * (orig_h / orig_w))
        display_frame = cv2.resize(processing_frame, (DISPLAY_WIDTH, display_height))
        cv2.imshow(f'Processing: {os.path.basename(video_path)}', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    if 'display_frame' in locals() and cv2.getWindowProperty(f'Processing: {os.path.basename(video_path)}', 0) >= 0 and cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("\n--- Batch processing complete! ---")