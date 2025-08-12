# brain.py - DEFINITIVE "Good" Working Version (On-Demand, Vector Intent)

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

# ==================================================================================
class PlayerState:
    def __init__(self, track_id, max_history=10):
        self.id = track_id
        self.positions = deque(maxlen=max_history)
        self.velocity = (0, 0)
        self.speed = 0.0
        self.hip_y_history = deque(maxlen=max_history)

    def add_position(self, center_point, skeleton):
        self.positions.append(center_point)
        if len(self.positions) > 1:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            self.velocity = (dx, dy)
            self.speed = math.sqrt(dx**2 + dy**2)
        else:
            self.velocity, self.speed = (0, 0), 0.0
        
        if skeleton:
            self.hip_y_history.append((skeleton[23].y + skeleton[24].y) / 2)
        else:
            self.hip_y_history.append(self.hip_y_history[-1] if self.hip_y_history else 0.5)

# ==================================================================================
class RefereeBrain:
    def __init__(self, tackle_distance_threshold=160, slide_threshold=0.7, jump_fall_threshold=0.03,
                 sprinting_speed=15, walking_speed=4, vector_tolerance=65.0):
        # --- Tuning Parameters ---
        self.BASE_TACKLE_DISTANCE = tackle_distance_threshold
        self.SPRINTING_SPEED_THRESHOLD = sprinting_speed
        self.WALKING_SPEED_THRESHOLD = walking_speed
        self.MOVEMENT_VECTOR_TOLERANCE = vector_tolerance
        self.SLIDE_LOW_POSTURE_THRESHOLD = slide_threshold
        self.JUMP_FALL_VELOCITY_THRESHOLD = jump_fall_threshold
        
        # --- State Tracking & Frame Data ---
        self.player_states = {}
        self.id_to_skeleton_map = {}
        self.tracked_players = np.empty((0, 5))
        self.ball_detections = None
        
        # --- Analysis Results ---
        self.player_with_ball_id = None
        self.tackling_player_ids = []
        self.foul_details = None
        self.is_tackle_event_active = False

    def _update_player_states(self, tracked_players, id_to_skeleton_map):
        current_ids = {int(p[4]) for p in tracked_players}
        for player in tracked_players:
            track_id = int(player[4])
            center_point = (int((player[0] + player[2]) / 2), int((player[1] + player[3]) / 2))
            skeleton = id_to_skeleton_map.get(track_id)
            if track_id not in self.player_states:
                self.player_states[track_id] = PlayerState(track_id)
            self.player_states[track_id].add_position(center_point, skeleton)
        for track_id in list(self.player_states.keys()):
            if track_id not in current_ids:
                del self.player_states[track_id]
                
    def process_frame_for_tackle_event(self, tracked_players, ball_detections):
        self.tracked_players = tracked_players
        self.ball_detections = ball_detections
        self.id_to_skeleton_map = {} # Skeletons not available in this fast pass
        self._update_player_states(tracked_players, self.id_to_skeleton_map)
        
        self.foul_details = None
        self.tackling_player_ids = []
        self.is_tackle_event_active = False
        self.player_with_ball_id = None

        player_with_ball_center = self._find_player_with_ball()
        if player_with_ball_center:
            self._find_tacklers(player_with_ball_center)

        if self.tackling_player_ids:
            self.is_tackle_event_active = True
            return True
        return False

    def process_skeletons_for_foul(self, pose_results, frame_width, frame_height):
        if not self.is_tackle_event_active or not pose_results or not pose_results.pose_landmarks: return
        
        skeletons = pose_results.pose_landmarks
        self.id_to_skeleton_map = self._match_skeletons_to_players(skeletons, frame_width, frame_height)
        
        self._update_player_states(self.tracked_players, self.id_to_skeleton_map)

        for tackler_id in self.tackling_player_ids:
            tackler_skeleton = self.id_to_skeleton_map.get(tackler_id)
            victim_skeleton = self.id_to_skeleton_map.get(self.player_with_ball_id)
            if not tackler_skeleton or not victim_skeleton: continue

            if self._is_player_sliding(tackler_skeleton):
                reaction = self._did_player_jump_or_trip(self.player_with_ball_id, victim_skeleton)
                if reaction:
                    self.foul_details = {"type": "Slide Foul", "tackler": tackler_id, "victim": self.player_with_ball_id, "reaction": reaction}
                    return

    def _get_vector_angle(self, v1, v2):
        v1_x, v1_y = v1
        v2_x, v2_y = v2
        dot_product = v1_x * v2_x + v1_y * v2_y
        magnitude_v1 = math.sqrt(v1_x**2 + v1_y**2)
        magnitude_v2 = math.sqrt(v2_x**2 + v2_y**2)
        if magnitude_v1 == 0 or magnitude_v2 == 0: return 180.0
        cosine_angle = max(-1.0, min(1.0, dot_product / (magnitude_v1 * magnitude_v2)))
        return math.degrees(math.acos(cosine_angle))

    def _find_tacklers(self, player_with_ball_center):
        possessor_state = self.player_states.get(self.player_with_ball_id)
        if not possessor_state: return

        current_tackle_threshold = self.BASE_TACKLE_DISTANCE
        if possessor_state.speed > self.SPRINTING_SPEED_THRESHOLD:
            current_tackle_threshold *= 1.20
        elif possessor_state.speed < self.WALKING_SPEED_THRESHOLD:
            current_tackle_threshold *= 0.85

        for player in self.tracked_players:
            challenger_id = int(player[4])
            if challenger_id == self.player_with_ball_id: continue
            challenger_state = self.player_states.get(challenger_id)
            if not challenger_state: continue
            
            challenger_center = (int((player[0] + player[2]) / 2), int((player[1] + player[3]) / 2))
            distance = math.sqrt((player_with_ball_center[0] - challenger_center[0])**2 + (player_with_ball_center[1] - challenger_center[1])**2)
            if distance > current_tackle_threshold: continue

            to_possessor_vector = (player_with_ball_center[0] - challenger_center[0], player_with_ball_center[1] - challenger_center[1])
            challenger_velocity_vector = challenger_state.velocity
            angle = self._get_vector_angle(challenger_velocity_vector, to_possessor_vector)
            if angle < self.MOVEMENT_VECTOR_TOLERANCE:
                self.tackling_player_ids.append(challenger_id)

    def _find_player_with_ball(self):
        if self.ball_detections.empty: return None
        ball_row = self.ball_detections.iloc[0]
        ball_box = (int(ball_row['xmin']), int(ball_row['ymin']), int(ball_row['xmax']), int(ball_row['ymax']))
        for player in self.tracked_players:
            player_box = (int(player[0]), int(player[1]), int(player[2]), int(player[3]))
            track_id = int(player[4])
            if self._do_boxes_overlap(player_box, ball_box):
                self.player_with_ball_id = track_id
                return (int((player_box[0] + player_box[2]) / 2), int((player_box[1] + player_box[3]) / 2))
        return None

    def _is_player_sliding(self, skeleton):
        left_hip_y, right_hip_y = skeleton[23].y, skeleton[24].y
        return (left_hip_y > self.SLIDE_LOW_POSTURE_THRESHOLD) or (right_hip_y > self.SLIDE_LOW_POSTURE_THRESHOLD)

    def _did_player_jump_or_trip(self, track_id, skeleton):
        state = self.player_states.get(track_id)
        if not state or len(state.hip_y_history) < 2: return None
        current_hip_y = state.hip_y_history[-1]
        last_hip_y = state.hip_y_history[-2]
        if current_hip_y > last_hip_y + self.JUMP_FALL_VELOCITY_THRESHOLD: return "Trip/Fall"
        return None

    def _match_skeletons_to_players(self, skeletons, frame_width, frame_height):
        if not skeletons or len(self.tracked_players) == 0: return {}
        skeleton_boxes = [self._get_skeleton_bounding_box(sk, frame_width, frame_height) for sk in skeletons]
        cost_matrix = np.zeros((len(self.tracked_players), len(skeletons)))
        for i, player in enumerate(self.tracked_players):
            player_box = (int(player[0]), int(player[1]), int(player[2]), int(player[3]))
            for j, sk_box in enumerate(skeleton_boxes):
                cost_matrix[i, j] = 1 - self._iou(player_box, sk_box)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        id_to_skeleton_map = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.7:
                track_id = int(self.tracked_players[r, 4])
                id_to_skeleton_map[track_id] = skeletons[c]
        return id_to_skeleton_map

    def _get_skeleton_bounding_box(self, skeleton, frame_width, frame_height):
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        for landmark in skeleton:
            min_x, min_y = min(min_x, landmark.x), min(min_y, landmark.y)
            max_x, max_y = max(max_x, landmark.x), max(max_y, landmark.y)
        return (int(min_x * frame_width), int(min_y * frame_height), int(max_x * frame_width), int(max_y * frame_height))

    def _iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0
        
    def _do_boxes_overlap(self, box1, box2):
        if not box1 or not box2: return False
        b1_xmin, b1_ymin, b1_xmax, b1_ymax = box1
        b2_xmin, b2_ymin, b2_xmax, b2_ymax = box2
        
        return not (b1_xmax < b2_xmin or b2_xmax < b1_xmin or b1_ymax < b2_ymin or b2_ymax < b1_ymin)