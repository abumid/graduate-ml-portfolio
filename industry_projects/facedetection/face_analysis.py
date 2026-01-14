import cv2
import numpy as np
import onnxruntime
import os
from scipy.spatial.distance import cosine
from time import time
from collections import OrderedDict

from ..model_zoo import model_zoo
from .common import Face
from eyelibuz.eye_openness import EyeOpennessDetector
from facelibuz.utils.sort_tracker import SORT
from facelibuz.utils.trackableobject import TrackableObject
onnxruntime.set_default_logger_severity(3)

# Timer tracking for persons
PERSON_TIMERS = {}  # {person_name: {'last_seen': timestamp, 'first_seen': timestamp}}
RESET_THRESHOLD = 10  # seconds - if not seen for this long, reset counter

def update_person_timer(person_name, current_time, is_timer_paused=False):
    """Update the timer for a person and return the seconds they've been visible"""
    global PERSON_TIMERS, RESET_THRESHOLD
    
    if person_name == "unknown" or not person_name:
        return 0
    
    if person_name in PERSON_TIMERS:
        # Check if person was absent for more than RESET_THRESHOLD seconds
        time_since_last_seen = current_time - PERSON_TIMERS[person_name]['last_seen']
        
        if time_since_last_seen > RESET_THRESHOLD:
            # Reset the timer - person was away too long
            PERSON_TIMERS[person_name] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'paused_at': None,
                'accumulated_time': 0
            }
            return 0
        else:
            # Update last seen time
            PERSON_TIMERS[person_name]['last_seen'] = current_time
            
            # Handle timer pause/resume
            if is_timer_paused:
                if PERSON_TIMERS[person_name].get('paused_at') is None:
                    # Just paused, save the accumulated time
                    if 'accumulated_time' not in PERSON_TIMERS[person_name]:
                        PERSON_TIMERS[person_name]['accumulated_time'] = int(current_time - PERSON_TIMERS[person_name]['first_seen'])
                    PERSON_TIMERS[person_name]['paused_at'] = current_time
                # Return accumulated time (frozen while paused)
                return PERSON_TIMERS[person_name].get('accumulated_time', 0)
            else:
                if PERSON_TIMERS[person_name].get('paused_at') is not None:
                    # Just resumed, adjust first_seen to account for paused duration
                    paused_duration = current_time - PERSON_TIMERS[person_name]['paused_at']
                    PERSON_TIMERS[person_name]['first_seen'] += paused_duration
                    PERSON_TIMERS[person_name]['paused_at'] = None
                
                # Return current duration
                return int(current_time - PERSON_TIMERS[person_name]['first_seen'])
    else:
        # New person - initialize timer
        PERSON_TIMERS[person_name] = {
            'first_seen': current_time,
            'last_seen': current_time,
            'paused_at': None,
            'accumulated_time': 0
        }
        return 0


class FaceAnalysis:

    def __init__(self, known_people=None, tracking=False):
        self.det_model = model_zoo.get_model('models/l/det_10g.onnx')
        self.rec_model = model_zoo.get_model('models/l/adaface.onnx')
        self.eye_openness_detector = EyeOpennessDetector(ear_threshold=0.12, max_num_faces=5)
        self.genderage_model = model_zoo.get_model('models/l/genderage.onnx')

        self.genders = ["Female", "Male", "None"]

        self.known_people=known_people

        self.tracker = None

        self.seen_people = {}
        self.record_people = {}

        print("Initialized face analysis")
        print("Tracking: ", tracking)
        if tracking:
            print("Tracking enabled")
            self.tracker = SORT(max_lost=30, iou_threshold=0.3)
            self.trackableObjects = OrderedDict()


    def prepare(
        self,
        ctx_id,
        det_thresh=0.7,
        rec_thresh=0.4,
        det_size=(640, 640),
    ):
        self.det_thresh = det_thresh
        self.rec_thresh = rec_thresh
        self.det_size = det_size

        self.det_model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
        self.rec_model.prepare(ctx_id)
        self.genderage_model.prepare(ctx_id)

    def find_face(self, embedding):
        best_matches = []

        for person in self.known_people:
            score = 1 - cosine(person['embedding'], embedding)

            if(score > self.rec_thresh):
                known = {}
                known['name']=person['name']
                known['score']= round(score, 2)
                known["pinfl"]=person["pinfl"]
                known['image_path'] = person['image_path']
                best_matches.append(known)

        if len(best_matches) == 0:
            return None

        best_matches.sort(key=lambda x: -x['score'])

        return best_matches[0]
    
    def match_eye_openness(self, eye_opennesses, bbox):
        if eye_opennesses is None or 'faces_data' not in eye_opennesses:
            return None
        
        x1, y1, x2, y2 = bbox
        matched_data = None
        max_iou = 0
        
        ious = []
        for face_data in eye_opennesses['faces_data']:
            ex1, ey1, ex2, ey2 = face_data['bbox']
            
            # Calculate intersection
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)
            
            if ix1 < ix2 and iy1 < iy2:  # There is an intersection
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = (x2 - x1) * (y2 - y1)
                bbox_area = (ex2 - ex1) * (ey2 - ey1)
                iou = inter_area / float(box_area + bbox_area - inter_area)
                
                ious.append((iou, face_data))
        
        if ious:
            max_iou_tuple = max(ious, key=lambda x: x[0])
            max_iou = max_iou_tuple[0]
            matched_data = max_iou_tuple[1] if max_iou > 0.4 else None
        
        return matched_data

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(
            img,
            max_num=max_num,
            metric='default',
        )
        eye_opennesses = self.eye_openness_detector.detect_eye_openness(img)
        faces = []
        rects = []
        kps_list = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None

            if kpss is not None:
                kps = kpss[i]

            if self.tracker is None:
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                self.rec_model.get(img, face)
                faces.append(face)
            else:
                landmarks = np.array(kps)
                landmarks = np.transpose(landmarks).reshape(10, -1)
                landmarks = np.transpose(landmarks)[0]

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                kps = landmarks.astype('int')
                rects.append([x1, y1, x2, y2])
                kps_list.append(kps)

        if self.tracker:
            objects = self.tracker.update(np.array(rects), np.array(kps_list), np.ones(len(rects)))

            for _, track_obj in self.trackableObjects.items():
                track_obj.live = False

            for obj in objects:
                objectID = obj[1]
                bbox = obj[2:6]
                kps = obj[6]

                eye_openness_data = self.match_eye_openness(eye_opennesses, bbox)
                
                facial5points = [[kps[j], kps[j + 5]] for j in range(5)]

                track_obj = self.trackableObjects.get(objectID, None)

                if track_obj is None:
                    track_obj = TrackableObject(objectID, obj)

                face = Face(bbox=np.array(bbox), kps=np.array(facial5points), det_score=1)
                track_obj.bbox = np.array(bbox)
                track_obj.kps = np.array(facial5points)
                track_obj.live = True
                track_obj.lost_count=0

                if not track_obj.recognized:
                    self.rec_model.get(img, face)

                    person = self.find_face(face.embedding)

                    if person:
                        if person['score']>track_obj.score:
                            track_obj.name=person['name']
                            track_obj.pinfl = person["pinfl"]
                            track_obj.score = person['score']
                            track_obj.image_path = person['image_path']
                            if person['name'] in self.record_people:
                                cur = time()
                                if cur - self.record_people[person['name']] > 15:
                                    self.seen_people[person['name']] += 1
                                self.record_people[person['name']] = cur
                            else:
                                self.record_people[person['name']] = time()
                                self.seen_people[person['name']] = self.seen_people.get(person['name'], 0) + 1
                            

                    if track_obj.score > 0:
                        track_obj.recognized = True

                self.genderage_model.get(img, face)
                track_obj.age = face['age']
                track_obj.gender = face['gender']
                track_obj.left_eye_open = "unknown"
                track_obj.right_eye_open = "unknown"
                
                # Track cumulative eye closure duration
                current_time = time()
                if eye_openness_data:
                    track_obj.left_eye_open = "Open" if eye_openness_data['left_eye_open'] else "Closed"
                    track_obj.right_eye_open = "Open" if eye_openness_data['right_eye_open'] else "Closed"
                    
                    # Check if ANY eye is closed (left, right, or both)
                    any_eye_closed = not eye_openness_data['left_eye_open'] or not eye_openness_data['right_eye_open']
                    
                    # Initialize last check time if needed
                    if track_obj.last_eye_check_time is None:
                        track_obj.last_eye_check_time = current_time
                    
                    # Calculate time elapsed since last check
                    time_elapsed = current_time - track_obj.last_eye_check_time
                    
                    if any_eye_closed:
                        # At least one eye is closed - accumulate the time
                        track_obj.cumulative_closed_time += time_elapsed
                        
                        # If cumulative closed time reaches 3+ seconds, pause the timer
                        if track_obj.cumulative_closed_time >= 3.0:
                            track_obj.timer_paused = True
                    else:
                        # Both eyes are open - resume timer and reset cumulative time
                        if track_obj.timer_paused:
                            track_obj.timer_paused = False
                        # Reset cumulative time so next closure starts fresh
                        track_obj.cumulative_closed_time = 0
                    
                    # Update last check time
                    track_obj.last_eye_check_time = current_time

                self.trackableObjects[objectID] = track_obj

        # Remove dead objects from trackableObjects (objects that are no longer detected)
        dead_objects = []
        for objectID, track_obj in self.trackableObjects.items():
            if not track_obj.live:
                dead_objects.append(objectID)
        
        for objectID in dead_objects:
            del self.trackableObjects[objectID]

        return faces

    def draw_on(self, img, faces):
        dimg = img.copy()
        for _, track_obj in self.trackableObjects.items():

            if not track_obj.live:
                continue
            person_info = []
            box = track_obj.bbox.astype(np.int32)

            if track_obj.name == 'unknown':
                color = (0, 255, 255)
                person_info = [
                    f'R: {track_obj.left_eye_open} L: {track_obj.right_eye_open}',
                    track_obj.name,
                ]

            elif 'red' in track_obj.name:
                color = (0, 0, 255)
                seen = self.seen_people[track_obj.name]
                name = track_obj.name[:track_obj.name.index('red') - 1]
                person_info = [
                    f'R: {track_obj.left_eye_open} L: {track_obj.right_eye_open}',
                    name,
                    # f'{track_obj.age}',
                    # f'{self.genders[track_obj.gender]}',
                    # f'seen: {seen}',
                ]
                # Add timer for recognized persons
                if hasattr(track_obj, 'recognized') and track_obj.recognized:
                    current_time = time()
                    is_paused = getattr(track_obj, 'timer_paused', False)
                    seconds_visible = update_person_timer(name, current_time, is_paused)
                    timer_text = f"Time: {seconds_visible}s"
                    if is_paused:
                        timer_text += " (PAUSED)"
                    person_info.append(timer_text)
            else:
                color = (0, 255, 0)
                seen = self.seen_people[track_obj.name]
                person_info = [
                    f'R: {track_obj.left_eye_open} L: {track_obj.right_eye_open}',
                    track_obj.name,
                    # f'{track_obj.age}',
                    # f'{self.genders[track_obj.gender]}',
                    # f'seen: {seen}',
                ]
                # Add timer for recognized persons
                if hasattr(track_obj, 'recognized') and track_obj.recognized:
                    current_time = time()
                    is_paused = getattr(track_obj, 'timer_paused', False)
                    seconds_visible = update_person_timer(track_obj.name, current_time, is_paused)
                    timer_text = f"Time: {seconds_visible}s"
                    if is_paused:
                        timer_text += " (PAUSED)"
                    person_info.append(timer_text)

            # bbox_width = box[2] - box[0]
            # bbox_height = box[3] - box[1]
            # bbox_size = (bbox_width + bbox_height) / 2

            # scale_factor = bbox_size / 200.0

            # font_scale = max(0.5, scale_factor)
            # thickness = max(1, int(scale_factor))

            color = (0, 255, 0) if track_obj.left_eye_open == "Open" and track_obj.right_eye_open == "Open" else (0, 0, 255)

            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)

            # y0, dy = box[1] - 50, int(23 * scale_factor)

            # for i, line in enumerate(person_info):
            #     y = y0 + i * dy

            #     cv2.putText(
            #         dimg,
            #         line,
            #         (box[0], y),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         font_scale,
            #         color,
            #         thickness,
            #     )
            # cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)

            y0, dy = box[1]-65, 30

            for i, line in enumerate(person_info):
                y = y0 + i*dy

                cv2.putText(
                    dimg,
                    line,
                    (box[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    4,
                )

        return dimg

    def draw_single_face(self, img, track_obj, padding=10):
        
        dimg = img.copy()
        
        # if not track_obj.live:
        #     return dimg

        box = track_obj.bbox.astype(np.int32)
        x1, y1, x2, y2 = box

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        passport_img = cv2.imread(track_obj.image_path) if track_obj.recognized else cv2.imread('./known_people/unk.jpg')
        cropped_img = dimg[y1:y2, x1:x2]
        age = track_obj.age
        person_info = [
            f'{track_obj.pinfl}',
            track_obj.name,
            # f'age: {age-5}-{age+5}',
            # f'score: {track_obj.score}',
        ]
        # person_info = [
        #     f'ID: {track_obj.objectID}',
        #     track_obj.name,
        #     f'age: {age-5}-{age+5}',
        #     # f'score: {track_obj.score}',
        # ]

        y0, dy = 60, 60 
        text_img = np.ones((200, 580, 3), dtype=np.uint8) * 255

        for i, line in enumerate(person_info):
            y = y0 + i * dy
            cv2.putText(
                text_img,
                line,
                (100, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (74, 155, 79),
                4,
            )
        return text_img, cropped_img, passport_img
    
    async def _add_todb(self, img, person_name, pinfl, image_path):

        print("LEN BEFORE: ", len(self.known_people))
        bboxes, kpss = self.det_model.detect(
            img,
            max_num=0,
            metric='default',
        )
        faces = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None

            if kpss is not None:
                kps = kpss[i]
                

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            self.rec_model.get(img, face)
            faces.append(face)
        if len(faces) == 0:
            return

        person = {}
        person["name"] = person_name
        person["pinfl"] = pinfl
        person["embedding"] = faces[0].embedding
        person["image_path"] = image_path
        self.known_people.append(person)

        print("LEN AFTER: ", len(self.known_people))