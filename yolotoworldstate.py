import cv2
import time
import math
from ultralytics import YOLO
from collections import deque
from ultralytics.utils.metrics import bbox_iou

IOU_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 100
MOVEMENT_THRESHOLD = 8
MAX_EVENTS = 20
DISAPPEARANCE_THRESHOLD = 15

DIRECTION_MAPPING = {
    -4: "left", 
    -3: "bottom left", 
    -2: "below", 
    -1: "bottom right",
    0: "right",
    1: "upper right", 
    2: "above", 
    3: "upper left", 
    4: "left",
}

# Calculate the Intersection over Union score
#iou_score = box_iou(ground_truth, predicted)

#print(f"IoU Score: {iou_score.item():.4f}")
# Output: IoU Score: 0.6806

class WorldState:
    """
    Attributes:
        objects: Dict[int, WorldObject]: A dictionary of objects keyed by tracking id from YOLO.
        relations: List[Relation]: A list of spatial relations between pairs of objects
        events: Deque[str]: Last MAX_EVENTS number of events
        version: int: How many updates to the WorldState
        frame_index: int: How many frames have been processed
    """
    def __init__(self, max_events=MAX_EVENTS):
        self.objects = {}
        self.relations = []
        self.events = deque(maxlen=max_events)

        self.version = 0
        self.frame_index = 0

    def update_from_detections(self, names, boxes):
        """
        """
        self.frame_index += 1
        if boxes.id is None:
            return
        
        self.version += 1
        seen_ids = set([int(item) for item in boxes.id.tolist()])
        num_detections = boxes.shape[0]
        #print(seen_ids)

        for i in range(num_detections):
            track_id = int(boxes.id[i].item())
            xyxy = boxes.xyxy[i]
            center = ((boxes.xyxy[i,0]+boxes.xyxy[i,2]).item()/2,(boxes.xyxy[i,1]+boxes.xyxy[i,3]).item()/2)
            class_name = names[boxes.cls[i].int().item()]
            conf = boxes.conf[i].item()
            #print(xyxy,track_id,center,class_name,conf)

        # Updates Existing Objects
            if track_id in self.objects:
                obj = self.objects[track_id]
                prev_moving_state = obj.moving
                obj.update(center, conf, self.frame_index, xyxy)

                if not prev_moving_state and obj.moving:
                    self.events.append(f"{obj.type}_{track_id} moved")
        # Adds New Objects
            else:
                self.objects[track_id] = WorldObject(track_id, class_name, center, conf, self.frame_index, xyxy)
                self.events.append(f"{class_name}_{track_id} appeared")

        # Handle disappearances
        for track_id, obj in self.objects.items():
            if (track_id not in seen_ids) and ((self.frame_index - obj.last_seen) > DISAPPEARANCE_THRESHOLD):
                if obj.visible:
                    obj.mark_missing()
                    self.events.append(f"{obj.type}_{track_id} disappeared")

        self._compute_relations()

#TODO: Reduce O(n^2) time complexity
    def _compute_relations(self):
        self.relations = []

        objs = list(self.objects.values())
        n = len(objs)

        for i in range(n):
            for j in range(i + 1, n):
                a = objs[i]
                b = objs[j]

                if not a.visible or not b.visible:
                    continue

                ax, ay = a.center
                bx, by = b.center
                dx = ax - bx
                dy = by - ay #flip cause y axis is downwards
                dist = math.dist(a.center,b.center)
                direction = DIRECTION_MAPPING[round(4*math.atan2(dy, dx)/math.pi)]
                IOU = bbox_iou(a.xyxy, b.xyxy)


                id_a = f"{a.type}_{a.track_id}"
                id_b = f"{b.type}_{b.track_id}"

                self.relations.append(Relation(id_a, direction, dist < DISTANCE_THRESHOLD, IOU > IOU_THRESHOLD, id_b))
                
    def snapshot(self):
        return {
            "world_version": self.version,
            "timestamp": self.frame_index,
            "objects": [obj.to_dict() for obj in self.objects.values() if obj.visible],
            "relations": [rel.to_dict() for rel in self.relations],
            "recent_events": list(self.events)
        }

class Relation:
    def __init__(self, subject_id, direction, near, overlapping, object_id):
        self.subject_id = subject_id
        self.direction = direction
        self.near = near
        self.overlapping = overlapping
        self.object_id = object_id

    def to_dict(self):
        return {
            "subject": self.subject_id,
            "relation": (self.direction + ", " + ("near" if self.near else "far") + ", " + ("overlapping" if self.overlapping else "not overlapping")),
            "object": self.object_id
        }

class WorldObject:
    def __init__(self, track_id, class_name, center, confidence, frame_idx, xyxy):
        self.track_id = track_id
        self.type = class_name

        self.xyxy = xyxy
        self.center = center
        self.prev_center = center
        self.velocity = (0.0, 0.0)

        self.visible = True
        self.moving = False

        self.confidence = confidence
        self.first_seen = frame_idx
        self.last_seen = frame_idx

    def update(self, center, confidence, frame_idx, xyxy):
        vx = center[0] - self.center[0]
        vy = center[1] - self.center[1]

        self.xyxy = xyxy
        self.prev_center = self.center
        self.center = center
        self.velocity = (vx, vy)

        self.moving = (abs(vx) + abs(vy)) > MOVEMENT_THRESHOLD
        self.confidence = confidence
        self.last_seen = frame_idx
        self.visible = True

    def mark_missing(self):
        self.visible = False

    def to_dict(self):
        return {
            "id": self.track_id,
            "type": self.type,
            "position": {"x": self.center[0], "y": self.center[1]},
            "velocity": {"x": self.velocity[0], "y": self.velocity[1]},
            "state": {
                "visible": self.visible,
                "moving": self.moving
            },
            "confidence": self.confidence
        }

#Initialize video stream, yolo and world state
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open video device. Check camera index or connection.")
    exit()
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width,frame_height)

model = YOLO("yolo26n.pt")

state = WorldState()

while True:
    #start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = model.track(frame, persist=True, tracker="botsort.yaml")
    #results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()
    # Display the annotated frame
    cv2.imshow("YOLO26 Tracking", annotated_frame)
    #print(type(results[0].names))
    #print(type(results[0].boxes))
    #print((results[0].boxes))
    state.update_from_detections(results[0].names, results[0].boxes)
    
    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #end_time = time.time()
    #elapsed_this_loop = end_time - start_time
    #print(f"Total time taken this loop: {elapsed_this_loop} seconds")
# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
#print([relation.to_dict() for relation in state.relations])
print(state.snapshot())