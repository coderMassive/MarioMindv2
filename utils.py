import numpy as np

def process_results(results, max_boxes):
    detections = []
    result = results[0]
    
    if result.boxes is not None:
        # Get boxes in xywh format (center_x, center_y, width, height)
        boxes = result.boxes.xywh.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            conf = confs[i]
            cls = clss[i]
            detections.append([x, y, w, h, conf, cls])

    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    detections = detections[:max_boxes]
    mario = next((d for d in detections if d[5] == 8), None)

    if mario:
        mario = mario.copy()
        dist = lambda d: (d[0]-mario[0])**2 + (d[1]-mario[1])**2
        detections = sorted(detections, key=dist)
    else:
        mario = [0, 0, 0, 0, 0, 0]

    for detection in detections:
        detection[0] = (detection[0] - mario[0]) / (2*256) + 0.5
        detection[1] = (detection[1] - mario[1]) / (2*240) + 0.5
        detection[2] /= 256
        detection[3] /= 240
        detection[4] /= 1
        detection[5] /= 20

    obs = np.zeros((max_boxes, 6), dtype=np.float32)

    for i, det in enumerate(detections):
        obs[i] = det

    return obs