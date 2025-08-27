
# import cv2
# import numpy as np
# import streamlit as st
# from sklearn.metrics import pairwise
# import time

# # Global variables
# background = None
# accumulated_weight = 0.5
# roi_top = 50
# roi_bottom = 300
# roi_right = 300
# roi_left = 600
# _MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# # Functions
# def calc_accum_avg(frame, accumulated_weight):
#     global background
#     if background is None:
#         background = frame.copy().astype("float")
#         return None
#     else:
#         cv2.accumulateWeighted(frame, background, accumulated_weight)

# def segment(frame, threshold_min=None):
#     diff = cv2.absdiff(background.astype('uint8'), frame)
#     diff = cv2.GaussianBlur(diff, (5, 5), 0)

#     if threshold_min is None:
#         _, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         _, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)

#     thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, _MORPH_KERNEL, iterations=2)
#     thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=2)

#     contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = [c for c in contours if cv2.contourArea(c) > 3000]

#     if not contours:
#         return None

#     hand_segments = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(hand_segments)
#     wrist_line = y + int(0.8 * h)
#     thresholded[wrist_line:, :] = 0

#     return (thresholded, hand_segments)

# def count_fingers(thresholded, hand_segment):
#     hull_idx = cv2.convexHull(hand_segment, returnPoints=False)
#     if hull_idx is None or len(hull_idx) < 3:
#         return 0

#     defects = cv2.convexityDefects(hand_segment, hull_idx)
#     if defects is None:
#         return 0

#     x, y, w, h = cv2.boundingRect(hand_segment)
#     scale = np.hypot(w, h)

#     finger_gaps = 0
#     for i in range(defects.shape[0]):
#         s, e, f, d = defects[i, 0]
#         start = hand_segment[s][0]
#         end = hand_segment[e][0]
#         far = hand_segment[f][0]

#         a = np.linalg.norm(end - start)
#         b = np.linalg.norm(far - start)
#         c = np.linalg.norm(far - end)

#         cos_angle = (b**2 + c**2 - a**2) / (2*b*c + 1e-7)
#         angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

#         depth = d / 256.0

#         if angle < 90 and depth > 0.05 * scale:
#             finger_gaps += 1

#     return min(finger_gaps + 1, 10)  # allow up to 10 fingers


# # Streamlit App
# st.title("✋ Finger Counter using OpenCV + Streamlit")

# run = st.checkbox("Start Camera")
# stframe = st.empty()
# info_text = st.empty()

# if run:
#     cap = cv2.VideoCapture(0)
#     num_frames = 0

#     while True:
#         ret, frames = cap.read()
#         if not ret:
#             st.error("Failed to access camera")
#             break

#         frames = cv2.flip(frames, 1)
#         frame_copy = frames.copy()

#         roi = frames[roi_top:roi_bottom, roi_right:roi_left]
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (7, 7), 0)

#         if num_frames < 60:
#             calc_accum_avg(gray, accumulated_weight)
#             cv2.putText(frame_copy, "WAIT... CAPTURING BACKGROUND", (100, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             hand = segment(gray)
#             if hand is not None:
#                 thresholded, hand_segment = hand
#                 hull_pts = cv2.convexHull(hand_segment)
#                 hull_shifted = (hull_pts + np.array([roi_right, roi_top])).astype(np.int32)
#                 cv2.polylines(frame_copy, [hull_shifted], True, (0, 255, 0), 2)

#                 fingers = count_fingers(thresholded, hand_segment)
#                 cv2.putText(frame_copy, f"Fingers: {fingers}", (70, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#         # Draw ROI box
#         cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)

#         num_frames += 1


# # Convert to RGB for Streamlit
#         frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
#         stframe.image(frame_rgb, channels="RGB")

#     cap.release()


import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Global variables
background = None
accumulated_weight = 0.5
roi_top, roi_bottom, roi_right, roi_left = 50, 300, 300, 600
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# Functions
def calc_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
    else:
        cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment(frame, threshold_min=None):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    if threshold_min is None:
        _, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)

    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, _MORPH_KERNEL, iterations=2)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=2)

    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 3000]

    if not contours:
        return None

    hand_segments = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(hand_segments)
    wrist_line = y + int(0.8 * h)
    thresholded[wrist_line:, :] = 0

    return (thresholded, hand_segments)


def count_fingers(thresholded, hand_segment):
    hull_idx = cv2.convexHull(hand_segment, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0

    defects = cv2.convexityDefects(hand_segment, hull_idx)
    if defects is None:
        return 0

    x, y, w, h = cv2.boundingRect(hand_segment)
    scale = np.hypot(w, h)

    finger_gaps = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = hand_segment[s][0]
        end = hand_segment[e][0]
        far = hand_segment[f][0]

        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(far - end)

        cos_angle = (b**2 + c**2 - a**2) / (2*b*c + 1e-7)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        depth = d / 256.0

        if angle < 90 and depth > 0.05 * scale:
            finger_gaps += 1

    return min(finger_gaps + 1, 10)


# Video Processor for WebRTC
class VideoProcessor(VideoTransformerBase):
    def __init__(self, mirror=True):
        self.num_frames = 0
        self.mirror = mirror

    def transform(self, frame):
        global background
        img = frame.to_ndarray(format="bgr24")

        # Flip if mirror mode is enabled
        if self.mirror:
            img = cv2.flip(img, 1)

        frame_copy = img.copy()

        roi = img[roi_top:roi_bottom, roi_right:roi_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.num_frames < 60:
            calc_accum_avg(gray, accumulated_weight)
            cv2.putText(frame_copy, "WAIT... CAPTURING BACKGROUND", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            hand = segment(gray)
            if hand is not None:
                thresholded, hand_segment = hand
                hull_pts = cv2.convexHull(hand_segment)
                hull_shifted = (hull_pts + np.array([roi_right, roi_top])).astype(np.int32)
                cv2.polylines(frame_copy, [hull_shifted], True, (0, 255, 0), 2)

                fingers = count_fingers(thresholded, hand_segment)
                cv2.putText(frame_copy, f"Fingers: {fingers}", (70, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)

        self.num_frames += 1
        return frame_copy


# Streamlit UI
st.title("✋ Finger Counter : ")

mirror_mode = st.checkbox("Mirror Camera (Selfie View)", value=False)

webrtc_streamer(
    key="finger-counter",
    video_processor_factory=lambda: VideoProcessor(mirror=mirror_mode)
)

