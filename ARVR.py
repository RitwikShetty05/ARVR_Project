import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.title("Snaochat Clone - AR Filters with Camera Feed")

st.markdown("""
This is a simplified clone of Snapchat's camera with AR overlay filters.
Select a filter to apply it to the live camera feed
Note: The app may restart the camera when changing filters due to Streamlit's rerun behavior.
""")

filter_type = st.selectbox("Choose Filter", ["None", "Glasses", "Dog Nose"])

def create_video_processor(selected_filter):
    mp_face_mesh = mp.solutions.face_mesh

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.selected_filter = selected_filter

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = img.shape
                    lm = face_landmarks.landmark

                    if self.selected_filter == "Glasses":
                        # Draw simple glasses: circles around eyes and bridge
                        # Left eye centers (approx using landmarks 33 and 133)
                        left_eye_outer = (int(lm[33].x * w), int(lm[33].y * h))
                        left_eye_inner = (int(lm[133].x * w), int(lm[133].y * h))
                        center_left = ((left_eye_outer[0] + left_eye_inner[0]) // 2, (left_eye_outer[1] + left_eye_inner[1]) // 2)
                        radius_left = abs(left_eye_outer[0] - left_eye_inner[0]) // 2 + 10
                        cv2.circle(img, center_left, radius_left, (0, 0, 0), 2)

                        # Right eye (landmarks 362 and 263)
                        right_eye_outer = (int(lm[362].x * w), int(lm[362].y * h))
                        right_eye_inner = (int(lm[263].x * w), int(lm[263].y * h))
                        center_right = ((right_eye_outer[0] + right_eye_inner[0]) // 2, (right_eye_outer[1] + right_eye_inner[1]) // 2)
                        radius_right = abs(right_eye_outer[0] - right_eye_inner[0]) // 2 + 10
                        cv2.circle(img, center_right, radius_right, (0, 0, 0), 2)

                        # Bridge between inner eyes
                        cv2.line(img, left_eye_inner, right_eye_outer, (0, 0, 0), 2)

                    elif self.selected_filter == "Dog Nose":
                        # Draw black circle on nose tip (landmark 1)
                        nose_tip = (int(lm[1].x * w), int(lm[1].y * h))
                        cv2.circle(img, nose_tip, 20, (0, 0, 0), -1)

                        # Simple dog ears: triangles above eyebrows
                        # Left ear above left eyebrow (approx using landmark 70)
                        left_brow = (int(lm[70].x * w), int(lm[70].y * h))
                        ear_left_points = [
                            (left_brow[0], left_brow[1] - 100),  # Top
                            (left_brow[0] - 50, left_brow[1] - 50),  # Left base
                            (left_brow[0] + 50, left_brow[1] - 50)   # Right base
                        ]
                        cv2.fillPoly(img, [np.array(ear_left_points)], (139, 69, 19))  # Brown color

                        # Right ear above right eyebrow (landmark 300)
                        right_brow = (int(lm[300].x * w), int(lm[300].y * h))
                        ear_right_points = [
                            (right_brow[0], right_brow[1] - 100),  # Top
                            (right_brow[0] - 50, right_brow[1] - 50),  # Left base
                            (right_brow[0] + 50, right_brow[1] - 50)   # Right base
                        ]
                        cv2.fillPoly(img, [np.array(ear_right_points)], (139, 69, 19))  # Brown color

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return VideoProcessor

webrtc_streamer(
    key="camera",
    video_processor_factory=lambda: create_video_processor(filter_type),
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("""
### Instructions:
- Allow camera access when prompted.
- Select a filter to see AR overlays on your face.
- This uses MediaPipe for face detection and OpenCV for drawing overlays.
- To host: Save as app.py, run `streamlit run app.py`. For Streamlit Cloud, upload with requirements.txt containing:
  streamlit
  streamlit-webrtc
  opencv-python-headless
  mediapipe
  av
""")