import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# 1. ลองโหลดโมเดล
try:
    # ปรับ confidence ให้น้อยลง (0.3) เพื่อให้จับคนง่ายขึ้น
    model = YOLO('yolov8n-pose.pt')
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ! เช็คไฟล์ yolov8n-pose.pt ใน GitHub ด่วน: {e}")
    st.stop()

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

class FitnessProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # สั่งให้ AI ทำงาน (ลดความเข้มงวดลงเหลือ 0.3)
        results = model(img, verbose=False, conf=0.3)
        
        # --- ส่วนสำคัญ: วาดเส้นกระดูกทับลงไปเลย (จะได้รู้ว่า AI เห็นไหม) ---
        img = results[0].plot() 

        try:
            # ดึงข้อมูลจุดต่างๆ
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # เช็คว่ามองเห็น ไหล่(5), ศอก(7), ข้อมือ(9) ครบไหม?
            if keypoints[5][2] > 0.3 and keypoints[7][2] > 0.3 and keypoints[9][2] > 0.3:
                p1 = keypoints[5][:2] # ไหล่
                p2 = keypoints[7][:2] # ศอก
                p3 = keypoints[9][:2] # ข้อมือ

                angle = calculate_angle(p1, p2, p3)
                
                # Logic นับ
                if angle > 160: self.stage = "down"
                if angle < 40 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                
                # แสดงสถานะ
                cv2.putText(img, f"Angle: {int(angle)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                cv2.putText(img, f"Count: {self.counter}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            else:
                 cv2.putText(img, "Show Arms Clearly!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        except:
            cv2.putText(img, "No Person Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🏋️ Test Mode: Debugging")
st.write("ถ้าเห็นเส้นสีๆ ขีดทับตัวคน แปลว่า AI ทำงานแล้วครับ")
webrtc_streamer(
    key="fitness-debug",
    video_processor_factory=FitnessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
