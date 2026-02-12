import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# โหลดโมเดล
try:
    model = YOLO('yolov8n-pose.pt')
except Exception as e:
    st.error("ไม่พบไฟล์ yolov8n-pose.pt ใน GitHub ของคุณ")
    st.stop()

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

class FitnessProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0        # นับจำนวนครั้ง
        self.set_count = 0      # นับจำนวนเซต
        self.stage = "down"
        self.reps_per_set = 10  # เป้าหมายต่อเซต (แก้เลขตรงนี้ได้)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # ให้ AI ทำงาน
        results = model(img, verbose=False, conf=0.5)
        
        try:
            # ดึงข้อมูลจุดต่างๆ
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # ตรวจสอบว่าเห็น ไหล่(5), ศอก(7), ข้อมือ(9) ไหม
            if keypoints[5][2] > 0.5 and keypoints[7][2] > 0.5 and keypoints[9][2] > 0.5:
                p1 = keypoints[5][:2]
                p2 = keypoints[7][:2]
                p3 = keypoints[9][:2]

                # คำนวณมุมแขน
                angle = calculate_angle(p1, p2, p3)
                
                # --- Logic การนับ (ยกดัมเบล) ---
                # แขนเหยียดตรง (>160 องศา) = ลงสุด
                if angle > 160: 
                    self.stage = "down"
                
                # แขนพับเข้ามา (<30 องศา) และเคยลงสุดมาก่อน = นับ 1
                if angle < 30 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                    
                    # --- Logic ตัดรอบเซต ---
                    if self.counter >= self.reps_per_set:
                        self.set_count += 1   # เพิ่มจำนวนเซต
                        self.counter = 0      # รีเซ็ตจำนวนครั้ง
                
                # --- แสดงผลหน้าจอ (UI แบบคลีนๆ) ---
                # วาดกล่องพื้นหลังสีดำจางๆ ให้อ่านตัวเลขง่าย
                cv2.rectangle(img, (0, 0), (250, 150), (0, 0, 0), -1)
                
                # แสดงจำนวนครั้ง (REPS)
                cv2.putText(img, f"REPS: {self.counter}/{self.reps_per_set}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # แสดงจำนวนเซต (SETS)
                cv2.putText(img, f"SETS: {self.set_count}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # แสดงสถานะ (ขึ้น/ลง)
                status_color = (0, 255, 255) if self.stage == "down" else (0, 0, 255)
                cv2.putText(img, f"STATE: {self.stage}", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

        except Exception:
            pass
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("💪 AI Fitness Trainer")
st.write(f"เป้าหมาย: 10 ครั้ง = 1 เซต")

webrtc_streamer(
    key="fitness-clean",
    video_processor_factory=FitnessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
