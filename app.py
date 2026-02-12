import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# 1. โหลดโมเดล (ถ้าไม่มีไฟล์ มันจะ Error บอกให้เอาไฟล์มาใส่)
try:
    model = YOLO('yolov8n-pose.pt')
except:
    st.error("ไม่พบไฟล์ yolov8n-pose.pt กรุณาอัปโหลดไฟล์นี้ขึ้น GitHub ด้วยครับ")
    st.stop()

# 2. ฟังก์ชันคำนวณมุม
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# 3. ตัวประมวลผลภาพ
class FitnessProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = "down"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # กลับด้านภาพเหมือนกระจก
        img = cv2.flip(img, 1)

        # ให้ AI มองหาคน
        results = model(img, verbose=False, conf=0.5)
        
        # วาดผลลัพธ์ลงบนภาพ
        for result in results:
            keypoints = result.keypoints.data[0].cpu().numpy()
            if len(keypoints) > 0:
                # พิกัดร่างกาย (ไหล่-ศอก-ข้อมือ)
                # 5=ไหล่ซ้าย, 7=ศอกซ้าย, 9=ข้อมือซ้าย
                p1 = keypoints[5][:2]
                p2 = keypoints[7][:2]
                p3 = keypoints[9][:2]

                # คำนวณมุม
                angle = calculate_angle(p1, p2, p3)
                
                # Logic การนับ (ยกตัวอย่าง Dumbbell Curl)
                if angle > 160: self.stage = "down"
                if angle < 30 and self.stage == "down":
                    self.stage = "up"
                    self.counter += 1
                
                # แสดงตัวเลขบนหน้าจอ
                cv2.putText(img, str(int(angle)), tuple(p2.astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(img, f"REPS: {self.counter}", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ส่วนแสดงผลหน้าเว็บ ---
st.title("🏋️ AI Fitness Web App")
st.write("รอสักครู่... แล้วกดปุ่ม START ด้านล่างเพื่อเริ่มออกกำลังกาย")

# กล่องเปิดกล้อง
webrtc_streamer(
    key="fitness",
    video_processor_factory=FitnessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
