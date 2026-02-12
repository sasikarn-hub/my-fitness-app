import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# --- Config ---
try:
    model = YOLO('yolov8n-pose.pt')
except Exception as e:
    st.error("ไม่พบไฟล์ yolov8n-pose.pt ใน GitHub ของคุณ")
    st.stop()

# ฟังก์ชันคำนวณมุม (3 จุด)
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# ตัวแปร Global เพื่อส่งค่าจาก Dropdown ไปหา Class AI
if 'exercise_mode' not in st.session_state:
    st.session_state['exercise_mode'] = "Standing Bicep Curl"

class FitnessProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.set_count = 0
        self.stage = "down"
        self.reps_per_set = 10
        self.warning_msg = "" # ข้อความเตือน

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # รับค่าท่าที่เลือกจาก Session State
        mode = st.session_state.get('exercise_mode', "Standing Bicep Curl")

        results = model(img, verbose=False, conf=0.5)
        
        try:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # ตรวจจับร่างกายซีกขวา (AI มองเป็นซ้ายของภาพ)
            # 6=ไหล่ขวา, 8=ศอกขวา, 10=ข้อมือขวา, 12=เอวขวา
            # เราใช้ซีกขวาเป็นหลัก (หรือจะแก้ให้ detect ทั้งสองข้างก็ได้)
            p_shoulder = keypoints[6][:2]
            p_elbow = keypoints[8][:2]
            p_wrist = keypoints[10][:2]
            p_hip = keypoints[12][:2]

            # เช็คความมั่นใจของจุดต่างๆ (ต้องเห็นชัด)
            if keypoints[6][2] > 0.5 and keypoints[8][2] > 0.5:
                
                # --- LOGIC 1: Bicep Curl (พับแขน) ---
                if mode == "Standing Bicep Curl":
                    # มุมข้อศอก (ไหล่-ศอก-ข้อมือ)
                    angle = calculate_angle(p_shoulder, p_elbow, p_wrist)
                    
                    # Cheat Check: ตัวโยก? (เช็คไหล่เทียบกับเอวในแนวแกน X)
                    # ถ้าไหล่ขยับหนีเอวมากเกินไป = เหวี่ยง
                    shoulder_sway = abs(p_shoulder[0] - p_hip[0])
                    if shoulder_sway > 50: # ค่าสมมติ ปรับได้
                        self.warning_msg = "!! DON'T SWING !!"
                    else:
                        self.warning_msg = ""

                    # Counting Logic
                    if angle > 160: self.stage = "down"
                    if angle < 30 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1

                # --- LOGIC 2: Upright Row (ดึงข้อศอก) ---
                elif mode == "Standing Upright Row":
                    # วัดมุมรักแร้ (ศอก-ไหล่-เอว) เพื่อดูการยกแขน
                    arm_body_angle = calculate_angle(p_elbow, p_shoulder, p_hip)
                    
                    # Cheat Check: ศอกสูงเกินไหล่? (เสี่ยงไหล่หนีบ)
                    # ถ้าค่า Y ของศอก น้อยกว่า ไหล่ (ในภาพ Y น้อย = อยู่สูงกว่า)
                    if p_elbow[1] < p_shoulder[1]: 
                        self.warning_msg = "!! ELBOW TOO HIGH !!"
                    else:
                        self.warning_msg = ""

                    # Counting Logic (ดูความสูงข้อมือเทียบเอว/อก)
                    # ลง: ข้อมืออยู่ต่ำกว่าเอว
                    if p_wrist[1] > p_hip[1]: 
                        self.stage = "down"
                    # ขึ้น: ข้อมืออยู่สูงระดับอก (และเคยลงมาก่อน)
                    if p_wrist[1] < p_shoulder[1] + 50 and self.stage == "down": 
                        self.stage = "up"
                        self.counter += 1

                # --- LOGIC 3: Front Raise (ยกแขนหน้า) ---
                elif mode == "Standing Front Raise":
                    # วัดมุมรักแร้ (ศอก-ไหล่-เอว)
                    arm_angle = calculate_angle(p_elbow, p_shoulder, p_hip)
                    
                    # Cheat Check: ยกสูงเกิน 90 องศา?
                    if arm_angle > 100:
                        self.warning_msg = "!! TOO HIGH (STOP AT 90) !!"
                    # Cheat Check: เอนหลัง? (ไหล่เลยเอวไปด้านหลัง)
                    elif p_shoulder[0] < p_hip[0] - 30: 
                        self.warning_msg = "!! DON'T LEAN BACK !!"
                    else:
                        self.warning_msg = ""

                    # Counting Logic
                    if arm_angle < 20: self.stage = "down"
                    if arm_angle > 85 and arm_angle < 100 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1

                # --- ส่วนจัดการ Set ---
                if self.counter >= self.reps_per_set:
                    self.set_count += 1
                    self.counter = 0

                # --- HUD Display ---
                # กล่องดำพื้นหลัง
                cv2.rectangle(img, (0, 0), (350, 200), (0, 0, 0), -1)
                
                # แสดง Mode
                cv2.putText(img, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # แสดง Warning (ถ้ามี)
                if self.warning_msg:
                    cv2.rectangle(img, (0, 200), (400, 250), (0, 0, 255), -1) # แถบแดง
                    cv2.putText(img, self.warning_msg, (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # แสดง Reps/Sets
                cv2.putText(img, f"REPS: {self.counter}/{self.reps_per_set}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(img, f"SETS: {self.set_count}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # แสดงสถานะแขน
                status_color = (0, 255, 255) if self.stage == "down" else (0, 0, 255)
                cv2.putText(img, f"STATE: {self.stage}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        except Exception as e:
            pass
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ส่วนหน้าเว็บ ---
st.title("🏋️ AI Fitness Pro: 3 Dumbbell Moves")

# สร้าง Dropdown สำหรับเลือกท่า
option = st.selectbox(
    'เลือกท่าออกกำลังกาย:',
    ('Standing Bicep Curl', 'Standing Upright Row', 'Standing Front Raise')
)

# อัปเดต session state เพื่อส่งค่าไปให้ AI
st.session_state['exercise_mode'] = option

# คำอธิบายท่า
if option == 'Standing Bicep Curl':
    st.info("💡 ทริค: ล็อกศอกให้นิ่ง อย่าโยกตัวช่วยเหวี่ยง")
elif option == 'Standing Upright Row':
    st.info("💡 ทริค: ดึงศอกแค่ระดับไหล่ (อย่าเกินหู) ระวังไหล่หนีบ")
elif option == 'Standing Front Raise':
    st.info("💡 ทริค: ยกแค่ระดับสายตา (90 องศา) อย่าแอ่นหลัง")

webrtc_streamer(
    key="fitness-pro",
    video_processor_factory=FitnessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
