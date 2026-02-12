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
    st.error("ไม่พบไฟล์ yolov8n-pose.pt ใน GitHub")
    st.stop()

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

if 'exercise_mode' not in st.session_state:
    st.session_state['exercise_mode'] = "Standing Bicep Curl"

class FitnessProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.set_count = 0
        self.stage = "down"  # บังคับให้เริ่มจากท่าลงสุดเสมอ
        self.reps_per_set = 10
        self.feedback = "GET READY"
        self.color = (255, 255, 0)
        self.cooldown = 0 # ป้องกันการนับซ้ำในเสี้ยววินาที

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        mode = st.session_state.get('exercise_mode', "Standing Bicep Curl")
        
        # ปรับ conf ให้สูงขึ้นเป็น 0.6 เพื่อลดการจับจุดมั่ว
        results = model(img, verbose=False, conf=0.6)
        
        try:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # จุดซีกขวา (AI มองเป็นซ้ายจอ)
            # 6=ไหล่, 8=ศอก, 10=ข้อมือ, 12=เอว
            p_sh = keypoints[6][:2]
            p_el = keypoints[8][:2]
            p_wr = keypoints[10][:2]
            p_hip = keypoints[12][:2]

            # ต้องมั่นใจในจุด ไหล่ ศอก และเอว (Confidence > 0.6)
            if keypoints[6][2] > 0.6 and keypoints[8][2] > 0.6 and keypoints[12][2] > 0.6:
                
                # --- 1. BICEP CURL (เข้มงวดขึ้น) ---
                if mode == "Standing Bicep Curl":
                    angle = calculate_angle(p_sh, p_el, p_wr)
                    sway = abs(p_sh[0] - p_hip[0])
                    
                    if sway > 50:
                        self.feedback = "STAY STILL! DON'T SWING"
                        self.color = (0, 0, 255)
                    elif angle < 35: # ขึ้นสุด (ต้องพับแขนเยอะขึ้น)
                        if self.stage == "down":
                            self.counter += 1
                            self.stage = "up"
                        self.feedback = "GOOD! NOW LOWER SLOWLY"
                        self.color = (0, 255, 0)
                    elif angle > 150: # ลงสุด (ต้องเหยียดเกือบสุด)
                        self.stage = "down"
                        self.feedback = "CURL UP!"
                        self.color = (0, 255, 0)

                # --- 2. UPRIGHT ROW (เช็คระยะข้อมือ) ---
                elif mode == "Standing Upright Row":
                    # ใช้ความสูงข้อมือเทียบไหล่
                    if p_el[1] < p_sh[1] - 20: # ศอกสูงเกินไหล่
                        self.feedback = "ELBOWS TOO HIGH! STOP AT SHOULDER"
                        self.color = (0, 0, 255)
                    elif p_wr[1] < p_sh[1] + 40: # ข้อมือขึ้นถึงระดับอก
                        if self.stage == "down":
                            self.counter += 1
                            self.stage = "up"
                        self.feedback = "WELL DONE"
                        self.color = (0, 255, 0)
                    elif p_wr[1] > p_hip[1] - 50: # ข้อมือลงต่ำถึงเอว
                        self.stage = "down"
                        self.feedback = "PULL UP"
                        self.color = (0, 255, 0)

                # --- 3. FRONT RAISE (เช็คองศาไหล่) ---
                elif mode == "Standing Front Raise":
                    arm_angle = calculate_angle(p_el, p_sh, p_hip)
                    
                    if arm_angle > 105:
                        self.feedback = "TOO HIGH! STOP AT 90 DEG"
                        self.color = (0, 0, 255)
                    elif arm_angle > 80: # ยกถึงระดับสายตา
                        if self.stage == "down":
                            self.counter += 1
                            self.stage = "up"
                        self.feedback = "PERFECT LEVEL"
                        self.color = (0, 255, 0)
                    elif arm_angle < 25: # วางแขนลงข้างลำตัว
                        self.stage = "down"
                        self.feedback = "RAISE UP"
                        self.color = (0, 255, 0)

                # จัดการ Set
                if self.counter >= self.reps_per_set:
                    self.set_count += 1
                    self.counter = 0

                # --- วาด UI ---
                cv2.rectangle(img, (0, 0), (640, 60), self.color, -1)
                cv2.putText(img, self.feedback, (20, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # แถบแสดง Reps
                cv2.rectangle(img, (0, 400), (220, 480), (0, 0, 0), -1)
                cv2.putText(img, f"REPS: {self.counter}", (10, 435), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f"SETS: {self.set_count}", (10, 470), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception:
            pass
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🏋️ AI Coach (Strict Mode)")
# ... (ส่วนเลือก Mode และ WebRTC เหมือนเดิม)
