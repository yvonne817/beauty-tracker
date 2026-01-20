import streamlit as st
import sys
import subprocess
import os
import time
from datetime import datetime, timedelta, date

# ==========================================
# 0. 環境建置
# ==========================================
def install(package):
    try: subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except: pass

try:
    import plotly.graph_objects as go
    from sklearn.cluster import KMeans
    import cv2
    import numpy as np
    from PIL import Image
    from streamlit_image_comparison import image_comparison
    import mediapipe as mp
except ImportError:
    packages = ["plotly", "scikit-learn", "opencv-python-headless", "Pillow", "streamlit-image-comparison", "mediapipe"]
    for p in packages: install(p)
    st.rerun()

if not os.path.exists("user_data"): os.makedirs("user_data")

# ==========================================
# 1. UI CSS
# ==========================================
st.set_page_config(page_title="美麗追蹤者 Beauty Tracker", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }
    .stApp { background-color: #fcfcfc; }
    
    /* 護理師對話框 */
    .nurse-box {
        border: 1px solid #e0e0e0; border-radius: 12px; padding: 15px;
        background-color: white; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* 狀態標籤 */
    .tag-danger { background-color: #ffebee; color: #c62828; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .tag-warn { background-color: #fff8e1; color: #f57f17; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .tag-good { background-color: #e8f5e9; color: #2e7d32; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    
    .metric-val { font-size: 28px; font-weight: 800; color: #333; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫 (V9 - 重置)
# ==========================================
@st.cache_resource
def get_db_v9(): 
    return {
        "0912345678": {
            "name": "王小美 (VIP)", "id": "A123456789", "treatment": "皮秒雷射 + 蜂巢探頭",
            "op_date": date.today() - timedelta(days=1), "history": [] 
        }
    }
USERS_DB = get_db_v9()

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_id' not in st.session_state: st.session_state.user_id = None

# ==========================================
# 3. 核心演算法
# ==========================================
class SkinEngine:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6
        )
    
    def align_faces(self, src_img, ref_img):
        h, w = ref_img.shape[:2]
        return cv2.resize(src_img, (w, h)), True

    def analyze(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # 退紅指數：分數越高 = 越白(健康)
        mean_a = np.mean(lab[:,:,1])
        # 調整靈敏度：128是中性，>128偏紅。係數調高讓紅腫照片分數更低
        red_score = 100 - (mean_a - 128) * 4.0 
        redness = max(20, min(99, red_score)) # 範圍 20~99

        # 斑點指數
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        spot_score = 100 - (np.sum(thresh)/thresh.size)*200
        spot = max(40, min(95, spot_score))
        
        # 紋路/毛孔
        edges = cv2.Canny(gray, 50, 150)
        wrinkle = max(50, 100 - (np.sum(edges)/edges.size)*500)
        
        pore = max(50, 100 - (np.sum(edges)/edges.size)*300)
        texture = (wrinkle + pore) / 2
        
        # 視覺化
        vis_spot = image.copy()
        vis_spot[thresh > 0] = [220, 0, 0]
        vis_spot = cv2.addWeighted(vis_spot, 0.3, image, 0.7, 0)
        
        return {
            "metrics": {
                "wrinkle": int(wrinkle), "spot": int(spot), "redness": int(redness), 
                "pore": int(pore), "texture": int(texture)
            },
            "vis_spot": vis_spot
        }

# --- 繪圖 ---
def plot_trend(history):
    labels = []
    for h in history:
        # 只取天數部分，例如 "Day 1"
        d = str(h['day'])
        labels.append(d)

    scores = [int(sum(h['metrics'].values())/5) for h in history]
    reds = [h['metrics']['redness'] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=scores, name="綜合評分", line=dict(color='#d4af37', width=5), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=labels, y=reds, name="退紅指數", line=dict(color='#e74c3c', width=3, dash='dot'), mode='lines+markers', yaxis='y2'))

    fig.update_layout(
        title="<b>術後恢復趨勢</b>",
        xaxis=dict(title="術後階段", showgrid=False),
        yaxis=dict(title="分數 (越高越好)", range=[0, 100], showgrid=True, gridcolor='#eee'),
        yaxis2=dict(title="退紅指數", overlaying='y', side='right', range=[0, 100], showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
        height=350, margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified", plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig

def plot_radar(curr):
    cats = ['紋路', '斑點', '退紅度', '毛孔', '平滑']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(curr.values()), theta=cats, fill='toself', name='本次', line_color='#d4af37'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor='rgba(0,0,0,0)'),
        dragmode=False, height=250, margin=dict(t=20, b=20, l=40, r=40), showlegend=False, paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def load_image(file_or_path):
    if not file_or_path: return None
    if isinstance(file_or_path, str):
        if not os.path.exists(file_or_path): return None
        img = Image.open(file_or_path).convert('RGB')
    else: img = Image.open(file_or_path).convert('RGB')
    return np.array(img)

# ==========================================
# 4. 主程式
# ==========================================
def main_app():
    user = USERS_DB[st.session_state.user_id]
    engine = SkinEngine()
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)
        st.title(f"{user['name']}")
        st.info(f"📋 療程：{user['treatment']}")
        if st.button("登出"): st.session_state.logged_in=False; st.rerun()

    st.markdown("## 美麗追蹤者 Beauty Tracker")
    tab1, tab2, tab3 = st.tabs(["🩺 AI 診斷 (Live)", "📊 歷史畫廊", "📅 預約回診"])

    # TAB 1: Live 診斷
    with tab1:
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                # 您的精準天數需求
                stage = st.selectbox("術後階段", [
                    "術後第 1 天", "術後第 2 天", "術後第 3 天", 
                    "術後第 7 天", "術後第 14 天", "術後第 30 天", "術後 30 天以上"
                ])
            with c2: f_curr = st.file_uploader("上傳今日照片", type=["jpg", "png"])
            with c3: 
                if os.path.exists("before.jpg"):
                    st.success("✅ 術前圖鎖定")
                    img_ref = load_image("before.jpg")
                else:
                    f_ref = st.file_uploader("術前圖", type=["jpg"])
                    img_ref = load_image(f_ref) if f_ref else None

        if f_curr and img_ref is not None:
            img_curr = load_image(f_curr)
            
            with st.spinner("AI 運算中..."):
                final, success = engine.align_faces(img_curr, img_ref)
                res = engine.analyze(final)
                metrics = res['metrics']
            
            col_L, col_R = st.columns([1, 1])
            
            # 左側：影像
            with col_L:
                st.markdown("### 👁️ 智能影像對比")
                image_comparison(img1=img_ref, img2=final, label1="術前", label2="目前", width=500, in_memory=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                red = metrics['redness']
                spot = metrics['spot']
                avg = int(sum(metrics.values())/5)
                
                # 顯示邏輯：分數越低 = 越紅 = 狀態越差
                # 但要分階段顯示標籤
                c_red = "tag-danger" if red < 50 else ("tag-warn" if red < 80 else "tag-good")
                
                if red < 50: t_red = "嚴重紅腫"
                elif red < 80: t_red = "術後泛紅" # 這是正常的中間值
                else: t_red = "恢復極佳"

                k1.markdown(f"""<div style="text-align:center"><span class="{c_red}">{t_red}</span><h2>{red}</h2><small>退紅指數</small></div>""", unsafe_allow_html=True)
                
                c_spot = "tag-warn" if spot < 70 else "tag-good"
                k2.markdown(f"""<div style="text-align:center"><span class="{c_spot}">{"代謝中" if spot<70 else "淡化顯著"}</span><h2>{spot}</h2><small>斑點指數</small></div>""", unsafe_allow_html=True)
                
                k3.markdown(f"""<div style="text-align:center"><span class="tag-good">綜合</span><h2>{avg}</h2><small>總評分</small></div>""", unsafe_allow_html=True)

            # 右側：AI 護理師 (修復後的邏輯)
            with col_R:
                st.markdown("### 👩‍⚕️ AI 護理師建議")
                nurse_avatar = "nurse.png" if os.path.exists("nurse.png") else "👩‍⚕️"
                
                with st.container(border=True):
                    
                    # 邏輯核心：結合「天數」與「分數」
                    is_early_stage = ("第 1 天" in stage) or ("第 2 天" in stage) or ("第 3 天" in stage)
                    is_mid_stage = ("第 7 天" in stage)
                    is_late_stage = ("30" in stage)

                    # [情境 1] 術後初期 (Day 1-3)
                    if is_early_stage:
                        st.info(f"📅 **目前階段：{stage} (黃金修復期)**")
                        
                        if red < 50: # 分數極低 -> 異常
                            st.chat_message("assistant", avatar="🚑").markdown("**⚠️ 警報：紅腫指數過低 (異常)**")
                            st.write("雖處於術後初期，但您的紅腫反應較為強烈。")
                            st.error("處置：請每 2 小時冰敷一次，並厚敷凡士林。若伴隨刺痛請回診。")
                            st.button("📞 SOS 緊急通話", type="primary", use_container_width=True)
                        
                        elif red < 80: # 分數 50-80 -> 正常 (這就是您要的！)
                            st.chat_message("assistant", avatar=nurse_avatar).markdown("**🌡️ 狀態：正常術後熱效應**")
                            st.write("目前的泛紅屬於**正常修復現象**，不必過度擔心。")
                            st.success("處置：請加強保濕，並早晚冰敷 15 分鐘即可退紅。")
                        
                        else: # 分數 > 80 -> 超好
                            st.chat_message("assistant", avatar="🎉").markdown("**✨ 狀態：天選之人**")
                            st.write("太驚人了！術後初期居然幾乎沒有紅腫，您的體質修復力極佳！")

                    # [情境 2] 術後中期 (Day 7)
                    elif is_mid_stage:
                         st.info(f"📅 **目前階段：{stage} (結痂脫落期)**")
                         st.chat_message("assistant", avatar=nurse_avatar).write("皮膚表面可能會有細微結痂，**請勿用手摳除**，以免留疤。請務必做好防曬。")

                    # [情境 3] 術後長期 (Day 30+)
                    elif is_late_stage:
                        st.info(f"📅 **目前階段：{stage} (成果驗收)**")
                        st.chat_message("assistant", avatar="🎉").write(f"恭喜畢業！綜合評分 {avg} 分。膚況已穩定，建議轉為保養型雷射維持。")
                        st.balloons()
                    
                    # [情境 4] 異常 (例如 Day 30 還很紅)
                    elif red < 60:
                        st.chat_message("assistant", avatar="🚑").markdown("**⚠️ 異常警報**")
                        st.error("術後已過恢復期，但退紅指數仍偏低。可能為酒糟或過敏發作，請立即預約回診檢查。")
                        st.button("📅 立即預約回診", type="primary", use_container_width=True)

                    else:
                        st.chat_message("assistant", avatar=nurse_avatar).write("膚況穩定，請繼續保持良好的生活作息與防曬習慣。")

                # 存檔
                if st.button("💾 存入病歷 (含照片)", type="primary", use_container_width=True):
                    save_path = f"user_data/{int(time.time())}.jpg"
                    Image.fromarray(final).save(save_path)
                    
                    # 簡單處理天數標籤
                    try: 
                        day_clean = stage.split(" ")[1] + " " + stage.split(" ")[2] # 取出 "第 1"
                        day_label = f"Day {day_clean}"
                    except:
                        day_label = stage

                    user['history'].append({
                        "day": day_label, "metrics": metrics, 
                        "date": str(date.today()), "img_path": save_path
                    })
                    st.toast("✅ 存檔成功！")
                    time.sleep(1)
                    st.rerun()

    # TAB 2: 歷史畫廊
    with tab2:
        if user['history']:
            st.markdown("### 📈 療程成效追蹤")
            st.plotly_chart(plot_trend(user['history']), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📸 歷史影像紀錄")
            for i, rec in enumerate(reversed(user['history'])):
                with st.container(border=True):
                    c_img, c_radar = st.columns([1, 2])
                    with c_img:
                        if rec.get("img_path") and os.path.exists(rec["img_path"]):
                            st.image(rec["img_path"], caption=f"{rec['day']} ({rec['date']})")
                    with c_radar:
                        st.markdown(f"**{rec['day']} 分析報告**")
                        st.plotly_chart(plot_radar(rec['metrics']), key=f"radar_{i}", use_container_width=True, height=200)
        else:
            st.info("尚無歷史數據")

    # TAB 3: 預約
    with tab3:
        st.subheader("📅 預約回診")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.date_input("日期")
            st.selectbox("時段", ["14:00", "15:00", "19:00"])
            if st.button("確認預約", type="primary"):
                st.success("預約成功！")
        with c2:
            st.info("📍 台北市信義區松高路 68 號")
            st.warning("📞 0800-888-888")

def login_page():
    st.title("Beauty Tracker Login")
    if st.button("登入測試帳號"):
        st.session_state.logged_in=True; st.session_state.user_id="0912345678"; st.rerun()

if __name__ == "__main__":
    if st.session_state.logged_in: main_app()
    else: login_page()