import os
import time
import uuid
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from streamlit_image_comparison import image_comparison

# =========================================================
# 0) App Config
# =========================================================
APP_TITLE = "美麗追蹤者 Beauty Tracker"
DB_PATH = "beauty_tracker.db"

DATA_DIR = Path("user_data")
DATA_DIR.mkdir(exist_ok=True)

# SMTP (optional) - if you want clinic to receive emails immediately
# set env:
#   BT_SMTP_HOST, BT_SMTP_PORT, BT_SMTP_USER, BT_SMTP_PASS, BT_NOTIFY_TO
SMTP_HOST = os.getenv("BT_SMTP_HOST", "")
SMTP_PORT = int(os.getenv("BT_SMTP_PORT", "587"))
SMTP_USER = os.getenv("BT_SMTP_USER", "")
SMTP_PASS = os.getenv("BT_SMTP_PASS", "")
NOTIFY_TO = os.getenv("BT_NOTIFY_TO", "")

# Clinic hours & appointment slot
CLINIC_OPEN_HOUR = 10
CLINIC_CLOSE_HOUR = 19
SLOT_MINUTES = 30  # internal slot step, UI will show dropdown

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }
.stApp { background-color: #fcfcfc; }
.metric-val { font-size: 26px; font-weight: 800; color: #222; line-height: 1.1; }
.metric-title { font-size: 13px; color: #555; margin-top: 4px; }
.metric-sub { font-size: 12px; color: #666; }
.pill { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; }
.pill-good { background:#e8f5e9; color:#1b5e20; }
.pill-warn { background:#fff8e1; color:#e65100; }
.pill-bad  { background:#ffebee; color:#b71c1c; }
.hr { height:1px; background:#eee; margin: 12px 0; }
small { color:#666; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1) DB
# =========================================================
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        treatment TEXT NOT NULL,
        op_date TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        stage_day INTEGER NOT NULL,
        stage_label TEXT NOT NULL,
        record_date TEXT NOT NULL,
        img_path TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        baseline_json TEXT NOT NULL,
        improvement_json TEXT NOT NULL,
        quality_json TEXT NOT NULL,
        symptoms_json TEXT NOT NULL,
        tasks_done_pct INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(user_id, stage_day)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS notifications (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        stage_day INTEGER NOT NULL,
        triage_level TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        appt_dt TEXT NOT NULL,
        note TEXT,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        UNIQUE(user_id, appt_dt)
    )
    """)

    conn.commit()
    conn.close()

init_db()

# =========================================================
# 2) Demo User (replace with real auth later)
# =========================================================
DEMO_USER = {
    "user_id": "0912345678",
    "display_name": "王小美 (VIP)",
    "treatment": "皮秒雷射 + 蜂巢探頭",
    "op_date": (date.today() - timedelta(days=1)).isoformat(),
}

def ensure_demo_user():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE user_id=?", (DEMO_USER["user_id"],))
    row = cur.fetchone()
    if not row:
        cur.execute(
            "INSERT INTO users(user_id, display_name, treatment, op_date) VALUES (?,?,?,?)",
            (DEMO_USER["user_id"], DEMO_USER["display_name"], DEMO_USER["treatment"], DEMO_USER["op_date"])
        )
        conn.commit()
    conn.close()

ensure_demo_user()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# =========================================================
# 3) Files & Images
# =========================================================
def user_dir(user_id: str) -> Path:
    d = DATA_DIR / user_id
    (d / "records").mkdir(parents=True, exist_ok=True)
    return d

def before_path(user_id: str) -> Path:
    return user_dir(user_id) / "before.jpg"

def load_image(file_or_path):
    if file_or_path is None:
        return None
    if isinstance(file_or_path, (str, Path)):
        fp = str(file_or_path)
        if not os.path.exists(fp):
            return None
        img = Image.open(fp).convert("RGB")
    else:
        img = Image.open(file_or_path).convert("RGB")
    return np.array(img)

def save_rgb_image(arr_rgb, dst_path: Path):
    Image.fromarray(arr_rgb).save(str(dst_path))

# =========================================================
# 4) Core Engine
# =========================================================
class SkinEngine:
    def align_faces(self, src_img, ref_img):
        # demo: stable resize alignment
        h, w = ref_img.shape[:2]
        return cv2.resize(src_img, (w, h)), True

    def analyze(self, image_rgb):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

        # redness from a-channel mean
        mean_a = float(np.mean(lab[:, :, 1]))
        red_score = 100 - (mean_a - 128) * 4.0
        redness = int(max(20, min(99, red_score)))

        # spots from adaptive threshold area
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10
        )
        spot_score = 100 - (np.sum(thresh) / thresh.size) * 200
        spot = int(max(40, min(95, spot_score)))

        # texture proxies
        edges = cv2.Canny(gray, 50, 150)
        wrinkle = float(max(50, 100 - (np.sum(edges) / edges.size) * 500))
        pore = float(max(50, 100 - (np.sum(edges) / edges.size) * 300))
        texture = float((wrinkle + pore) / 2)

        return {
            "wrinkle": int(wrinkle),
            "spot": int(spot),
            "redness": int(redness),
            "pore": int(pore),
            "texture": int(texture),
        }

def metrics_avg(metrics: dict):
    return int(sum(metrics.values()) / max(1, len(metrics)))

def improvement_pct(curr_score: int, base_score: int):
    # higher is better
    base = max(0, min(100, int(base_score)))
    curr = max(0, min(100, int(curr_score)))
    denom = max(1, 100 - base)
    pct = (curr - base) / denom * 100.0
    return float(max(-100.0, min(100.0, pct)))

def quality_check(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean_b = float(np.mean(gray))
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    tags = []
    score = 100.0

    if mean_b < 70:
        tags.append("太暗")
        score -= min(30, (70 - mean_b) * 0.6)
    if mean_b > 185:
        tags.append("太亮/過曝")
        score -= min(30, (mean_b - 185) * 0.6)
    if blur_var < 80:
        tags.append("偏模糊")
        score -= min(35, (80 - blur_var) * 0.4)

    score = max(0.0, min(100.0, score))
    ok = score >= 60 and ("太暗" not in tags) and ("太亮/過曝" not in tags)
    return {"ok": ok, "score": int(score), "brightness": int(mean_b), "sharpness": int(blur_var), "tags": tags}

# =========================================================
# 5) Product-like Care Journey
# =========================================================
STAGES = [
    ("術後第 1 天", 1),
    ("術後第 2 天", 2),
    ("術後第 3 天", 3),
    ("術後第 7 天", 7),
    ("術後第 14 天", 14),
    ("術後第 30 天", 30),
    ("術後 30 天以上", 999),
]

def stage_tasks(stage_day: int):
    if stage_day <= 3:
        return [
            "冰敷 10–15 分鐘（每 2–3 小時一次）",
            "加強保濕（至少 3 次）",
            "避免熱敷、劇烈運動、烤箱/三溫暖",
            "避免搓揉、去角質、酸類保養",
            "外出防曬（遮蔽 + SPF）",
        ]
    if stage_day <= 14:
        return [
            "加強保濕（至少 2–3 次）",
            "避免摳痂/抓癢，讓其自然脫落",
            "外出防曬（遮蔽 + SPF）",
            "避免酸類/刺激性保養至穩定",
            "每日溫和清潔（不過度清潔）",
        ]
    return [
        "日常防曬（SPF + 遮蔽）",
        "保濕維持（早晚）",
        "避免過度去角質與刺激性療程",
        "觀察是否有局部色素沉著並記錄",
    ]

def triage_from_symptoms(pain, heat, swelling, oozing, fever):
    if fever or oozing:
        return ("紅燈", "建議立即聯絡診所並安排回診；若合併劇痛、發燒或持續滲液，請立即就醫。", "pill-bad")
    if pain >= 7 or swelling >= 7:
        return ("紅燈", "疼痛/腫脹偏高，建議立即聯絡診所評估，並依醫師指示處理。", "pill-bad")
    if heat >= 6 or pain >= 5 or swelling >= 5:
        return ("黃燈", "症狀略高於一般預期，建議今日聯絡診所諮詢，並密切觀察是否加劇。", "pill-warn")
    return ("綠燈", "目前屬常見恢復反應，持續保濕、防曬與溫和照護即可。", "pill-good")

def explain_improvements(impr: dict, stage_day: int, q: dict):
    lines = []
    if not q["ok"]:
        lines.append(f"本次照片品質：{q['score']} 分（{', '.join(q['tags']) if q['tags'] else '可再提升'}）。建議依拍攝指引重拍，讓改善%更可信。")
        return lines

    if stage_day <= 3:
        lines.append("屬術後早期：泛紅/熱感可能波動，重點是舒緩與穩定。")
    elif stage_day <= 14:
        lines.append("屬修復代謝期：防曬與保濕會明顯影響成效。")
    else:
        lines.append("屬穩定維持期：以日常保養/維持型療程延續效果。")

    red = impr["redness"]
    if red >= 12:
        lines.append(f"退紅改善明顯（+{red:.0f}%）：泛紅趨勢下降，恢復進度良好。")
    elif red <= -10:
        lines.append(f"退紅較術前偏弱（{red:.0f}%）：可能受光線或刺激影響，建議加強舒緩並觀察趨勢。")
    else:
        lines.append(f"退紅變化中（{red:.0f}%）：屬正常波動，請以趨勢判讀。")

    spot = impr["spot"]
    if spot >= 8:
        lines.append(f"斑點指標提升（+{spot:.0f}%）：代謝啟動，後續 7–14 天通常更有感。")
    else:
        lines.append(f"斑點變化（{spot:.0f}%）：色素改善通常較慢，建議以長期趨勢判讀。")

    return lines

# =========================================================
# 6) Charts (keys set outside)
# =========================================================
def plot_trend(records):
    rows = sorted(records, key=lambda r: (int(r["stage_day"]), r["record_date"]))
    labels = [r["stage_label"] for r in rows]
    avgs = [int(r["avg"]) for r in rows]
    reds = [int(r["metrics"]["redness"]) for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=avgs, name="綜合評分", line=dict(color="#d4af37", width=5), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=labels, y=reds, name="退紅指數", line=dict(color="#e74c3c", width=3, dash="dot"),
                             mode="lines+markers", yaxis="y2"))
    fig.update_layout(
        title="<b>術後恢復趨勢</b>",
        xaxis=dict(title="術後階段", showgrid=False),
        yaxis=dict(title="分數 (越高越好)", range=[0, 100], showgrid=True, gridcolor="#eee"),
        yaxis2=dict(title="退紅指數", overlaying="y", side="right", range=[0, 100], showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        height=320, margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white"
    )
    return fig

def plot_radar(curr):
    cats = ["紋路", "斑點", "退紅度", "毛孔", "平滑"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(curr.values()), theta=cats, fill="toself", name="本次", line_color="#d4af37"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor="rgba(0,0,0,0)"),
        dragmode=False, height=240, margin=dict(t=20, b=20, l=40, r=40),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# =========================================================
# 7) Records: Upsert by (user_id, stage_day)
# =========================================================
def record_exists(user_id: str, stage_day: int) -> bool:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM records WHERE user_id=? AND stage_day=?", (user_id, stage_day))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def upsert_record(user_id: str, record: dict):
    """
    Best practice: DB-level unique constraint, plus upsert logic.
    Also deletes old photo when overwriting to avoid storage bloat.
    """
    conn = db_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute("SELECT id, img_path FROM records WHERE user_id=? AND stage_day=?", (user_id, int(record["stage_day"])))
    row = cur.fetchone()

    if row:
        existing_id, old_img_path = row[0], row[1]

        # delete old photo if different
        try:
            if old_img_path and os.path.exists(old_img_path) and old_img_path != record["img_path"]:
                os.remove(old_img_path)
        except Exception:
            pass

        cur.execute("""
        UPDATE records SET
            stage_label=?, record_date=?, img_path=?,
            metrics_json=?, baseline_json=?, improvement_json=?, quality_json=?,
            symptoms_json=?, tasks_done_pct=?, updated_at=?
        WHERE id=?
        """, (
            record["stage_label"], record["record_date"], record["img_path"],
            json.dumps(record["metrics"], ensure_ascii=False),
            json.dumps(record["baseline_metrics"], ensure_ascii=False),
            json.dumps(record["improvement_pct"], ensure_ascii=False),
            json.dumps(record["quality"], ensure_ascii=False),
            json.dumps(record["symptoms"], ensure_ascii=False),
            int(record["tasks_done_pct"]), now,
            existing_id
        ))
        conn.commit()
        conn.close()
        return "updated", existing_id

    else:
        cur.execute("""
        INSERT INTO records(
            id, user_id, stage_day, stage_label, record_date, img_path,
            metrics_json, baseline_json, improvement_json, quality_json,
            symptoms_json, tasks_done_pct, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record["id"], user_id, int(record["stage_day"]), record["stage_label"], record["record_date"], record["img_path"],
            json.dumps(record["metrics"], ensure_ascii=False),
            json.dumps(record["baseline_metrics"], ensure_ascii=False),
            json.dumps(record["improvement_pct"], ensure_ascii=False),
            json.dumps(record["quality"], ensure_ascii=False),
            json.dumps(record["symptoms"], ensure_ascii=False),
            int(record["tasks_done_pct"]), now
        ))
        conn.commit()
        conn.close()
        return "inserted", record["id"]

def fetch_records(user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT id, stage_day, stage_label, record_date, img_path,
           metrics_json, baseline_json, improvement_json, quality_json,
           symptoms_json, tasks_done_pct
    FROM records
    WHERE user_id=?
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    out = []
    for r in rows:
        rec = {
            "id": r[0],
            "stage_day": int(r[1]),
            "stage_label": r[2],
            "record_date": r[3],
            "img_path": r[4],
            "metrics": json.loads(r[5]),
            "baseline_metrics": json.loads(r[6]),
            "improvement_pct": json.loads(r[7]),
            "quality": json.loads(r[8]),
            "symptoms": json.loads(r[9]),
            "tasks_done_pct": int(r[10]),
        }
        rec["avg"] = metrics_avg(rec["metrics"])
        out.append(rec)
    return out

# =========================================================
# 8) Notifications: store + optional email
# =========================================================
def send_email_smtp(subject: str, body: str):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and NOTIFY_TO):
        return False, "SMTP 未配置：已改為僅寫入系統通知（DB）"

    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = NOTIFY_TO

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [NOTIFY_TO], msg.as_string())
        return True, "診所 Email 已送出"
    except Exception as e:
        return False, f"Email 送出失敗：{e}"

def create_notification(user_id: str, stage_day: int, triage_level: str, payload: dict):
    conn = db_conn()
    cur = conn.cursor()
    nid = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute("""
    INSERT INTO notifications(id, user_id, stage_day, triage_level, payload_json, created_at, status)
    VALUES (?,?,?,?,?,?,?)
    """, (nid, user_id, int(stage_day), triage_level, json.dumps(payload, ensure_ascii=False), now, "created"))

    conn.commit()
    conn.close()
    return nid

def notify_clinic_now(user_id: str, stage_day: int, triage_level: str, payload: dict):
    nid = create_notification(user_id, stage_day, triage_level, payload)

    subject = f"[BeautyTracker通報] 用戶{user_id} {triage_level}｜術後第{stage_day}天"
    body = (
        f"用戶：{user_id}\n"
        f"階段：術後第{stage_day}天\n"
        f"分級：{triage_level}\n"
        f"時間：{datetime.now().isoformat(timespec='seconds')}\n\n"
        f"內容：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )

    ok, msg = send_email_smtp(subject, body)

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE notifications SET status=? WHERE id=?", ("emailed" if ok else "stored", nid))
    conn.commit()
    conn.close()

    return ok, msg

# =========================================================
# 9) Appointments: date restriction + dropdown + list refresh
# =========================================================
def list_slots_for_date(d: date):
    slots = []
    start = datetime(d.year, d.month, d.day, CLINIC_OPEN_HOUR, 0)
    end = datetime(d.year, d.month, d.day, CLINIC_CLOSE_HOUR, 0)
    t = start
    while t < end:
        slots.append(t)
        t += timedelta(minutes=SLOT_MINUTES)
    return slots

def create_appointment(user_id: str, appt_dt: datetime, note: str):
    conn = db_conn()
    cur = conn.cursor()
    appt_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    try:
        cur.execute("""
        INSERT INTO appointments(id, user_id, appt_dt, note, created_at, status)
        VALUES (?,?,?,?,?,?)
        """, (appt_id, user_id, appt_dt.isoformat(timespec="minutes"), note, now, "requested"))
        conn.commit()
        conn.close()
        return True, "預約已送出（等待診所確認）"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "此時段你已經送出過預約（避免重複）"

def fetch_appointments(user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT id, appt_dt, note, created_at, status
    FROM appointments
    WHERE user_id=?
    ORDER BY appt_dt ASC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    return [{
        "id": r[0],
        "appt_dt": r[1],
        "note": r[2] if r[2] else "",
        "created_at": r[3],
        "status": r[4]
    } for r in rows]

# =========================================================
# 10) UI Pages
# =========================================================
def login_page():
    st.title("Beauty Tracker Login（上架級 Demo）")
    st.caption("此版本包含：DB持久化、同天數覆蓋更新、存檔防誤按、症狀通報、升級預約。")
    if st.button("登入測試帳號", type="primary", use_container_width=True):
        st.session_state.logged_in = True
        st.session_state.user_id = DEMO_USER["user_id"]
        st.rerun()

def main_app():
    user_id = st.session_state.user_id
    engine = SkinEngine()

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=70)
        st.title(DEMO_USER["display_name"])
        st.info(f"📋 療程：{DEMO_USER['treatment']}")
        st.caption(f"📅 療程日期：{DEMO_USER['op_date']}")
        st.markdown("---")
        if st.button("安全登出"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.rerun()

    st.markdown(f"## {APP_TITLE}")
    tab1, tab2, tab3 = st.tabs(["🩺 追蹤分析 (Live)", "📊 成效報告/歷史", "📅 預約回診"])

    # -------------------------
    # TAB 1: Live
    # -------------------------
    with tab1:
        st.markdown("### 1) 選擇階段與照片")
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 2])

            with c1:
                stage_label = st.selectbox("術後階段", [s[0] for s in STAGES], index=0)
                stage_day = dict(STAGES)[stage_label]

            with c2:
                f_curr = st.file_uploader("上傳今日照片", type=["jpg", "jpeg", "png"], key="curr")

            with c3:
                bp = before_path(user_id)
                if bp.exists():
                    st.success("✅ 術前圖已鎖定")
                    img_ref = load_image(bp)
                    if st.button("重新設定術前圖（慎用）"):
                        try:
                            bp.unlink(missing_ok=True)
                        except Exception:
                            pass
                        st.rerun()
                else:
                    f_ref = st.file_uploader("上傳術前圖（會鎖定）", type=["jpg", "jpeg", "png"], key="before")
                    img_ref = load_image(f_ref) if f_ref else None
                    consent = st.checkbox("我同意上傳照片用於術後追蹤分析（可隨時要求刪除資料）", value=False)
                    if img_ref is not None and consent and st.button("鎖定為術前圖", type="primary", use_container_width=True):
                        save_rgb_image(img_ref, bp)
                        st.toast("✅ 術前圖已鎖定")
                        time.sleep(0.3)
                        st.rerun()

        if img_ref is None:
            st.info("請先鎖定術前圖，才能計算改善%。")
            return
        if f_curr is None:
            st.info("請上傳今日照片開始分析。")
            return

        img_curr = load_image(f_curr)

        with st.spinner("AI 運算中..."):
            aligned, _ = engine.align_faces(img_curr, img_ref)
            q = quality_check(aligned)
            metrics = engine.analyze(aligned)
            base_metrics = engine.analyze(img_ref)
            impr = {k: improvement_pct(metrics[k], base_metrics[k]) for k in metrics.keys()}
            avg = metrics_avg(metrics)
            base_avg = metrics_avg(base_metrics)
            avg_impr = improvement_pct(avg, base_avg)

        colL, colR = st.columns([1.15, 1.0])

        with colL:
            st.markdown("### 2) 術前/目前對比 + 改善%")
            image_comparison(img1=img_ref, img2=aligned, label1="術前", label2="目前", width=620, in_memory=True)

            if q["score"] >= 80:
                pill_cls = "pill pill-good"; q_text = "拍攝品質：優"
            elif q["score"] >= 60:
                pill_cls = "pill pill-warn"; q_text = "拍攝品質：可"
            else:
                pill_cls = "pill pill-bad"; q_text = "拍攝品質：需重拍"

            st.markdown(
                f'<div><span class="{pill_cls}">{q_text}</span> '
                f'<small>（亮度 {q["brightness"]} / 清晰度 {q["sharpness"]}）'
                f'{"｜問題：" + "、".join(q["tags"]) if q["tags"] else ""}</small></div>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)

            def metric_card(col, title, val, pct):
                sign = "+" if pct >= 0 else ""
                col.markdown(
                    f"""
                    <div style="text-align:center; padding:10px; border:1px solid #eee; border-radius:12px; background:white;">
                      <div class="metric-val">{val}</div>
                      <div class="metric-title">{title}</div>
                      <div class="metric-sub">改善 {sign}{pct:.0f}%（相對術前）</div>
                    </div>
                    """, unsafe_allow_html=True
                )

            metric_card(k1, "退紅指數", metrics["redness"], impr["redness"])
            metric_card(k2, "斑點指數", metrics["spot"], impr["spot"])
            metric_card(k3, "綜合評分", avg, avg_impr)

            st.caption("註：改善%以術前為基準；若拍攝品質不佳，改善%僅供趨勢參考。")

            st.markdown("### 3) 雷達圖")
            st.plotly_chart(plot_radar(metrics), use_container_width=True, key="radar_live")

        with colR:
            st.markdown("### 👩‍⚕️ 術後照護面板（上架級）")

            # A) Symptoms + triage + notify button
            with st.container(border=True):
                st.markdown("#### A. 症狀回報（30 秒）")
                c1, c2 = st.columns(2)
                with c1:
                    pain = st.slider("疼痛程度", 0, 10, 2)
                    heat = st.slider("灼熱/熱感", 0, 10, 2)
                with c2:
                    swelling = st.slider("腫脹程度", 0, 10, 2)
                    oozing = st.checkbox("是否有滲液/水泡/明顯滲出？", value=False)
                    fever = st.checkbox("是否有發燒或全身不適？", value=False)

                triage_level, triage_msg, triage_pill = triage_from_symptoms(pain, heat, swelling, oozing, fever)
                st.markdown(f'<div><span class="pill {triage_pill}">風險分級：{triage_level}</span></div>', unsafe_allow_html=True)
                st.write(triage_msg)

                if triage_level in ("黃燈", "紅燈"):
                    payload = {
                        "user_id": user_id,
                        "stage_label": stage_label,
                        "stage_day": stage_day,
                        "triage_level": triage_level,
                        "symptoms": {"pain": pain, "heat": heat, "swelling": swelling, "oozing": oozing, "fever": fever},
                        "quality": q,
                        "improvement_pct": {k: round(float(impr[k]), 2) for k in impr},
                        "time": datetime.now().isoformat(timespec="seconds")
                    }
                    confirm_notify = st.checkbox("我同意將本次狀況通報診所，以便診所立即致電關心。", value=False)
                    if st.button("🚨 立即通報診所（請求回電關懷）", type="primary", use_container_width=True, disabled=not confirm_notify):
                        ok, msg = notify_clinic_now(user_id, stage_day, triage_level, payload)
                        st.toast("✅ 通報已送出")
                        st.info(msg)

            # B) Tasks checklist
            with st.container(border=True):
                st.markdown("#### B. 今日照護任務清單")
                tasks = stage_tasks(stage_day)
                t_key = f"tasks_{stage_day}"
                if t_key not in st.session_state:
                    st.session_state[t_key] = {t: False for t in tasks}

                done = 0
                for t in tasks:
                    st.session_state[t_key][t] = st.checkbox(t, value=st.session_state[t_key].get(t, False))
                    if st.session_state[t_key][t]:
                        done += 1

                total = max(1, len(tasks))
                st.progress(done / total)
                tasks_done_pct = int(done / total * 100)
                st.write(f"今日完成度：{tasks_done_pct}%")

            # C) Nurse summary
            with st.container(border=True):
                st.markdown("#### C. AI 護理師結論（可理解、可行動）")
                for ln in explain_improvements(impr, stage_day, q):
                    st.write(f"- {ln}")

            # D) Save record: Best anti-misclick flow
            with st.container(border=True):
                st.markdown("#### D. 存入病歷（防誤按）")

                will_overwrite = record_exists(user_id, stage_day)
                if will_overwrite:
                    st.warning("此術後階段已存在紀錄；本次存入將【覆蓋更新】原紀錄（不會重複新增）。")

                if not q["ok"]:
                    st.warning("照片品質不足，建議重拍後再存檔（避免改善%失真）。")

                confirm_save = st.checkbox(
                    "我已確認照片與術後階段無誤，且同意存入病歷（包含覆蓋更新）。",
                    value=False
                )

                # 2-step confirmation
                if "save_armed" not in st.session_state:
                    st.session_state.save_armed = False
                if "save_armed_until" not in st.session_state:
                    st.session_state.save_armed_until = 0.0

                if time.time() > st.session_state.save_armed_until:
                    st.session_state.save_armed = False

                btn_label = "① 先按此鍵進入確認（防誤按）" if not st.session_state.save_armed else "② 確認存入病歷（立即寫入）"
                btn_disabled = (not q["ok"]) or (not confirm_save)

                if st.button(btn_label, type="primary", use_container_width=True, disabled=btn_disabled):
                    if not st.session_state.save_armed:
                        st.session_state.save_armed = True
                        st.session_state.save_armed_until = time.time() + 12
                        st.info("已進入確認狀態：請在 12 秒內再按一次完成寫入。")
                    else:
                        rec_id = str(uuid.uuid4())
                        img_path = user_dir(user_id) / "records" / f"{rec_id}.jpg"
                        save_rgb_image(aligned, img_path)

                        record = {
                            "id": rec_id,
                            "stage_day": stage_day,
                            "stage_label": stage_label,
                            "record_date": date.today().isoformat(),
                            "img_path": str(img_path),
                            "metrics": metrics,
                            "baseline_metrics": base_metrics,
                            "improvement_pct": {k: round(float(impr[k]), 2) for k in impr},
                            "quality": q,
                            "symptoms": {"pain": pain, "heat": heat, "swelling": swelling, "oozing": oozing, "fever": fever, "triage": triage_level},
                            "tasks_done_pct": tasks_done_pct,
                        }

                        action, _ = upsert_record(user_id, record)

                        st.session_state.save_armed = False
                        st.session_state.save_armed_until = 0.0

                        st.toast("✅ 已更新該術後階段紀錄（避免重複）" if action == "updated" else "✅ 已新增病歷")
                        time.sleep(0.3)
                        st.rerun()

    # -------------------------
    # TAB 2: Report / History
    # -------------------------
    with tab2:
        records = fetch_records(user_id)
        if not records:
            st.info("尚無歷史數據。請先在 Live 頁存入一筆病歷。")
        else:
            st.markdown("### 📈 成效趨勢（自動排序）")
            st.plotly_chart(plot_trend(records), use_container_width=True, key="trend_chart")

            st.markdown("### 🗂️ 成效報告（單一階段僅保留一筆）")
            rows = sorted(records, key=lambda r: (r["stage_day"], r["record_date"]), reverse=True)

            for rec in rows:
                with st.container(border=True):
                    c1, c2 = st.columns([1.2, 2.0])
                    with c1:
                        if rec.get("img_path") and os.path.exists(rec["img_path"]):
                            st.image(rec["img_path"], caption=f"{rec['stage_label']}｜{rec['record_date']}")
                        else:
                            st.info("照片檔案不存在")

                    with c2:
                        q = rec["quality"]
                        q_score = int(q.get("score", 0))
                        if q_score >= 80:
                            pill_cls = "pill pill-good"; q_text = "品質：優"
                        elif q_score >= 60:
                            pill_cls = "pill pill-warn"; q_text = "品質：可"
                        else:
                            pill_cls = "pill pill-bad"; q_text = "品質：弱"

                        st.markdown(f"**{rec['stage_label']}**  <span class='{pill_cls}'>{q_text} {q_score}</span>", unsafe_allow_html=True)

                        imp = rec["improvement_pct"]
                        st.write(f"- 綜合評分：{rec['avg']}")
                        st.write(f"- 退紅改善：{imp.get('redness', 0)}%｜斑點改善：{imp.get('spot', 0)}%｜紋理改善：{imp.get('wrinkle', 0)}%")
                        sym = rec.get("symptoms", {})
                        st.write(f"- 風險分級：{sym.get('triage','-')}｜疼痛 {sym.get('pain','-')}｜熱感 {sym.get('heat','-')}｜腫脹 {sym.get('swelling','-')}")
                        st.write(f"- 今日照護完成度：{rec.get('tasks_done_pct', 0)}%")

                        st.plotly_chart(plot_radar(rec["metrics"]), use_container_width=True, key=f"radar_{rec['id']}")

    # -------------------------
    # TAB 3: Appointment (Upgraded)
    # -------------------------
    with tab3:
        st.subheader("📅 預約回診（升級版）")
        st.caption("限制：今天以前不可選；年份僅今年與明年。時段以『下拉式』選擇。")

        today = date.today()
        max_day = date(today.year + 1, 12, 31)

        appt_date = st.date_input(
            "選擇日期",
            value=today + timedelta(days=7),
            min_value=today,
            max_value=max_day
        )

        note = st.text_input("備註（可選）", value="術後追蹤回診")

        # Build slots and filter past times if same day
        slots = list_slots_for_date(appt_date)
        now_dt = datetime.now()
        if appt_date == now_dt.date():
            slots = [t for t in slots if t > now_dt]

        if not slots:
            st.warning("此日期已無可預約時段，請選擇其他日期。")
        else:
            slot_labels = [t.strftime("%H:%M") for t in slots]
            selected_label = st.selectbox("選擇可預約時段", slot_labels, index=0)
            chosen = slots[slot_labels.index(selected_label)]

            st.markdown(f"已選擇：**{chosen.strftime('%Y-%m-%d %H:%M')}**")

            confirm_appt = st.checkbox("我確認要送出此預約時段", value=False)
            if st.button("送出預約", type="primary", disabled=not confirm_appt):
                ok, msg = create_appointment(user_id, chosen, note)
                if ok:
                    st.success(msg)
                    time.sleep(0.2)
                    st.rerun()
                else:
                    st.warning(msg)

        st.markdown("#### 我的預約清單")
        appts = fetch_appointments(user_id)
        if not appts:
            st.info("目前沒有預約。")
        else:
            for a in appts:
                st.write(f"- {a['appt_dt']}｜狀態：{a['status']}｜備註：{a.get('note','')}")

# =========================================================
# 11) Run
# =========================================================
if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()
