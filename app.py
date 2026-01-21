import os
import time
import uuid
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import streamlit as st

# Optional imports (keep app runnable even if some libs missing)
try:
    import numpy as np
    import cv2
    from PIL import Image
except Exception:
    np = None
    cv2 = None
    Image = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from streamlit_image_comparison import image_comparison
except Exception:
    image_comparison = None


# =========================================================
# 0) Basic setup
# =========================================================
APP_TITLE = "美麗追蹤者 Beauty Tracker"
DB_PATH = "beauty_tracker.db"
DATA_DIR = "user_data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }
.stApp { background-color: #fcfcfc; }
.card { border: 1px solid #e8e8e8; border-radius: 14px; padding: 14px; background: white; }
.small { color: #666; font-size: 12px; }
.hint { color: #333; font-size: 14px; line-height: 1.5; }
.badge-ok { color: #2e7d32; font-weight: 800; }
.badge-warn { color: #f57f17; font-weight: 800; }
.badge-bad { color: #c62828; font-weight: 900; }
.metric-row { display:flex; gap:12px; flex-wrap:wrap; }
.metric-box { flex:1; min-width:210px; border:1px solid #eee; border-radius:12px; padding:12px; background:white; }
.metric-title{ font-size:13px; color:#666; margin-bottom:4px; }
.metric-val{ font-size:26px; font-weight:900; color:#222; line-height:1.1; }
.metric-sub{ font-size:12px; color:#666; margin-top:4px; }
hr { border: none; border-top: 1px solid #eee; margin: 12px 0; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 1) DB + migration (auto add missing columns)
# =========================================================
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table: str) -> set:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return set([r["name"] for r in rows])


def ensure_columns(conn, table: str, columns_sql: dict):
    """
    columns_sql example: {"name":"TEXT", "op_date":"TEXT"}
    """
    existing = table_columns(conn, table)
    cur = conn.cursor()
    for col, col_type in columns_sql.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
    conn.commit()


def db_init_and_migrate():
    conn = db_conn()
    cur = conn.cursor()

    # Create tables (if not exist)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            phone TEXT,
            name TEXT,
            treatment TEXT,
            op_date TEXT,
            before_img_path TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            record_date TEXT NOT NULL,
            postop_date TEXT,
            uploaded_at TEXT,
            img_path TEXT,
            q_score INTEGER,
            confidence INTEGER,
            wrinkle INTEGER,
            spot INTEGER,
            redness INTEGER,
            pore INTEGER,
            texture INTEGER,
            note TEXT,
            UNIQUE(user_id, stage),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            appt_dt TEXT NOT NULL,
            note TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, appt_dt),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            stage TEXT,
            severity TEXT,
            reason TEXT,
            symptoms TEXT,
            metrics_json TEXT,
            img_path TEXT,
            resolved INTEGER DEFAULT 0,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    conn.commit()

    # Migrate/add missing columns safely (for old DBs)
    ensure_columns(conn, "users", {
        "phone": "TEXT",
        "name": "TEXT",
        "treatment": "TEXT",
        "op_date": "TEXT",
        "before_img_path": "TEXT",
    })
    ensure_columns(conn, "records", {
        "q_score": "INTEGER",
        "confidence": "INTEGER",
        "note": "TEXT",
        "wrinkle": "INTEGER",
        "spot": "INTEGER",
        "redness": "INTEGER",
        "pore": "INTEGER",
        "texture": "INTEGER",
        "img_path": "TEXT",
        "record_date": "TEXT",
        "postop_date": "TEXT",
        "uploaded_at": "TEXT",
    })
    ensure_columns(conn, "appointments", {
        "note": "TEXT",
        "status": "TEXT",
        "created_at": "TEXT",
    })
    ensure_columns(conn, "alerts", {
        "stage": "TEXT",
        "severity": "TEXT",
        "reason": "TEXT",
        "symptoms": "TEXT",
        "metrics_json": "TEXT",
        "img_path": "TEXT",
        "resolved": "INTEGER",
    })

    conn.close()


db_init_and_migrate()


# =========================================================
# 2) Demo user (for your project)
# =========================================================
def ensure_demo_user():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE user_id=?", ("0912345678",))
    row = cur.fetchone()
    if not row:
        cur.execute(
            """
            INSERT INTO users (user_id, phone, name, treatment, op_date, before_img_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "0912345678",
                "0912345678",
                "王小美 (VIP)",
                "皮秒雷射 + 蜂巢探頭",
                str(date.today() - timedelta(days=1)),
                None,
            ),
        )
    else:
        # Make sure demo user has essential fields
        cur.execute(
            """
            UPDATE users
            SET name=COALESCE(name, ?),
                treatment=COALESCE(treatment, ?),
                op_date=COALESCE(op_date, ?),
                phone=COALESCE(phone, ?)
            WHERE user_id=?
            """,
            ("王小美 (VIP)", "皮秒雷射 + 蜂巢探頭", str(date.today() - timedelta(days=1)), "0912345678", "0912345678")
        )
    conn.commit()
    conn.close()


ensure_demo_user()


def get_user(user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def set_before_img(user_id: str, path: str | None):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET before_img_path=? WHERE user_id=?", (path, user_id))
    conn.commit()
    conn.close()


# =========================================================
# 3) Helpers (images, metrics, quality, confidence)
# =========================================================
STAGES = [
    "術後第 1 天",
    "術後第 2 天",
    "術後第 3 天",
    "術後第 7 天",
    "術後第 14 天",
    "術後第 30 天",
    "術後 30 天以上",
]


def stage_order(stage: str) -> int:
    if stage in STAGES:
        return STAGES.index(stage)
    return 999



def stage_to_days(stage: str):
    """Extract N from '術後第 N 天'. Return None for non-fixed stages (e.g., '術後 30 天以上')."""
    if not stage:
        return None
    m = re.search(r"第\s*(\d+)\s*天", stage)
    if m:
        return int(m.group(1))
    return None


def save_rgb_image(rgb_np, prefix="img") -> str:
    ts = int(time.time() * 1000)
    fname = f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(DATA_DIR, fname)
    Image.fromarray(rgb_np).save(path, quality=95)
    return path


def load_image_rgb(file_or_path):
    if file_or_path is None:
        return None
    if isinstance(file_or_path, str):
        if not os.path.exists(file_or_path):
            return None
        img = Image.open(file_or_path).convert("RGB")
    else:
        img = Image.open(file_or_path).convert("RGB")
    return np.array(img)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def metrics_avg(m: dict) -> int:
    if not m:
        return 0
    return int(round(sum(m.values()) / len(m)))


def improvement_pct(curr: int, base: int) -> int:
    # score higher = better, so percent = (curr-base)/base
    if base is None or base <= 0:
        return 0
    return int(round(((curr - base) / base) * 100))


@dataclass
class QualityResult:
    score: int
    brightness: int
    sharpness: int
    framing: int
    tips: str


def quality_check(rgb_img) -> QualityResult:
    """
    Simple, fast quality scoring (0-100):
      - brightness: too dark/bright penalty
      - sharpness: Laplacian variance
      - framing: center-edge ratio proxy (no heavy face detector)
    """
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    mean_b = int(np.mean(gray))
    bright_score = 100 - int(abs(mean_b - 145) * 1.2)
    bright_score = clamp(bright_score, 0, 100)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    sharp_score = int(clamp((var / 180.0) * 100, 0, 100))

    edges = cv2.Canny(gray, 50, 150)
    cx0, cx1 = int(w * 0.33), int(w * 0.67)
    cy0, cy1 = int(h * 0.33), int(h * 0.67)
    center = edges[cy0:cy1, cx0:cx1]
    framing_ratio = (np.sum(center) + 1) / (np.sum(edges) + 1)
    framing_score = int(clamp((framing_ratio / 0.55) * 100, 0, 100))

    score = int(round(0.35 * bright_score + 0.40 * sharp_score + 0.25 * framing_score))

    tips = []
    if bright_score < 60:
        tips.append("光線不佳：請面向窗戶或白光、避免背光。")
    if sharp_score < 60:
        tips.append("畫面偏糊：擦拭鏡頭、手肘靠桌、對焦臉部。")
    if framing_score < 55:
        tips.append("構圖偏移：臉置中、保持正臉，避免太近或太遠。")
    if not tips:
        tips.append("拍攝品質良好。")

    return QualityResult(score, bright_score, sharp_score, framing_score, " ".join(tips))


class SkinEngine:
    def _normalize_lighting(self, src_rgb, ref_rgb):
        # 讓目前照與術前照的亮度分布更接近（只調 L，不動色相）
        src = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        sL, sA, sB = cv2.split(src)
        rL, _, _ = cv2.split(ref)

        s_mean, s_std = cv2.meanStdDev(sL)
        r_mean, r_std = cv2.meanStdDev(rL)

        # meanStdDev 回傳的是 (1,1) array，要取出純數字
        s_mean = float(s_mean[0][0])
        s_std  = float(s_std[0][0])
        r_mean = float(r_mean[0][0])
        r_std  = float(r_std[0][0])

        s_std = max(1e-6, s_std)
        r_std = max(1e-6, r_std)

        L = (sL - s_mean) * (r_std / s_std) + r_mean
        L = np.clip(L, 0, 255)

        merged = cv2.merge([L, sA, sB]).astype(np.uint8)
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


    def align_faces(self, src_img_rgb, ref_img_rgb):
        """
        ORB + RANSAC affine registration.
        Returns (aligned_rgb, success(bool), inlier_ratio(float))
        """
        H, W = ref_img_rgb.shape[:2]
        src_resized = cv2.resize(src_img_rgb, (W, H))

        g1 = cv2.cvtColor(ref_img_rgb, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(src_resized, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=1200)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)

        if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
            aligned = self._normalize_lighting(src_resized, ref_img_rgb)
            return aligned, False, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)
        good = matches[:140]

        if len(good) < 25:
            aligned = self._normalize_lighting(src_resized, ref_img_rgb)
            return aligned, False, 0.0

        pts_ref = np.float32([k1[m.queryIdx].pt for m in good])
        pts_src = np.float32([k2[m.trainIdx].pt for m in good])

        M, inliers = cv2.estimateAffinePartial2D(
            pts_src, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if M is None or inliers is None:
            aligned = self._normalize_lighting(src_resized, ref_img_rgb)
            return aligned, False, 0.0

        inlier_ratio = float(np.mean(inliers))
        if inlier_ratio < 0.25:
            aligned = self._normalize_lighting(src_resized, ref_img_rgb)
            return aligned, False, inlier_ratio

        aligned = cv2.warpAffine(
            src_resized, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        aligned = self._normalize_lighting(aligned, ref_img_rgb)
        return aligned, True, inlier_ratio

    def analyze(self, rgb_img):
        """
        Lightweight demo metrics (0-100): redness, spots, wrinkles, pores, texture
        """
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)

        # redness: inverse of A channel shift
        mean_a = float(np.mean(lab[:, :, 1]))
        red_score = 100 - (mean_a - 128) * 4.0
        redness = clamp(int(red_score), 20, 99)

        # spots: adaptive threshold area
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25, 10
        )
        spot_score = 100 - (float(np.sum(thresh)) / float(thresh.size)) * 200
        spot = clamp(int(spot_score), 40, 95)

        # texture proxies
        edges = cv2.Canny(gray, 50, 150)
        wrinkle = clamp(int(100 - (float(np.sum(edges)) / float(edges.size)) * 500), 40, 99)
        pore = clamp(int(100 - (float(np.sum(edges)) / float(edges.size)) * 300), 40, 99)
        texture = int(round((wrinkle + pore) / 2))

        return {
            "wrinkle": wrinkle,
            "spot": spot,
            "redness": redness,
            "pore": pore,
            "texture": texture,
        }


def compute_confidence(align_success: bool, inlier_ratio: float, q_score: int) -> int:
    base = 55
    base += int(round((q_score - 60) * 0.6))
    if align_success:
        base += 15
        base += int(round(inlier_ratio * 20))
    else:
        base -= 10
    return clamp(base, 10, 98)


def badge_conf(conf: int) -> str:
    if conf >= 80:
        return "badge-ok"
    if conf >= 60:
        return "badge-warn"
    return "badge-bad"


def conf_label(conf: int) -> str:
    if conf >= 80:
        return "可信度高"
    if conf >= 60:
        return "可信度中"
    return "可信度低（建議重拍）"


# =========================================================
# 4) Records (upsert) + fetch
# =========================================================
def upsert_record(user_id: str, stage: str, op_date: str | None, img_path: str,
                  q_score: int, confidence: int, metrics: dict, note: str = ""):
    """Upsert one record per (user_id, stage).
    Stores BOTH:
      - postop_date: computed from op_date + stage (if stage is fixed-day)
      - uploaded_at: actual save timestamp (user may upload late)
    For backward compatibility, record_date is set to postop_date if available, else today's date.
    """
    conn = db_conn()
    cur = conn.cursor()

    rec_id = uuid.uuid4().hex
    uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compute postop_date from surgery date + day offset
    postop_date = None
    try:
        if op_date:
            d = stage_to_days(stage)
            if d is not None:
                base = datetime.strptime(op_date, "%Y-%m-%d").date()
                postop_date = (base + timedelta(days=d)).isoformat()
    except Exception:
        postop_date = None

    record_date = postop_date or str(date.today())

    cur.execute(
        """
        INSERT INTO records (
            id, user_id, stage, record_date, postop_date, uploaded_at, img_path, q_score, confidence,
            wrinkle, spot, redness, pore, texture, note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, stage) DO UPDATE SET
            record_date=excluded.record_date,
            postop_date=excluded.postop_date,
            uploaded_at=excluded.uploaded_at,
            img_path=excluded.img_path,
            q_score=excluded.q_score,
            confidence=excluded.confidence,
            wrinkle=excluded.wrinkle,
            spot=excluded.spot,
            redness=excluded.redness,
            pore=excluded.pore,
            texture=excluded.texture,
            note=excluded.note
        """,
        (
            rec_id, user_id, stage, record_date, postop_date, uploaded_at, img_path, q_score, confidence,
            int(metrics["wrinkle"]), int(metrics["spot"]), int(metrics["redness"]),
            int(metrics["pore"]), int(metrics["texture"]), note
        )
    )
    conn.commit()
    conn.close()


def fetch_records(user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM records WHERE user_id=?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    recs = [dict(r) for r in rows]
    recs.sort(key=lambda r: stage_order(r.get("stage", "")))
    return recs


# =========================================================
# 5) Alerts
# =========================================================
def create_alert(user_id: str, stage: str, severity: str, reason: str, symptoms: str, metrics: dict, img_path: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO alerts (id, user_id, created_at, stage, severity, reason, symptoms, metrics_json, img_path, resolved)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            uuid.uuid4().hex,
            user_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            stage,
            severity,
            reason,
            symptoms,
            json.dumps(metrics, ensure_ascii=False),
            img_path,
        )
    )
    conn.commit()
    conn.close()


def fetch_alerts(limit=30):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.*, u.name, u.treatment
        FROM alerts a
        LEFT JOIN users u ON u.user_id=a.user_id
        ORDER BY a.created_at DESC
        LIMIT ?
        """,
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =========================================================
# 6) Appointments
# =========================================================
def create_appointment(user_id: str, appt_dt: str, note: str):
    conn = db_conn()
    cur = conn.cursor()
    appt_id = uuid.uuid4().hex
    try:
        cur.execute(
            """
            INSERT INTO appointments (id, user_id, appt_dt, note, status, created_at)
            VALUES (?, ?, ?, ?, 'requested', ?)
            """,
            (appt_id, user_id, appt_dt, note, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return True, "預約已送出（待診所確認）"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "此時段已送出過預約（避免重複）"


def fetch_appointments(user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM appointments
        WHERE user_id=?
        ORDER BY appt_dt ASC
        """,
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def cancel_appointment(appt_id: str, user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE appointments
        SET status='cancelled'
        WHERE id=? AND user_id=? AND status IN ('requested','confirmed')
        """,
        (appt_id, user_id)
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


# =========================================================
# 7) Charts
# =========================================================
def plot_trend(records):
    if go is None or not records:
        return None
    x = [r["stage"] for r in records]
    avg_scores = [
        metrics_avg({
            "wrinkle": r.get("wrinkle", 0),
            "spot": r.get("spot", 0),
            "redness": r.get("redness", 0),
            "pore": r.get("pore", 0),
            "texture": r.get("texture", 0),
        })
        for r in records
    ]
    reds = [r.get("redness", 0) for r in records]
    confs = [r.get("confidence", 0) for r in records]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=avg_scores, name="綜合分數", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=x, y=reds, name="退紅指數", mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Bar(x=x, y=confs, name="可信度", yaxis="y3", opacity=0.35))

    fig.update_layout(
        title="術後恢復趨勢",
        height=380,
        xaxis=dict(title="術後階段"),
        yaxis=dict(title="綜合分數", range=[0, 100]),
        yaxis2=dict(title="退紅指數", overlaying="y", side="right", range=[0, 100]),
        yaxis3=dict(title="可信度", anchor="free", overlaying="y", side="right", position=0.95, range=[0, 100]),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified"
    )
    return fig


def plot_radar(m):
    if go is None or not m:
        return None
    cats = ["紋路", "斑點", "退紅", "毛孔", "平滑"]
    vals = [m["wrinkle"], m["spot"], m["redness"], m["pore"], m["texture"]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name="本次"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=260,
        margin=dict(t=20, b=20, l=40, r=40)
    )
    return fig


# =========================================================
# 8) Session / Login
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None


def login_page():
    st.title("Beauty Tracker Login")
    st.caption("（專題展示用：一鍵登入測試帳號）")

    if st.button("登入測試帳號（VIP）", type="primary", use_container_width=True):
        st.session_state.logged_in = True
        st.session_state.user_id = "0912345678"
        st.rerun()

    st.markdown("---")
    st.markdown("上架版可改為：簡訊 OTP 驗證 + 隱私/同意書流程 + 真正診所端後台。")


# =========================================================
# 9) Main app
# =========================================================
def main_app():
    if np is None or cv2 is None or Image is None:
        st.error("此版本需要：opencv-python-headless、numpy、Pillow。請確認 requirements.txt 已安裝。")
        st.stop()

    user = get_user(st.session_state.user_id)
    if not user:
        st.error("使用者不存在")
        st.stop()

    engine = SkinEngine()

    # Sidebar
    with st.sidebar:
        st.title(user.get("name", "VIP 客戶"))
        st.caption(f"療程：{user.get('treatment', '—')}")
        st.caption(f"療程日期：{user.get('op_date', '—')}")
        st.markdown("---")

        alerts = fetch_alerts(limit=20)
        unresolved = [a for a in alerts if int(a.get("resolved", 0) or 0) == 0]
        st.markdown(f"**診所通報未處理：{len(unresolved)}**")

        if st.button("安全登出", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.rerun()

    st.markdown(f"## {APP_TITLE}")

    tab1, tab2, tab3, tab4 = st.tabs(["🩺 術後追蹤（核心）", "📊 成效報告", "📅 預約回診", "🏥 診所通報（Demo）"])

    # -----------------------------------------------------
    # Tab1: Post-op tracking (BEST)
    # -----------------------------------------------------
    with tab1:
        st.markdown(
            """
<div class="card">
  <div class="hint"><b>拍攝指引（讓成效更準、更像真實醫美服務）</b><br/>
  1) 面向窗戶或白光、避免背光　2) 正臉、眼睛水平　3) 距離約 30–40 cm　4) 不用濾鏡/美肌　5) 背景盡量純色</div>
</div>
""",
            unsafe_allow_html=True,
        )

        cA, cB, cC = st.columns([1.6, 1.8, 1.2])

        stage = cA.selectbox("術後階段", STAGES, index=0)
        record_note = cA.text_input("備註（選填）", placeholder="例如：今天有上修復霜/戶外曝曬...")

        curr_file = cB.file_uploader("上傳今日照片（正臉）", type=["jpg", "jpeg", "png"])

        cC.markdown("**術前照片（Baseline）**")
        before_path = user.get("before_img_path")
        if before_path and os.path.exists(before_path):
            cC.success("✅ 已鎖定術前圖")
            if cC.button("重新上傳術前圖", use_container_width=True):
                set_before_img(user["user_id"], None)
                st.rerun()
            before_file = None
        else:
            before_file = cC.file_uploader("上傳術前照片", type=["jpg", "jpeg", "png"])

        img_ref = load_image_rgb(before_path) if (before_path and os.path.exists(before_path)) else load_image_rgb(before_file)
        if (not before_path or not os.path.exists(before_path)) and img_ref is not None:
            # Save and lock baseline
            path = save_rgb_image(img_ref, prefix=f"before_{user['user_id']}")
            set_before_img(user["user_id"], path)
            user = get_user(st.session_state.user_id)
            st.toast("✅ 術前圖已鎖定")

        img_curr = load_image_rgb(curr_file) if curr_file else None

        st.markdown("---")

        if img_curr is None or img_ref is None:
            st.info("請先上傳「術前照片」與「今日照片」，系統才會產生改善%與成效報告。")
        else:
            with st.spinner("AI 分析中..."):
                aligned, align_ok, inlier_ratio = engine.align_faces(img_curr, img_ref)
                q = quality_check(aligned)
                conf = compute_confidence(align_ok, inlier_ratio, q.score)

                base_metrics = engine.analyze(img_ref)
                curr_metrics = engine.analyze(aligned)

                base_avg = metrics_avg(base_metrics)
                curr_avg = metrics_avg(curr_metrics)

            # always show pct (BEST for demo), but tag when low confidence
            low_conf = (conf < 60) or (q.score < 55)
            pct_tag = "（建議同光源重拍）" if low_conf else ""

            left, right = st.columns([1.15, 0.85])

            with left:
                st.markdown("### 1) 前後對比（自動校正角度/尺寸/光線）")
                if image_comparison is not None:
                    image_comparison(img1=img_ref, img2=aligned, label1="術前", label2="目前（已校正）", width=720, in_memory=True)
                else:
                    st.image(img_ref, caption="術前", width=340)
                    st.image(aligned, caption="目前（已校正）", width=340)

                st.markdown("---")
                st.markdown("### 2) 拍攝品質與可信度（像真實醫療系統）")
                b_class = badge_conf(conf)
                st.markdown(
                    f"""
<div class="card">
  <div><b>拍攝品質：</b> {q.score}/100　<span class="small">(亮度 {q.brightness}｜清晰 {q.sharpness}｜構圖 {q.framing})</span></div>
  <div class="small">{q.tips}</div>
  <hr/>
  <div><b>分析可信度：</b> <span class="{b_class}">{conf_label(conf)}（{conf}/100）</span></div>
  <div class="small">可信度低仍會顯示改善%，但會加註提醒，避免誤判。</div>
</div>
""",
                    unsafe_allow_html=True
                )

            with right:
                st.markdown("### 3) 成效摘要（客人最有感）")

                avg_impr = improvement_pct(curr_avg, base_avg)
                red_impr = improvement_pct(curr_metrics["redness"], base_metrics["redness"])

                st.markdown(
                    f"""
<div class="metric-row">
  <div class="metric-box">
    <div class="metric-title">綜合分數</div>
    <div class="metric-val">{curr_avg}/100</div>
    <div class="metric-sub">改善：{avg_impr:+d}% {pct_tag}</div>
  </div>
  <div class="metric-box">
    <div class="metric-title">退紅指數</div>
    <div class="metric-val">{curr_metrics['redness']}/100</div>
    <div class="metric-sub">改善：{red_impr:+d}% {pct_tag}</div>
  </div>
</div>
""",
                    unsafe_allow_html=True
                )

                st.markdown("---")
                st.markdown("### 4) 分項改善（%）")
                if low_conf:
                    st.warning("本次照片條件可能影響精準度：改善%仍顯示，但建議依拍攝指引重拍一次以提高可信度。")

                def pct_text(curr, base):
                    return f"{improvement_pct(curr, base):+d}% {pct_tag}"

                st.write(f"紋路：{pct_text(curr_metrics['wrinkle'], base_metrics['wrinkle'])}")
                st.write(f"斑點：{pct_text(curr_metrics['spot'], base_metrics['spot'])}")
                st.write(f"毛孔：{pct_text(curr_metrics['pore'], base_metrics['pore'])}")
                st.write(f"平滑：{pct_text(curr_metrics['texture'], base_metrics['texture'])}")

                st.markdown("---")
                st.markdown("### 5) AI 護理師（更像真服務）")

                red = curr_metrics["redness"]
                severe_flag = False
                advice = []

                # stage-based
                if stage in ("術後第 1 天", "術後第 2 天", "術後第 3 天"):
                    advice.append("目前屬正常術後反應期：加強保濕、避免高溫環境與劇烈運動。")
                    advice.append("建議每 2–3 小時補一次修復保濕，外出務必防曬。")
                    if red < 55:
                        severe_flag = True
                        advice.append("紅腫指數偏低：可能反應較強，建議加強冰敷並主動回報。")
                elif stage == "術後第 7 天":
                    advice.append("進入代謝/結痂期：請勿摳除，洗臉輕柔，外出加強防曬。")
                elif stage in ("術後第 14 天", "術後第 30 天"):
                    advice.append("進入穩定期：持續修復、防曬與作息，能讓成效維持更久。")
                else:
                    advice.append("膚況大致穩定：依醫師建議規劃保養型維持療程。")

                if low_conf:
                    advice.append("本次拍攝條件可能影響判讀：建議在同光源、同距離、同角度重拍以提高準確性。")

                st.markdown('<div class="card">', unsafe_allow_html=True)
                for s in advice:
                    st.write("• " + s)
                if severe_flag:
                    st.markdown('<div class="badge-bad">⚠ 系統判定可能需要協助</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 6) 症狀回報（讓客人覺得真的有人在看）")
            s1, s2, s3, s4, s5 = st.columns(5)
            sym_red = s1.checkbox("紅腫明顯")
            sym_pain = s2.checkbox("刺痛/灼熱")
            sym_itch = s3.checkbox("搔癢")
            sym_peel = s4.checkbox("脫皮/緊繃")
            sym_ooze = s5.checkbox("滲液/疑似感染")

            sym_note = st.text_area("補充描述（選填）", placeholder="例如：下午開始刺痛、紅腫擴大...")

            symptoms_list = []
            if sym_red: symptoms_list.append("紅腫明顯")
            if sym_pain: symptoms_list.append("刺痛/灼熱")
            if sym_itch: symptoms_list.append("搔癢")
            if sym_peel: symptoms_list.append("脫皮/緊繃")
            if sym_ooze: symptoms_list.append("滲液/疑似感染")
            if sym_note.strip():
                symptoms_list.append("備註：" + sym_note.strip())
            symptoms_text = "；".join(symptoms_list) if symptoms_list else ""

            st.markdown("---")
            st.markdown("### 7) 存入病歷（同階段覆蓋更新 + 防誤按）")

            # Save preview image so saved record matches what user saw
            preview_path = save_rgb_image(aligned, prefix=f"rec_{user['user_id']}")

            save_confirm = st.checkbox("我確認：這是我要存入的照片與術後階段（同一階段會覆蓋更新）", value=False)
            save_btn = st.button("💾 存入病歷（含照片）", type="primary", use_container_width=True, disabled=not save_confirm)

            if save_btn:
                upsert_record(
                    user_id=user["user_id"],
                    stage=stage,
                    op_date=user.get("op_date"),
                    img_path=preview_path,
                    q_score=int(q.score),
                    confidence=int(conf),
                    metrics=curr_metrics,
                    note=record_note.strip(),
                )
                st.toast("✅ 已存入病歷（同階段已更新，不會重複）")
                time.sleep(0.2)
                st.rerun()

            st.markdown("---")
            st.markdown("### 8) 一鍵通報診所（讓客人覺得真的被照顧）")

            auto_bad = severe_flag or sym_ooze or (sym_pain and sym_red) or (curr_metrics["redness"] < 55)
            reason = "系統判定狀況可能不理想" if auto_bad else "客人主動通報"

            st.markdown(
                f"""
<div class="card">
  <div><b>通報理由：</b> {reason}</div>
  <div class="small">按下後，診所端會立即收到紀錄，可致電關心並安排回診。</div>
</div>
""",
                unsafe_allow_html=True
            )

            alert_btn = st.button("📣 通報診所（立即請求協助）", use_container_width=True)

            if alert_btn:
                severity = "high" if auto_bad else "normal"
                create_alert(
                    user_id=user["user_id"],
                    stage=stage,
                    severity=severity,
                    reason=reason,
                    symptoms=symptoms_text,
                    metrics=curr_metrics,
                    img_path=preview_path,
                )
                st.success("已通報診所。診所將盡快與您聯繫。")
                st.balloons()

    # -----------------------------------------------------
    # Tab2: Report / History
    # -----------------------------------------------------
    with tab2:
        st.subheader("成效報告（可給客人看的版本）")
        recs = fetch_records(user["user_id"])
        if not recs:
            st.info("尚無病歷資料。請在「術後追蹤」存入至少一筆。")
        else:
            if go is not None:
                fig = plot_trend(recs)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plotly 未安裝，將不顯示圖表。")

            st.markdown("---")
            st.markdown("### 歷史紀錄")
            for r in recs:
                m = {
                    "wrinkle": int(r.get("wrinkle") or 0),
                    "spot": int(r.get("spot") or 0),
                    "redness": int(r.get("redness") or 0),
                    "pore": int(r.get("pore") or 0),
                    "texture": int(r.get("texture") or 0),
                }
                col1, col2 = st.columns([1.0, 1.2])
                with col1:
                    if r.get("img_path") and os.path.exists(r["img_path"]):
                        st.image(r["img_path"], caption=f"{r.get('stage','')}｜術後日 {r.get('postop_date') or r.get('record_date','')}｜上傳 {r.get('uploaded_at','') or '—'}", use_container_width=True)
                with col2:
                    st.markdown(f"**{r.get('stage','')}｜術後日 {r.get('postop_date') or r.get('record_date','')}**")
                    st.caption(f"上傳時間：{r.get('uploaded_at', '') or '—'}")
                    st.caption(f"拍攝品質：{r.get('q_score',0)} / 可信度：{r.get('confidence',0)}")
                    if go is not None:
                        radar = plot_radar(m)
                        if radar is not None:
                            st.plotly_chart(radar, use_container_width=True)
                    else:
                        st.json(m)
                    if r.get("note"):
                        st.caption("備註：" + str(r["note"]))

    # -----------------------------------------------------
    # Tab3: Appointment
    # -----------------------------------------------------
    with tab3:
        st.subheader("預約回診（上架版介面）")

        today = date.today()
        end_next_year = date(today.year + 1, 12, 31)

        d = st.date_input("日期（不可選今天以前；僅今年~明年）", value=today + timedelta(days=7), min_value=today, max_value=end_next_year)

        slots = [
            "10:00", "10:30", "11:00", "11:30",
            "14:00", "14:30", "15:00", "15:30",
            "16:00", "16:30", "17:00"
        ]
        t = st.selectbox("時段（下拉選擇）", slots, index=0)
        note = st.text_input("備註（選填）", value="術後追蹤回診")

        appt_dt = f"{d.isoformat()} {t}"

        confirm_send = st.checkbox("我確認送出此預約時段", value=False)
        if st.button("送出預約", type="primary", use_container_width=True, disabled=not confirm_send):
            ok, msg = create_appointment(user["user_id"], appt_dt, note.strip())
            if ok:
                st.success(msg)
                time.sleep(0.2)
                st.rerun()
            else:
                st.warning(msg)

        st.markdown("---")
        st.markdown("####### 我的預約清單（取消後會直接消失）")

        appts_all = fetch_appointments(user["user_id"])
        appts = [a for a in appts_all if a.get("status") in ("requested", "confirmed")]

        if not appts:
            st.info("目前沒有有效預約。")
        else:
            for a in appts:
                c1, c2, c3 = st.columns([2.7, 1.1, 1.2])
                c1.write(f"🗓️ {a.get('appt_dt','')} | 備註：{a.get('note','')}")
                c2.write(f"狀態：**{a.get('status','')}**")
                confirm = c3.checkbox("確認取消", key=f"confirm_{a['id']}")
                if c3.button("取消預約", key=f"cancel_{a['id']}", disabled=not confirm):
                    ok = cancel_appointment(a["id"], user["user_id"])
                    if ok:
                        st.toast("已取消預約")
                        time.sleep(0.2)
                        st.rerun()
                    else:
                        st.warning("取消失敗")

    # -----------------------------------------------------
    # Tab4: Clinic alerts (demo)
    # -----------------------------------------------------
    with tab4:
        st.subheader("診所端：客戶通報（Demo）")
        al = fetch_alerts(limit=50)
        if not al:
            st.info("目前沒有通報。")
        else:
            for a in al:
                sev = a.get("severity", "normal")
                sev_txt = "高" if sev == "high" else "一般"
                st.markdown(
                    f"""
<div class="card">
  <div><b>{a.get('name','(未填姓名)')}</b>｜{a.get('treatment','')}</div>
  <div class="small">時間：{a.get('created_at','')}｜術後階段：{a.get('stage','')}｜嚴重度：{sev_txt}</div>
  <hr/>
  <div><b>原因：</b> {a.get('reason','')}</div>
  <div><b>症狀：</b> {a.get('symptoms','（未填）') if a.get('symptoms') else '（未填）'}</div>
  <div class="small">指標：{a.get('metrics_json','')}</div>
</div>
""",
                    unsafe_allow_html=True
                )
                if a.get("img_path") and os.path.exists(a["img_path"]):
                    st.image(a["img_path"], caption="通報當下照片（已校正分析）", width=360)
                st.markdown("")


# =========================================================
# 10) Run
# =========================================================
if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()
