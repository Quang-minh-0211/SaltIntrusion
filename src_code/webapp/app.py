"""
Salinity Intrusion Forecasting — Web Application
Framework : Streamlit
Scenario  : SC3 (3 stations → BenLuc)
Models    : RNN, LSTM, GRU, CNN-LSTM
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from pathlib import Path
from datetime import timedelta
from models import build_model

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dự báo xâm nhập mặn",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Elegant, minimal, professional
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ─────────────────────────────────────────── */
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    
    /* ── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2633 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #e8ecf1 !important;
    }

    /* ── Headers ────────────────────────────────────────── */
    .main-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a2633;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #6b7c8d;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* ── Metric Cards ───────────────────────────────────── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e8ecf1;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        transition: box-shadow 0.2s ease;
    }
    .metric-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6b7c8d;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a2633;
    }

    /* ── Section dividers ───────────────────────────────── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a2633;
        border-bottom: 2px solid #e8ecf1;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.01em;
    }

    /* ── Data input area ────────────────────────────────── */
    .input-info {
        background: #f7f9fb;
        border: 1px solid #e8ecf1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #4a5c6d;
        line-height: 1.6;
    }

    /* ── Result table ───────────────────────────────────── */
    .result-table {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }

    /* ── Hide streamlit branding ────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ── Reduce padding ─────────────────────────────────── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

MODEL_DIR = BASE_DIR / "output" / "models"

STATIONS       = ["BenLuc", "CauNoi", "TanAn"]
TARGET_STATION = "BenLuc"
TARGET_COL     = f"Salinity_{TARGET_STATION}"

MODEL_NAMES = ["RNN", "LSTM", "GRU", "CNN_LSTM"]
MODEL_DISPLAY = {"RNN": "RNN", "LSTM": "LSTM",
                 "GRU": "GRU", "CNN_LSTM": "CNN-LSTM"}

LOOKBACK_OPTIONS = {
    "6 giờ (3 bước)":  3,
    "12 giờ (6 bước)": 6,
    "24 giờ (12 bước)": 12,
    "48 giờ (24 bước)": 24,
}
HORIZON_OPTIONS = {
    "12 giờ (6 bước)": 6,
    "24 giờ (12 bước)": 12,
    "48 giờ (24 bước)": 24,
}

FEATURE_COLS  = ["wind_speed", "temp", "total_precipitation"]
TEMPORAL_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
                 "month_sin", "month_cos"]
LAG_STEPS     = [6, 12, 24]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_INPUT_COLS = [f"Salinity_{s}" for s in STATIONS] + FEATURE_COLS



# ─────────────────────────────────────────────────────────────
# LOAD MODEL + SCALERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scalers(model_name, lookback, horizon):
    lb_h = lookback * 2
    hz_h = horizon * 2
    tag  = (f"SC{2}_{'_'.join(STATIONS)}_to_{TARGET_STATION}"
            f"_{model_name}_lb{lb_h}h_hz{hz_h}h")

    pt_path  = MODEL_DIR / f"{tag}.pt"
    scX_path = MODEL_DIR / f"{tag}_scaler_X.pkl"
    scY_path = MODEL_DIR / f"{tag}_scaler_y.pkl"

    for p in [pt_path, scX_path, scY_path]:
        if not p.exists():
            return None, None, None, f"Không tìm thấy file: {p.name}"

    scX = joblib.load(scX_path)
    scY = joblib.load(scY_path)

    input_size = scX.n_features_in_
    model      = build_model(model_name, input_size, horizon,)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    return model, scX, scY, None


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────
def predict(model, scX, scY, df_input, lookback, horizon):
    """
    df_input: DataFrame với cột Time + 6 raw features.
    """
    sal_cols   = [f"Salinity_{s}" for s in STATIONS]
    all_feats  = sal_cols + FEATURE_COLS

    df_clean = df_input.dropna(subset=all_feats).reset_index(drop=True)
    if len(df_clean) < lookback:
        return None, None, f"Cần ít nhất {lookback} dòng dữ liệu. Hiện có {len(df_clean)} dòng."

    # Lấy lookback dòng cuối
    window = df_clean.iloc[-lookback:]
    X_raw  = window[all_feats].values.astype(np.float32)
    X_sc   = scX.transform(X_raw)
    X_t    = torch.tensor(X_sc, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_sc = model(X_t).cpu().numpy().reshape(-1, 1)

    pred_values = scY.inverse_transform(pred_sc).flatten()

    # Tạo timestamps cho dự báo
    last_time   = window["Time"].iloc[-1]
    pred_times  = [last_time + timedelta(hours=2*(i+1)) for i in range(horizon)]

    return pred_values, pred_times, None

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Cấu hình dự báo")
    st.markdown("---")

    model_choice = st.selectbox(
        "Mô hình",
        MODEL_NAMES,
        format_func=lambda x: MODEL_DISPLAY[x],
    )

    lookback_choice = st.selectbox("Dữ liệu đầu vào (Lookback)",
                                   list(LOOKBACK_OPTIONS.keys()))
    lookback_steps  = LOOKBACK_OPTIONS[lookback_choice]

    horizon_choice  = st.selectbox("Mốc dự báo (Horizon)",
                                   list(HORIZON_OPTIONS.keys()))
    horizon_steps   = HORIZON_OPTIONS[horizon_choice]

    st.markdown("---")
    st.markdown(f"""
    <div class="input-info">
        <strong>Yêu cầu dữ liệu:</strong><br>
        Số dòng tối thiểu: <strong>{lookback_steps + max(LAG_STEPS)}</strong><br>
        (= {lookback_steps} lookback + {max(LAG_STEPS)} lag)<br><br>
        <strong>6 cột bắt buộc:</strong><br>
        Salinity_BenLuc<br>
        Salinity_CauNoi<br>
        Salinity_TanAn<br>
        wind_speed<br>
        temp<br>
        total_precipitation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="color:#6b7c8d; font-size:0.75rem;">'
        'Mô hình dự báo xâm nhập mặn<br>'
        'Sông Vàm Cỏ — Long An'
        '</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Dự báo xâm nhập mặn — Sông Vàm Cỏ</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">'
    'Hệ thống dự báo độ mặn sử dụng mô hình học sâu '
    f'— Kịch bản: 3 trạm (BenLuc, CauNoi, TanAn) → {TARGET_STATION}'
    '</p>',
    unsafe_allow_html=True
)

# ── Bản đồ ──────────────────────────────────────────────────
st.markdown('<p class="section-header">Vị trí các trạm quan trắc</p>',
            unsafe_allow_html=True)

map_html = """
<iframe
    width="100%"
    height="350"
    style="border:1px solid #e8ecf1; border-radius:10px;"
    loading="lazy"
    allowfullscreen
    referrerpolicy="no-referrer-when-downgrade"
    src="https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d125830.!2d106.4!3d10.55!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e0!3m2!1sen!2svn!4v1700000000000!5m2!1sen!2svn&q=sông+vàm+cỏ+long+an">
</iframe>
"""
st.markdown(map_html, unsafe_allow_html=True)

# ── Input Data ──────────────────────────────────────────────
st.markdown('<p class="section-header">Nhập dữ liệu đầu vào</p>',
            unsafe_allow_html=True)

min_rows = lookback_steps

tab_manual, tab_csv = st.tabs(["Nhập tay", "Upload CSV"])

df_input = None

with tab_manual:
    st.markdown(
        f'<div class="input-info">'
        f'Nhập <strong>{min_rows}</strong> dòng dữ liệu, '
        f'mỗi dòng cách nhau <strong>2 giờ</strong>. '
        f'Thời gian bắt đầu sẽ được tính ngược từ thời gian kết thúc.'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    col_date, col_time = st.columns(2)
    with col_date:
        end_date = st.date_input("Ngày kết thúc chuỗi đo",
                                  value=pd.Timestamp("2026-05-18"))
    with col_time:
        end_hour = st.selectbox("Giờ kết thúc",
                                [f"{h:02d}:00" for h in range(1, 24, 2)],
                                index=0)

    end_dt = pd.Timestamp(f"{end_date} {end_hour}")
    time_index = pd.date_range(
        end=end_dt, periods=min_rows, freq="2h"
    )

    # Tạo DataFrame mẫu để người dùng điền
    default_data = pd.DataFrame({
        "Time": time_index,
        "Salinity_BenLuc": [0.0] * min_rows,
        "Salinity_CauNoi": [0.0] * min_rows,
        "Salinity_TanAn":  [0.0] * min_rows,
        "wind_speed":      [0.0] * min_rows,
        "temp":            [0.0] * min_rows,
        "total_precipitation": [0.0] * min_rows,
    })

    edited_df = st.data_editor(
        default_data,
        column_config={
            "Time": st.column_config.DatetimeColumn(
                "Thời gian", format="DD/MM/YYYY HH:mm", disabled=True
            ),
            "Salinity_BenLuc": st.column_config.NumberColumn(
                "Mặn BenLuc (‰)", min_value=0.0, step=0.1, format="%.1f"
            ),
            "Salinity_CauNoi": st.column_config.NumberColumn(
                "Mặn CauNoi (‰)", min_value=0.0, step=0.1, format="%.1f"
            ),
            "Salinity_TanAn": st.column_config.NumberColumn(
                "Mặn TanAn (‰)", min_value=0.0, step=0.1, format="%.1f"
            ),
            "wind_speed": st.column_config.NumberColumn(
                "Gió (m/s)", min_value=0.0, step=0.1, format="%.1f"
            ),
            "temp": st.column_config.NumberColumn(
                "Nhiệt độ (°C)", min_value=0.0, step=0.1, format="%.1f"
            ),
            "total_precipitation": st.column_config.NumberColumn(
                "Mưa (mm)", min_value=0.0, step=0.1, format="%.1f"
            ),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
    )

    if st.button("Dự báo", type="primary", use_container_width=True,
                  key="btn_manual"):
        df_input = edited_df.copy()

with tab_csv:
    st.markdown(
        '<div class="input-info">'
        'Upload file CSV với các cột: '
        '<strong>Time, Salinity_BenLuc, Salinity_CauNoi, Salinity_TanAn, '
        'wind_speed, temp, total_precipitation</strong>. '
        'Cột Time có dạng <code>YYYY-MM-DD HH:MM:SS</code> '
        'hoặc <code>DD/MM/YYYY HH:MM</code>.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    uploaded = st.file_uploader("Chọn file CSV", type=["csv"],
                                 label_visibility="collapsed")

    if uploaded is not None:
        try:
            csv_df = pd.read_csv(uploaded, parse_dates=["Time"])
            st.dataframe(csv_df.tail(10), use_container_width=True,
                         hide_index=True)
            st.markdown(f"Tổng số dòng: **{len(csv_df)}** "
                        f"(cần tối thiểu **{min_rows}**)")

            if st.button("Dự báo", type="primary",
                          use_container_width=True, key="btn_csv"):
                df_input = csv_df.copy()
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")


# ─────────────────────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────────────────────
if df_input is not None:
    # Validate
    df_input["Time"] = pd.to_datetime(df_input["Time"])
    missing_cols = [c for c in RAW_INPUT_COLS if c not in df_input.columns]
    if missing_cols:
        st.error(f"Thiếu cột: {', '.join(missing_cols)}")
    elif len(df_input) < min_rows:
        st.error(f"Cần tối thiểu {min_rows} dòng, hiện có {len(df_input)}.")
    else:
        # Load model
        model, scX, scY, err = load_model_and_scalers(
            model_choice, lookback_steps, horizon_steps
        )
        if err:
            st.error(err)
        else:
            # Predict
            with st.spinner("Đang dự báo..."):
                pred_vals, pred_times, pred_err = predict(
                    model, scX, scY, df_input,
                    lookback_steps, horizon_steps
                )

            if pred_err:
                st.error(pred_err)
            else:
                # ── Results ──────────────────────────────────
                st.markdown(
                    '<p class="section-header">Kết quả dự báo</p>',
                    unsafe_allow_html=True
                )

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Mô hình</div>'
                        f'<div class="metric-value">{MODEL_DISPLAY[model_choice]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Đầu vào</div>'
                        f'<div class="metric-value">{lookback_steps*2}h</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col3:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Dự báo</div>'
                        f'<div class="metric-value">{horizon_steps*2}h</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col4:
                    max_sal = pred_vals.max()
                    alert   = "Cao" if max_sal > 4.0 else "TB" if max_sal > 1.0 else "Thấp"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Mức cảnh báo</div>'
                        f'<div class="metric-value">{alert}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("")

                # ── Chart ────────────────────────────────────
                # Input salinity (last lookback points)
                df_input_sorted = df_input.sort_values("Time")
                hist_time = df_input_sorted["Time"].iloc[-lookback_steps:].values
                hist_sal  = df_input_sorted[TARGET_COL].iloc[-lookback_steps:].values

                fig = go.Figure()

                # Historical line
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(hist_time),
                    y=hist_sal,
                    mode="lines+markers",
                    name="Dữ liệu đầu vào",
                    line=dict(color="#3d5a80", width=2),
                    marker=dict(size=5, color="#3d5a80"),
                ))

                # Connect historical → forecast
                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(hist_time[-1]), pred_times[0]],
                    y=[hist_sal[-1], pred_vals[0]],
                    mode="lines",
                    line=dict(color="#adb5bd", width=1.5, dash="dot"),
                    showlegend=False,
                ))

                # Forecast line
                fig.add_trace(go.Scatter(
                    x=pred_times,
                    y=pred_vals,
                    mode="lines+markers",
                    name="Dự báo",
                    line=dict(color="#e63946", width=2.5),
                    marker=dict(size=7, color="#e63946",
                                symbol="diamond"),
                ))

                # Threshold line
                fig.add_hline(y=1.0, line_dash="dash",
                              line_color="#adb5bd",
                              annotation_text="Ngưỡng 1‰",
                              annotation_position="top left",
                              annotation_font_size=11,
                              annotation_font_color="#6b7c8d")

                fig.update_layout(
                    title=dict(
                        text=f"Dự báo độ mặn — Trạm {TARGET_STATION}",
                        font=dict(family="DM Sans", size=16, color="#1a2633"),
                    ),
                    xaxis_title="Thời gian",
                    yaxis_title="Độ mặn (‰)",
                    template="plotly_white",
                    font=dict(family="DM Sans", size=12),
                    legend=dict(
                        orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1
                    ),
                    margin=dict(l=40, r=20, t=60, b=40),
                    height=420,
                    plot_bgcolor="#fafbfc",
                )
                fig.update_xaxes(
                    gridcolor="#f0f0f0",
                    tickformat="%d/%m\n%H:%M",
                )
                fig.update_yaxes(gridcolor="#f0f0f0")

                st.plotly_chart(fig, use_container_width=True)

                # ── Prediction Table ─────────────────────────
                st.markdown(
                    '<p class="section-header">Chi tiết dự báo</p>',
                    unsafe_allow_html=True
                )

                df_result = pd.DataFrame({
                    "Thời gian": [t.strftime("%d/%m/%Y %H:%M")
                                  for t in pred_times],
                    "Bước":     [f"t+{(i+1)*2}h" for i in range(horizon_steps)],
                    "Độ mặn dự báo (‰)": [round(v, 2) for v in pred_vals],
                })

                st.dataframe(
                    df_result,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Độ mặn dự báo (‰)": st.column_config.ProgressColumn(
                            min_value=0,
                            max_value=max(pred_vals.max() * 1.2, 5.0),
                            format="%.2f",
                        )
                    },
                )

                # ── Download results ─────────────────────────
                csv_download = df_result.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Tải kết quả CSV",
                    csv_download,
                    file_name=f"forecast_{model_choice}_{lookback_steps*2}h_{horizon_steps*2}h.csv",
                    mime="text/csv",
                )
