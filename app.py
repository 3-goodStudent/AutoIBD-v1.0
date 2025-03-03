# ------------------------------
# 核心依赖（严格保持顺序）
# ------------------------------
import streamlit as st

# 必须第一个设置页面配置
st.set_page_config(
    page_title="IBD智能诊断系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64
import pandas as pd
import joblib

# ------------------------------
# 背景设置（与文档2一致）
# ------------------------------
def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
        }}
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px 0 0 10px;
            box-shadow: 2px 0 8px rgba(0,0,0,0.1);
        }}
        [data-testid="stSidebar"] .block-container {{
            padding: 2rem 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# 模型加载（与原始文档一致）
# ------------------------------
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return joblib.load(model_path)

catboost_model = load_model('IBD_vs_HC_best_model.pkl')
lightgbm_model = load_model('CD_vs_UC_best_model.pkl')

# ------------------------------
# 预处理函数（保持文档结构）
# ------------------------------
def preprocess_data(df):
    try:
        labels = df.iloc[0, 1:].replace({'IBD': 1, 'HC': 0}).astype(int)
        features = df.iloc[1:, 1:].copy()
        features.index = df.iloc[1:, 0].str.split('s__').str[-1]
        features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
        return features.T, labels
    except Exception as e:
        st.error(f"数据处理错误: {str(e)}")
        return None, None

# ------------------------------
# 主界面（优化后的交互）
# ------------------------------
set_bg_local("background.jpg")  # 调用背景设置

st.title("IBD Diagnosis and Subtyping Online System")
st.markdown("""
This application enables non-invasive IBD diagnosis and subtyping based on a two-stage machine learning model:

**Stage 1** 🔍 CatBoost classification (IBD vs Healthy)  
**Stage 2** 🧬 LightGBM classification (CD vs UC)
""")

# 侧边栏上传（保持文档逻辑）
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], help="Upload microbiome data in required format")

if uploaded_file is not None:
    try:
        with st.spinner('Parsing data...'):
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(), height=200)
            
            X, _ = preprocess_data(df)
            st.success(f"Valid feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

        # 预测逻辑（保持原始业务逻辑）
        if st.sidebar.button("Run Analysis", type="primary"):
            with st.status("Analyzing...", expanded=True) as status:
                st.write("Stage 1: IBD Detection")
                stage1_pred = catboost_model.predict(X)
                
                if stage1_pred[0] == 1:
                    st.write("Stage 2: Disease Subtyping")
                    stage2_pred = lightgbm_model.predict(X)
                    
                    status.update(label="Analysis Complete", state="complete")
                    st.success(f"**Final Diagnosis**: {'Crohn’s Disease' if stage2_pred[0]==1 else 'Ulcerative Colitis'}")
                else:
                    status.update(label="Analysis Complete", state="complete")
                    st.success("**Result**: Healthy Control")

    except Exception as e:
        st.error(f"Processing Error: {e}")
