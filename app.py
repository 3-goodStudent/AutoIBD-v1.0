# ------------------------------
# 核心依赖（保持严格顺序）
# ------------------------------
import streamlit as st

# 必须第一个设置页面配置
st.set_page_config(
    page_title="IBD Intelligent Diagnostic System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64
import pandas as pd
import joblib

# ------------------------------
# 背景与样式设置
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
            background-color: rgba(255, 255, 255, 0.88);
            border-radius: 12px;
            padding: 2.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }}
        
        /* 侧边栏样式 */
        [data-testid="stSidebar"] {{
            background: linear-gradient(175deg, rgba(255,255,255,0.96) 0%, rgba(249,249,249,0.96) 100%) !important;
            border-right: 1px solid #eee;
            box-shadow: 5px 0 15px rgba(0,0,0,0.03);
        }}
        [data-testid="stSidebar"] .block-container {{
            padding: 2rem 1.2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# 模型加载（保持不变）
# ------------------------------
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return joblib.load(model_path)

catboost_model = load_model('IBD_vs_HC_best_model.pkl')
lightgbm_model = load_model('CD_vs_UC_best_model.pkl')

# ------------------------------
# 预处理函数
# ------------------------------
def preprocess_data(df):
    try:
        labels = df.iloc[0, 1:].replace({'IBD': 1, 'HC': 0}).astype(int)
        features = df.iloc[1:, 1:].copy()
        features.index = df.iloc[1:, 0].str.split('s__').str[-1]
        features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
        return features.T, labels
    except Exception as e:
        st.error(f"Data processing errors: {str(e)}")
        return None, None

# ------------------------------
# 初始化背景
# ------------------------------
set_bg_local("background.jpg")  # 调用背景设置

# ------------------------------
# 主界面
# ------------------------------
st.title("Intelligent Diagnostic System for Intestinal Flora IBD")
st.markdown("""
**A two-stage diagnostic model based on machine learning**  
- **Stage1**: Screening for IBD (Inflammatory Bowel Disease) using the CatBoost algorithm 🦠  
- **Stage2**: Distinguishing CD and UC subtypes using the LightGBM algorithm 🔬
""")

# ------------------------------
# 侧边栏 - 数据上传
# ------------------------------
with st.sidebar:
    st.header("Data upload")
    uploaded_file = st.file_uploader(
        "Upload test data（*.csv / *.xlsx）",
        type=["csv", "xlsx"],
        help="Document formatting requirements：\n1. Sample labels for the first row\n2. First listed as species name of the colony"
    )
    
    if uploaded_file:
        st.success("✔️ File uploaded successfully")
        st.caption(f"Documents received：`{uploaded_file.name}`")

# ------------------------------
# 主内容区
# ------------------------------
if uploaded_file:
    try:
        # 数据加载与预处理
        with st.spinner('Parsing data...'):
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            X, _ = preprocess_data(df)
        
        # 布局设置
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # ===== 数据预览部分 =====
            st.subheader("Data overview")
            st.dataframe(df.head(3), use_container_width=True, height=200)
            
            # 显示数据特征统计
            with st.expander("View Statistics"):
                st.write(f"Number of features：{X.shape[1]}")
                st.write(f"Sample size：{X.shape[0]}")
            
            # ===== 诊断按钮区域 =====
            st.divider()
            btn_col, status_col = st.columns([2, 4])
            
            with btn_col:
                if st.button(
                    "🚀 Initiate Intelligent Diagnostics",
                    help="Click to start the analysis process",
                    use_container_width=True,
                    type="primary"
                ):
                    st.session_state.run_diagnosis = True
            
            with status_col:
                if 'run_diagnosis' not in st.session_state:
                    st.info("Waiting to start diagnostic analysis...")
                else:
                    st.empty()

        # ===== 执行诊断流程 =====
        if 'run_diagnosis' in st.session_state:
            with col1:
                with st.status("In-depth analysis in progress...", expanded=True) as status:
                    # Stage1预测
                    st.write("**Stage1 - IBD initial screening**")
                    stage1_pred = catboost_model.predict(X)
                    prob1 = catboost_model.predict_proba(X)[0][1] * 100
                    st.write(f"IBD likelihood：{prob1:.1f}%")
                    
                    # Stage2预测（如果预测为IBD）
                    if stage1_pred[0] == 1:
                        st.write("**Stage2 - Disease subtype analysis**")
                        stage2_pred = lightgbm_model.predict(X)
                        prob2 = lightgbm_model.predict_proba(X)[0][1] * 100
                        st.write(f"CD likelihood：{prob2:.1f}%")
                        
                        status.update(
                            label="Analysis completed ✅",
                            state="complete",
                            expanded=False
                        )
                        st.success(f"**Final diagnosis**: {'Crohn\'s disease（CD）' if stage2_pred[0]==1 else 'ulcerative colitis（UC）'}")
                    else:
                        status.update(
                            label="Analysis completed ✅",
                            state="complete",
                            expanded=False
                        )
                        st.success("**diagnosis result**: health control（HC）")
                    
        with col2:
            # ===== 可视化区域 =====
            st.subheader("characterization")
            st.write("*SHAP visualization components can be integrated here*")
            
    except Exception as e:
        st.error(f"encounter an error: {str(e)}")
        st.info("Please check the data format for compliance")
