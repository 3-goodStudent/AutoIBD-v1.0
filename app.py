# ------------------------------
# 核心依赖（保持严格顺序）
# ------------------------------
import streamlit as st

# 必须第一个设置页面配置
st.set_page_config(
    page_title="IBD智能诊断系统",
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
        st.error(f"数据处理错误: {str(e)}")
        return None, None

# ------------------------------
# 初始化背景
# ------------------------------
set_bg_local("background.jpg")  # 调用背景设置

# ------------------------------
# 主界面
# ------------------------------
st.title("肠道菌群IBD智能诊断系统")
st.markdown("""
**基于机器学习的两阶段诊断模型**  
- **Stage1**: 采用CatBoost算法筛查IBD（炎症性肠病） 🦠  
- **Stage2**: 使用LightGBM算法区分CD和UC亚型 🔬
""")

# ------------------------------
# 侧边栏 - 数据上传
# ------------------------------
with st.sidebar:
    st.header("数据上传")
    uploaded_file = st.file_uploader(
        "上传检测数据（*.csv / *.xlsx）",
        type=["csv", "xlsx"],
        help="文件格式要求：\n1. 首行为样本标签\n2. 首列为菌群物种名称"
    )
    
    if uploaded_file:
        st.success("✔️ 文件上传成功")
        st.caption(f"已接收文件：`{uploaded_file.name}`")

# ------------------------------
# 主内容区
# ------------------------------
if uploaded_file:
    try:
        # 数据加载与预处理
        with st.spinner('正在解析数据...'):
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            X, _ = preprocess_data(df)
        
        # 布局设置
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # ===== 数据预览部分 =====
            st.subheader("数据概览")
            st.dataframe(df.head(3), use_container_width=True, height=200)
            
            # 显示数据特征统计
            with st.expander("查看数据统计"):
                st.write(f"特征数量：{X.shape[1]}")
                st.write(f"样本数量：{X.shape[0]}")
            
            # ===== 诊断按钮区域 =====
            st.divider()
            btn_col, status_col = st.columns([2, 4])
            
            with btn_col:
                if st.button(
                    "🚀 启动智能诊断",
                    help="点击开始分析流程",
                    use_container_width=True,
                    type="primary"
                ):
                    st.session_state.run_diagnosis = True
            
            with status_col:
                if 'run_diagnosis' not in st.session_state:
                    st.info("等待启动诊断分析...")
                else:
                    st.empty()

        # ===== 执行诊断流程 =====
        if 'run_diagnosis' in st.session_state:
            with col1:
                with st.status("正在进行深度分析...", expanded=True) as status:
                    # Stage1预测
                    st.write("**Stage1 - IBD初步筛查**")
                    stage1_pred = catboost_model.predict(X)
                    prob1 = catboost_model.predict_proba(X)[0][1] * 100
                    st.write(f"IBD可能性：{prob1:.1f}%")
                    
                    # Stage2预测（如果预测为IBD）
                    if stage1_pred[0] == 1:
                        st.write("**Stage2 - 疾病亚型分析**")
                        stage2_pred = lightgbm_model.predict(X)
                        prob2 = lightgbm_model.predict_proba(X)[0][1] * 100
                        st.write(f"CD可能性：{prob2:.1f}%")
                        
                        status.update(
                            label="分析完成 ✅",
                            state="complete",
                            expanded=False
                        )
                        st.success(f"**最终诊断**: {'克罗恩病（CD）' if stage2_pred[0]==1 else '溃疡性结肠炎（UC）'}")
                    else:
                        status.update(
                            label="分析完成 ✅",
                            state="complete",
                            expanded=False
                        )
                        st.success("**诊断结果**: 健康对照（HC）")
                    
        with col2:
            # ===== 可视化区域 =====
            st.subheader("特征分析")
            st.write("*此处可集成SHAP可视化组件*")
            
    except Exception as e:
        st.error(f"遇到错误: {str(e)}")
        st.info("请检查数据格式是否符合要求")
