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
# 模型元数据加载
# ------------------------------
@st.cache(allow_output_mutation=True)
def load_model_with_meta(model_path):
    """加载包含元数据的模型"""
    model_dict = joblib.load(model_path)
    return model_dict['model'], model_dict['feature_names'], model_dict.get('feature_stats', {})

catboost_model, cb_features, cb_stats = load_model_with_meta('IBD_vs_HC_best_model_with_meta.pkl')
lightgbm_model, lgb_features, lgb_stats = load_model_with_meta('CD_vs_UC_best_model_with_meta.pkl')

# ------------------------------
# 特征对齐函数（核心逻辑）
# ------------------------------
def align_features(input_df, model_features, feature_stats=None):
    """对齐输入特征与模型特征"""
    try:
        # 初始化标准化DataFrame
        aligned = pd.DataFrame(columns=model_features)
        
        # 清洗输入特征名称
        cleaned_input = (
            input_df.columns
            .str.replace('.*[sgtpf]__', '', regex=True)  # 去除分类层级
            .str.replace('[^a-zA-Z0-9]', '_', regex=True)
            .str.strip('_')
            .str.lower()
        )
        
        # 构建特征映射表 (模型特征小写->标准特征名)
        model_feature_map = {f.lower(): f for f in model_features}
        
        # 遍历清洗后的输入特征
        for input_name in cleaned_input:
            if input_name in model_feature_map:
                std_name = model_feature_map[input_name]
                aligned[std_name] = input_df[input_df.columns[cleaned_input == input_name][0]]
        
        # 填充缺失特征
        missing_features = set(model_features) - set(aligned.columns)
        for f in missing_features:
            fill_value = feature_stats.get(f, {}).get('min', 0) if feature_stats else 0
            aligned[f] = fill_value
        
        # 确保特征顺序
        return aligned[model_features]
        
    except Exception as e:
        st.error(f"Feature alignment failed: {str(e)}")
        return None

# ------------------------------
# 数据预处理流程
# ------------------------------
def preprocess_prediction_data(raw_df, stage=1):
    """处理预测数据:
    raw_df: 行名是菌种名称，列名是样本名称
    stage: 1表示第一阶段模型，2表示第二阶段模型
    """
    try:
        # 选择对应模型的参数
        target_model = catboost_model if stage ==1 else lightgbm_model
        target_features = cb_features if stage ==1 else lgb_features
        stats = cb_stats if stage ==1 else lgb_stats
        
        # 转置为样本×特征格式
        features = raw_df.T
        
        # 对齐特征
        aligned_features = align_features(features, target_features, stats)
        if aligned_features is None:
            return None
            
        # 数值转换
        aligned_features = aligned_features.apply(pd.to_numeric, errors='coerce').fillna(0)
        return aligned_features
    
    except Exception as e:
        st.error(f"预处理失败: {str(e)}")
        return None

# ------------------------------
# 初始化背景
# ------------------------------
set_bg_local("background.jpg")  # 请确保存在背景图片

# ------------------------------
# 主界面
# ------------------------------
st.title("Intestinal Flora IBD Diagnostic System")
st.markdown("""
**Two-Stage Machine Learning Diagnostic Model**  
- **Stage1**: IBD Screening with CatBoost 
- **Stage2**: CD/UC Classification with LightGBM
""")

# ------------------------------
# 侧边栏 - 数据上传
# ------------------------------
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Test Data (.xlsx)",
        type=["xlsx"],
        help="File format requirements:\n1. First row: Sample names\n2. First column: Microorganism names"
    )
    
    if uploaded_file:
        st.success("✔️ File uploaded successfully")
        st.caption(f"Received file: `{uploaded_file.name}`")

# ------------------------------
# 主内容区
# ------------------------------
if uploaded_file:
    try:
        # 读取数据
        with st.spinner('Loading data...'):
            raw_df = pd.read_excel(uploaded_file, index_col=0)
            st.session_state.raw_df = raw_df
            
        # 显示数据预览
        st.subheader("Data Preview")
        st.write("**Sample Columns:**", raw_df.columns.tolist()[:5], "...")
        st.write("**First 5 Microorganisms:**", raw_df.index.tolist()[:5])
        st.dataframe(raw_df.iloc[:5, :3].style.format("{:.4f}"))
        
        # Stage1预处理
        st.divider()
        with st.spinner('Preprocessing data for Stage1...'):
            X_stage1 = preprocess_prediction_data(raw_df, stage=1)
            
        if X_stage1 is None:
            st.stop()
            
        # 显示特征匹配报告
        with st.expander("Feature Matching Report"):
            matched = len(set(X_stage1.columns) & set(cb_features))
            missing = len(cb_features) - matched
            st.write(f"✅ Matched Features: {matched}")
            st.write(f"⚠️ Missing Features (filled with defaults): {missing}")
            if missing > 0:
                st.write("Example Missing Features:", list(set(cb_features) - set(X_stage1.columns))[:3])
        
        # 诊断按钮
        if st.button("🚀 Start Diagnosis", type="primary"):
            # Stage1预测
            with st.status("Stage1: IBD Screening...", expanded=True) as status1:
                stage1_pred = catboost_model.predict(X_stage1)
                proba1 = catboost_model.predict_proba(X_stage1)
                
                # 展示结果
                results_stage1 = pd.DataFrame({
                    'Sample': X_stage1.index,
                    'Diagnosis': ['IBD' if p==1 else 'Healthy' for p in stage1_pred],
                    'Confidence (%)': [f"{x[1]*100:.1f}" for x in proba1]
                })
                st.dataframe(results_stage1)
                status1.update(label="Stage1 Completed ✅", state="complete")
                
                # Stage2处理IBD样本
                if 1 in stage1_pred:
                    st.divider()
                    with st.status("Stage2: CD/UC Classification...") as status2:
                        ibd_samples = X_stage1[stage1_pred == 1].index
                        # Stage2预处理
                        X_stage2 = preprocess_prediction_data(
                            raw_df[ibd_samples], 
                            stage=2
                        )
                        if X_stage2 is None:
                            st.stop()
                            
                        # Stage2预测
                        stage2_pred = lightgbm_model.predict(X_stage2)
                        proba2 = lightgbm_model.predict_proba(X_stage2)
                        
                        # 展示结果
                        results_stage2 = pd.DataFrame({
                            'Sample': X_stage2.index,
                            'Subtype': ['CD' if p==1 else 'UC' for p in stage2_pred],
                            'Confidence (%)': [f"{x[1]*100:.1f}" for x in proba2]
                        })
                        st.dataframe(results_stage2)
                        status2.update(label="Stage2 Completed ✅", state="complete")
                        
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.info("Please verify the file format meets requirements")

