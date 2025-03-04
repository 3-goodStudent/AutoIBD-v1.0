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
# 主内容区完整代码
# ------------------------------
if uploaded_file:
    try:
        # ======================
        # 数据加载与预处理
        # ======================
        with st.spinner('🔄 Loading data...'):
            raw_df = pd.read_excel(uploaded_file, index_col=0)
            st.session_state.raw_df = raw_df
            
        # 数据预览（独立折叠面板）
        with st.expander("▸ Raw Data Overview (First 3 Samples)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Sample Names:**", raw_df.columns.tolist()[:3])
                st.write("**Total Features:**", len(raw_df))
                st.write("**Sample Count:**", len(raw_df.columns))
            with col2:
                st.dataframe(raw_df.iloc[:5, :3].style.format("{:.4f}"))
        
        # ======================
        # 第一阶段预处理
        # ======================
        st.divider()
        with st.status("🔍 Stage1 Feature Preprocessing...", expanded=False) as pre_status:
            with st.spinner('Aligning features...'):
                X_stage1 = preprocess_prediction_data(raw_df, stage=1)
            
            if X_stage1 is None:
                pre_status.update(label="Preprocessing Failed ❌", state="error")
                st.stop()
                
            pre_status.update(label="Feature Alignment Completed ✅", state="complete")
        
        # 特征质量报告
        with st.expander("▸ Feature Quality Report", expanded=True):
            tab1, tab2 = st.columns([2,3])
            
            with tab1:
                matched = len(set(X_stage1.columns) & set(cb_features))
                st.metric("Matched Features", f"{matched}/{len(cb_features)}")
                st.metric("Samples Ready", len(X_stage1))
                
            with tab2:
                missing_ratio = (X_stage1 == 0).mean().mean() * 100
                st.write("**Missing Value Distribution**")
                st.bar_chart((X_stage1 == 0).mean().sort_values(ascending=False).head(10))
                if missing_ratio > 15:
                    st.warning(f"High missing ratio: {missing_ratio:.1f}%")

        # ======================
        # 诊断主流程完整代码
        # ======================
        if st.button("🚀 Start Intelligent Diagnosis", type="primary", use_container_width=True):
            # ------------------
            # Stage1 预测流程
            # ------------------
            with st.status("🔬 Stage1: Inflammatory Status Screening...", expanded=True) as status1:
                try:
                    # 执行预测
                    stage1_pred = catboost_model.predict(X_stage1)
                    proba1 = catboost_model.predict_proba(X_stage1) * 100
                    
                    # 构建结果数据
                    results_stage1 = pd.DataFrame({
                        'Sample': X_stage1.index,
                        'Prediction': ['IBD' if p ==1 else 'Healthy' for p in stage1_pred],
                        'Healthy (%)': proba1[:, 0].round(1),
                        'IBD (%)': proba1[:, 1].round(1),
                        'Conf. Gap': (np.abs(proba1[:, 1] - proba1[:, 0])).round(1)
                    })
                    
                    # 显示配置
                    st.write("### Stage1 Prediction Report")
                    st.dataframe(
                        results_stage1.sort_values('Conf. Gap', ascending=False),
                        hide_index=True,
                        column_config={
                            "Healthy (%)": st.column_config.ProgressColumn(
                                "Healthy",
                                help="Non-inflammatory probability",
                                min_value=0,
                                max_value=100,
                                format="%.1f%%"
                            ),
                            "IBD (%)": st.column_config.ProgressColumn(
                                "IBD",
                                help="Inflammatory probability",
                                min_value=0,
                                max_value=100,
                                format="%.1f%%"
                            ),
                            "Conf. Gap": st.column_config.NumberColumn(
                                "Conf. Diff",
                                help="Probabilistic difference between predictions",
                                format="%.1f%%"
                            )
                        }
                    )
                    
                    # 更新状态
                    status1.update(
                        label=f"Stage1 Complete: {sum(stage1_pred)} IBD Cases Identified ✅", 
                        state="complete"
                    )
                    
                except Exception as e:
                    status1.update(label="Stage1 Failed ❌", state="error")
                    st.error(f"Stage1 Error: {str(e)}")
                    st.stop()
            
            # ======================
            # Stage2 预测流程
            # ======================
            if sum(stage1_pred) > 0:
                st.divider()
                
                with st.status("🔬 Stage2: Disease Subtyping Analysis...", expanded=True) as status2:
                    try:
                        # Stage2预处理
                        ibd_samples = X_stage1[stage1_pred == 1].index
                        X_stage2 = preprocess_prediction_data(raw_df[ibd_samples], stage=2)
                        
                        if X_stage2 is None:
                            status2.update(label="Feature Alignment Failed ❌", state="error")
                            st.stop()
                        
                        # 执行预测
                        stage2_pred = lightgbm_model.predict(X_stage2)
                        proba2 = lightgbm_model.predict_proba(X_stage2) * 100
                        
                        # 构建结果数据
                        results_stage2 = pd.DataFrame({
                            'Sample': X_stage2.index,
                            'Prediction': ['CD' if p ==1 else 'UC' for p in stage2_pred],
                            'UC (%)': proba2[:, 0].round(1),
                            'CD (%)': proba2[:, 1].round(1),
                            'Conf. Gap': (np.abs(proba2[:, 1] - proba2[:, 0])).round(1)
                        })
                        
                        # 显示配置
                        st.write("### Stage2 Subtype Classification")
                        st.dataframe(
                            results_stage2.sort_values('Conf. Gap', ascending=False),
                            hide_index=True,
                            column_config={...}  # 保持原配置
                        )
                        
                        # 完成状态更新
                        cd_count = sum(stage2_pred)
                        status2.update(
                            label=f"Stage2 Complete: {cd_count} CD | {len(stage2_pred)-cd_count} UC ✅",
                            state="complete"
                        )
                    
                    except Exception as e:
                        status2.update(label="Stage2 Failed ❌", state="error")
                        st.error(f"Stage2 Error: {str(e)}")
                
        # ======================
        # 统一置信度说明
        # ======================
        st.divider()
        with st.expander("ℹ️ Interpretation Guidelines"):
            st.markdown("""
            **Confidence Evaluation Criteria**  
            ▾▾▾▾▾▾▾▾▾▾▾▾▾▾▾▾▾▾
            - 🟢 **High Reliability (Conf. Gap ≥30%)**  
              临床结论可信度高，可直接用于诊断决策
            - 🟡 **Moderate Reliability (15% ≤ Gap <30%)**  
              建议结合其他临床指标综合判断
            - 🔴 **Low Reliability (Gap <15%)**  
              需人工复核检测数据或重新采样
            """)
                    
