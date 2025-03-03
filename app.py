import streamlit as st
import pandas as pd
import joblib


# ------------------------------
# 1. 模型加载函数
# ------------------------------
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    """
    加载保存好的模型文件
    """
    return joblib.load(model_path)


# 加载模型文件（确保这些文件已上传到项目目录中）
catboost_model = load_model('IBD_vs_HC_best_model.pkl')
lightgbm_model = load_model('CD_vs_UC_best_model.pkl')


# ------------------------------
# 2. 数据预处理函数
# ------------------------------
def preprocess_data(df):
    """
    根据训练时的预处理流程，对上传的数据进行预处理。
    假设数据格式为：第一行包含标签信息（从第二列开始），第一列为物种名称，
    第二行及以后为各样本的相对丰度数据。
    """
    try:
        # 获取标签信息：第一行第二列开始的部分
        labels = df.iloc[0, 1:].replace({'IBD': 1, 'HC': 0}).astype(int)
        # 获取特征矩阵：从第二行开始，第一列为物种名称
        features = df.iloc[1:, 1:].copy()
        # 简化物种名称：以 "s__" 分割并取最后一部分
        features.index = df.iloc[1:, 0].str.split('s__').str[-1]
        # 将特征数据转换为数值型，缺失值填充为 0
        features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
        # 转置为样本×特征格式
        return features.T, labels
    except Exception as e:
        st.error(f"Data preprocessing error: {str(e)}")
        return None, None


# ------------------------------
# 3. Streamlit 应用主体
# ------------------------------
st.title("IBD Diagnosis and Staging Online System")
st.write("""
This application enables non-invasive IBD diagnosis and subtyping based on a two-stage machine learning model:
1. The first stage utilizes the CatBoost model to differentiate between IBD and healthy controls;
2. In the second stage, the LightGBM model was used to further differentiate Crohn's Disease (CD) from Ulcerative Colitis (UC) in samples predicted to have IBD.
""")

# 侧边栏上传数据
st.sidebar.header("Upload input data")
uploaded_file = st.sidebar.file_uploader("Please upload a CSV or Excel file containing the feature data.", type=["csv", "xlsx"])

if uploaded_file is not None:
    # 根据文件类型读取数据
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {str(e)}")

    st.write("### Upload data preview")
    st.write(df.head())

    # 对上传数据进行预处理
    X, y_dummy = preprocess_data(df)
    if X is None:
        st.error("Data preprocessing failed, please check if the file format is correct.")
    else:
        st.write(f"Shape of preprocessed data:{X.shape}")

        # 提供按钮进行预测
        if st.sidebar.button("Start forecasting"):
            # 第一阶段预测：IBD vs. Healthy
            # 这里假设模型直接输出类别，1 表示 IBD，0 表示 Healthy
            stage1_pred = catboost_model.predict(X)

            if stage1_pred[0] == 1:
                st.success("Predicted outcome: IBD")
                # 第二阶段预测：CD vs. UC
                stage2_pred = lightgbm_model.predict(X)
                # 假设输出 1 表示 CD，0 表示 UC
                if stage2_pred[0] == 1:
                    st.info("Staging results: Crohn's Disease (CD)")
                else:
                    st.info("Staging results: Ulcerative Colitis (UC)")
            else:
                st.success("Predicted outcome: Healthy")
