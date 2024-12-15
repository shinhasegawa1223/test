# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 今回使用するデータ
# from sklearn.datasets import load_iris

# # 学習データとテストデータを分ける
# from sklearn.model_selection import train_test_split

# # 決定木を使う
# from sklearn.tree import DecisionTreeClassifier

# # 正解率、予測結果を算出するやつ
# from sklearn.metrics import accuracy_score

# st.title("Iris data")

# iris = load_iris()

# # 説明変数
# X =pd.DataFrame(iris.data, columns=iris.feature_names)

# # 目的変数
# y = pd.Series(iris.target, name = "species")

# # 学習データとテストデータに分割
# X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=0)

# # ベーシックな決定木を使用する
# model = DecisionTreeClassifier()

# model.fit(X_train,y_train)

# # テストデータの予測
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# st.write(accuracy)

# # ユーザー入力
# st.header("アヤメの予測値を入力してください")
# sepal_length =st.number_input("がく片長(cm)0-3", min_value=0, max_value=3)
# sepal_width = st.number_input("がく片幅(cm)0-3", min_value=0, max_value=3)
# petal_length =st.number_input("花びら長(cm)0-3", min_value=0, max_value=3)
# petal_width= st.number_input("花びら幅(cm)0-3", min_value=0, max_value=3)


# input_data = pd.DataFrame({
# "sepal length (cm)":[sepal_length],
# "sepal width (cm)":[sepal_width],
# "petal length (cm)":[petal_length],
# "petal width (cm)":[petal_width],
# })

# if st.button("predict"):
#     prediction = model.predict(input_data)
#     prediction_proba = model.predict_proba(input_data)
#     species = iris.target_names[prediction][0]

#     fig, ax = plt.subplots()

#     # 花びら長, 花びら幅
#     scatter = ax.scatter(X["petal length (cm)"],X["petal width (cm)"], c=y, label = iris.target_names)
#     ax.scatter(petal_length,petal_width , c="red")
#     ax.set_xlabel("petal length (cm)")
#     ax.set_ylabel("petal width (cm)")

#     # 凡例の追加
#     hanrei, _ =scatter.legend_elements(prop="colors")
#     legend_labels = iris.target_names
#     ax.legend(hanrei, legend_labels, title= "Species")

#     st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Irisデータセットのロード
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# 学習データとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 決定木モデルの作成と学習
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# テストデータでの正解率を計算
accuracy = accuracy_score(y_test, model.predict(X_test))

# タイトル
st.write("Iris Data Prediction")

# 正解率の表示
st.markdown(f"<div style='font-size: small;'>Model Accuracy: <b>{accuracy:.2f}</b></div>", unsafe_allow_html=True)

# 入力フォーム

with st.form("input_form"):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1, value=5.0, key="sepal_length")
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1, value=1.5, key="petal_length")
    with col2:
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0, key="sepal_width")
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1, value=0.5, key="petal_width")

    submitted = st.form_submit_button("Predict")

if submitted:
    # 入力値のデータフレーム化
    input_data = pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width],
    })

    # 予測
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    species = iris.target_names[prediction][0]

    # 結果の表示
    st.subheader("Prediction")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown(f"<div style='font-size: small;'>Predicted Species: <b>{species}</b></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='font-size: small;'>Prediction Probabilities:</div>", unsafe_allow_html=True)
        for i, prob in enumerate(prediction_proba[0]):
            st.markdown(f"<div style='font-size: small;'>{iris.target_names[i]}: {prob:.2f}</div>", unsafe_allow_html=True)

    # 可視化
    fig, ax = plt.subplots(figsize=(4, 3))
    scatter = ax.scatter(X["petal length (cm)"], X["petal width (cm)"], c=y, cmap="viridis", s=20)
    ax.scatter(petal_length, petal_width, c="red", label="Input", edgecolor="black", s=50)
    ax.set_xlabel("Petal Length (cm)", fontsize=8)
    ax.set_ylabel("Petal Width (cm)", fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    legend_labels = iris.target_names
    handles, _ = scatter.legend_elements(prop="colors")
    ax.legend(handles, legend_labels, title="Species", fontsize=6, title_fontsize=8)
    st.pyplot(fig)
