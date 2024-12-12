import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 今回使用するデータ
from sklearn.datasets import load_iris

# 学習データとテストデータを分ける
from sklearn.model_selection import train_test_split

# 決定木を使う
from sklearn.tree import DecisionTreeClassifier

# 正解率、予測結果を算出するやつ
from sklearn.metrics import accuracy_score


st.title("Iris data")

iris = load_iris()

# 説明変数
X =pd.DataFrame(iris.data, columns=iris.feature_names)

# 目的変数
y = pd.Series(iris.target, name = "species")




