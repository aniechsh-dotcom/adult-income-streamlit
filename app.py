
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

@st.cache_resource
def train_gb_best():
    df_dirty = pd.read_csv("data.adult.csv")

    df_clean = df_dirty[~(df_dirty == "?").any(axis=1)]

    target_col_raw = [c for c in df_clean.columns if "50K" in c][0]

    df = df_clean.copy()
    df[target_col_raw] = df[target_col_raw].map({"<=50K": 0, ">50K": 1})
    df = df.rename(columns={target_col_raw: "income"})

    y = df["income"].copy()
    X = df.drop("income", axis=1)

    X_num = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    cat_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), X_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    gb_clf = GradientBoostingClassifier(
        n_estimators=80,
        criterion="squared_error",
        max_features=None,
        random_state=42,
    )

    gb_best = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", gb_clf),
        ]
    )

    gb_best.fit(X, y)

    return gb_best, X_num, cat_cols


gb_best, X_num, cat_cols = train_gb_best()


st.set_page_config(page_title="Прогноз дохода >50K", page_icon="💰", layout="centered")

st.title("💰 Прогноз: превысит ли доход 50K?")
st.write(
    "Модель: градиентный бустинг (80 деревьев), обученный на датасете Adult. "
    "Введите свои характеристики — и модель оценит вероятность дохода выше $50K."
)

st.header("Введите данные")

input_data = {}


st.subheader("Числовые признаки")

if "age" in X_num:
    input_data["age"] = st.number_input("Возраст (age)", min_value=16, max_value=90, value=30)

if "fnlwgt" in X_num:
    input_data["fnlwgt"] = st.number_input(
        "fnlwgt", min_value=0, max_value=1_500_000, value=100_000
    )

if "education-num" in X_num:
    input_data["education-num"] = st.number_input(
        "Education-num", min_value=0, max_value=20, value=10
    )

if "capital-gain" in X_num:
    input_data["capital-gain"] = st.number_input(
        "Capital gain", min_value=0, max_value=100_000, value=0
    )

if "capital-loss" in X_num:
    input_data["capital-loss"] = st.number_input(
        "Capital loss", min_value=0, max_value=5_000, value=0
    )

if "hours-per-week" in X_num:
    input_data["hours-per-week"] = st.number_input(
        "Часы работы в неделю (hours-per-week)", min_value=1, max_value=99, value=40
    )

st.subheader("Категориальные признаки")

if "workclass" in cat_cols:
    input_data["workclass"] = st.selectbox(
        "Тип занятости (workclass)",
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
    )

if "education" in cat_cols:
    input_data["education"] = st.selectbox(
        "Образование (education)",
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
    )

if "marital-status" in cat_cols:
    input_data["marital-status"] = st.selectbox(
        "Семейное положение (marital-status)",
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
    )

if "occupation" in cat_cols:
    input_data["occupation"] = st.selectbox(
        "Профессия (occupation)",
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
    )

if "relationship" in cat_cols:
    input_data["relationship"] = st.selectbox(
        "Статус в семье (relationship)",
        ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    )

if "race" in cat_cols:
    input_data["race"] = st.selectbox(
        "Раса (race)",
        [
            "White",
            "Asian-Pac-Islander",
            "Amer-Indian-Eskimo",
            "Other",
            "Black",
        ],
    )

if "sex" in cat_cols:
    input_data["sex"] = st.selectbox(
        "Пол (sex)",
        ["Female", "Male"],
    )


if st.button("Спрогнозировать доход"):
    input_df = pd.DataFrame([input_data])

    proba = gb_best.predict_proba(input_df)[0, 1]
    pred_class = int(proba >= 0.5)

    st.write("---")
    st.write(f"**Вероятность дохода > 50K:** {proba:.3f}")

    if pred_class == 1:
        st.success("Скорее всего, доход пользователя **выше 50K** 💸")
    else:
        st.info("Скорее всего, доход пользователя **не превышает 50K**.")

