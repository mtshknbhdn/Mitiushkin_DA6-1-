import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Дашборд компаній",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("streamlit_dataset (2).csv")
        return df
    except FileNotFoundError:
        st.error("Файл 'streamlit_dataset (2).csv' не знайдено. Будь ласка, переконайтеся, що він знаходиться в одній папці з додатком.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.sidebar.title("Панель фільтрації")
    st.sidebar.markdown("""
    Використовуйте ці фільтри для налаштування відображення даних на головній панелі.
    """)

    available_years = sorted(df["Year"].dropna().unique())
    selected_year = st.sidebar.selectbox("Оберіть рік:", available_years)

    available_regions = df["Region"].dropna().unique()
    selected_regions = st.sidebar.multiselect(
        "Оберіть регіон(и):", 
        options=available_regions, 
        default=available_regions
    )

    available_industries = df["Industry"].dropna().unique()
    selected_industry = st.sidebar.radio("Оберіть галузь:", available_industries)

    show_map = st.sidebar.checkbox("Показати карту компаній", value=False)
    
    df_filtered = df[
        (df["Year"] == selected_year) &
        (df["Region"].isin(selected_regions)) &
        (df["Industry"] == selected_industry)
    ]

    st.title("Дашборд компаній")
    st.subheader(f"Відфільтровано {df_filtered.shape[0]} компаній")

    st.markdown("### Таблиця результатів")
    st.markdown("Оберіть колонки, які ви хочете бачити в таблиці:") 
   
    all_columns = df_filtered.columns.tolist()
    default_columns = ["Company", "Region", "Industry", "Revenue", "Profit", "ROI"]
    
    default_columns = [col for col in default_columns if col in all_columns]

    selected_columns = st.multiselect(
        "Стовпці для відображення",
        options=all_columns,
        default=default_columns
    )

    if selected_columns:
        st.dataframe(df_filtered[selected_columns], use_container_width=True)
    else:
        st.info("Оберіть хоча б одну колонку для відображення.")

    if show_map:
        st.markdown("### Географія компаній")
        if "Latitude" in df_filtered.columns and "Longitude" in df_filtered.columns:
            map_data = df_filtered[["Latitude", "Longitude"]].dropna().rename(
                columns={"Latitude": "latitude", "Longitude": "longitude"}
            )
            if not map_data.empty:
                st.map(map_data)
            else:
                st.warning("Немає достатньо даних координат для відображення на карті.")
        else:
            st.warning("У датасеті відсутні колонки Latitude або Longitude.")

    st.markdown("---")
    
    st.markdown("### Візуалізація даних")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Scatter Plot: Витрати та Дохід**")
        if "Expenses" in df_filtered.columns and "Revenue" in df_filtered.columns:
            scatter_chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
                x='Expenses:Q',
                y='Revenue:Q',
                color='Region:N',
                tooltip=['Company', 'Expenses', 'Revenue', 'Region']
            ).interactive()
            st.altair_chart(scatter_chart, use_container_width=True)
        else:
            st.warning("Необхідні дані (Expenses, Revenue) відсутні.")

    with col2:
        st.markdown("**2. Boxplot: Розподіл прибутку за регіонами**")
        if "Region" in df_filtered.columns and "Profit" in df_filtered.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_filtered, x="Region", y="Profit", ax=ax, palette="Set2")
            ax.set_title("Прибуток по регіонах")
            st.pyplot(fig)
        else:
            st.warning("Необхідні дані (Region, Profit) відсутні.")

    st.markdown("**3. Bar Chart: Кількість клієнтів по компаніях**")
    if "Company" in df_filtered.columns and "Customers" in df_filtered.columns:
        bar_chart = px.bar(
            df_filtered, 
            x='Company', 
            y='Customers', 
            color='Customers',
            title='Клієнтська база',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(bar_chart, use_container_width=True)
    else:
        st.warning("Необхідні дані (Company, Customers) відсутні.")

    st.markdown("---")

    st.markdown("### Модель лінійної регресії")
    st.markdown("Дослідіть лінійну залежність між двома числовими показниками.")
    
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) >= 2:
        reg_col1, reg_col2 = st.columns(2)
        with reg_col1:
            x_var = st.selectbox("Оберіть незалежну змінну (X):", numeric_cols, index=0)
        with reg_col2:
            y_index = 1 if len(numeric_cols) > 1 else 0
            y_var = st.selectbox("Оберіть залежну змінну (Y):", numeric_cols, index=y_index)

        build_model = st.button("Побудувати модель")

        if build_model:
            df_model = df_filtered[[x_var, y_var]].dropna()
            
            if df_model.shape[0] >= 3:
                X = df_model[[x_var]]
                y = df_model[y_var]
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                r_sq = model.score(X, y)
                coef = model.coef_[0]
                intercept = model.intercept_
                
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Коефіцієнт детермінації (R²)", f"{r_sq:.4f}")
                res_col2.metric("Коефіцієнт нахилу (β)", f"{coef:.4f}")
                res_col3.metric("Вільний член (Intercept)", f"{intercept:.4f}")
                
                fig_reg, ax_reg = plt.subplots(figsize=(8, 4))
                sns.scatterplot(x=df_model[x_var], y=df_model[y_var], ax=ax_reg, color="blue", label="Фактичні дані")
                sns.lineplot(x=df_model[x_var], y=y_pred, ax=ax_reg, color="red", label="Лінія регресії")
                ax_reg.set_title(f"Лінійна регресія: {y_var} залежно від {x_var}")
                st.pyplot(fig_reg)
            else:
                st.warning("Недостатньо даних для побудови регресії (потрібно хоча б 3 записи без пропусків).")
    else:
        st.info("У датасеті недостатньо числових колонок для побудови регресії.")

else:
    st.info("Очікування завантаження даних")
