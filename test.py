# Import libraries
from sklearn.tree import plot_tree
import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from st_circular_progress import CircularProgress
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Diabetes Melitus Prediction",
    page_icon="✅",
    layout="wide",
)

# Load data
df = pd.read_csv("data\df_clear.csv")

# data preparation
X = df.drop("Hasil_Tes", axis = 1)
y = df["Hasil_Tes"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# css
st.markdown("""
<style>
.hero {
  background-image: url("https://img.freepik.com/free-vector/vector-white-gradient-background-modern-design_361591-4420.jpg?t=st=1727968064~exp=1727971664~hmac=54a41fc43c36b4a04fe0dfd5c479bde252d7ceac420386dc1c4ff674d7c6d02a&w=996");
  background-size: cover;
  background-position: center;
  height: 300px;
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>

<div class="hero">
  <h1>Prediksi Diabetes Melitus Untuk Anak Usia Remaja<br>Dengan Random Forest</br></h1>
</div>
""", unsafe_allow_html=True)

# Create a CSV file to store the input data
data_file = 'input_data.csv'
if not os.path.exists(data_file):
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=["Usia", "Jenis_Kelamin", "BMI", "Kadar_Gula_Darah", "Tekanan_Darah", "HbA1c", "Hasil_Tes"])
    df.to_csv(data_file, index=False)

# Load input data into session state
if 'input_data' not in st.session_state:
    st.session_state.input_data = pd.read_csv(data_file)
    st.session_state.input_data.drop_duplicates(inplace=True)
# Tabs
tab1, tab2, tab3 = st.tabs(["Prediksi", "Data", "Visual"])
with tab1:
    #form
    st.subheader("Prediksi Diabetes Melitus Untuk Anak Usia Remaja")
    with st.form("diabetes_form"):
        Usia = st.slider("Usia (Age)", 10, 19, 10)
        Jenis_Kelamin = st.selectbox("Jenis Kelamin (Gender)", ["Laki-laki", "Perempuan"])
        Jenis_Kelamin_value = 1 if Jenis_Kelamin == "Laki-laki" else 0
        BMI = st.number_input("BMI", min_value=10.0, max_value=35.0, value=20.0)
        Kadar_Gula_Darah = st.slider("Kadar Gula Darah", 80, 240, 140)
        Tekanan_Darah = st.slider("Tekanan Darah", 70, 180, 100)
        HbA1c = st.number_input("HbA1c", min_value=3.0, max_value=10.0, value=4.0)
        submitted = st.form_submit_button("Predict")

        # If the form is submitted, predict the probability of diabetes mellitus
    # If the form is submitted, predict the probability of diabetes mellitus
    if submitted:
        input_values = pd.DataFrame([[Usia, Jenis_Kelamin_value, BMI, Kadar_Gula_Darah, Tekanan_Darah, HbA1c]],
                                    columns=["Usia", "Jenis_Kelamin", "BMI", "Kadar_Gula_Darah", "Tekanan_Darah", "HbA1c"])
        probability = model.predict_proba(input_values)
        probability_scalar = probability[0, 1]
        hasil_tes = 1 if probability_scalar > 0.5 else 0
        new_data = pd.DataFrame([[Usia, Jenis_Kelamin_value, BMI, Kadar_Gula_Darah, Tekanan_Darah, HbA1c, hasil_tes]],
                                columns=["Usia", "Jenis_Kelamin", "BMI", "Kadar_Gula_Darah", "Tekanan_Darah", "HbA1c", "Hasil_Tes"])
        st.session_state.input_data = pd.concat([st.session_state.input_data, new_data], ignore_index=True)

        # Drop duplicate
        st.session_state.input_data.drop_duplicates(inplace=True)
        # Save the updated input data
        st.session_state.input_data.to_csv(data_file, index=False)
        # Display the predicted probability
        st.write("Random Forest menghasilkan kemungkinan", str(round(probability_scalar * 100)) + "%", "Terkena diabetes mellitus.")
        if hasil_tes == 1:
            st.write("Anda terduga terkena diabetes mellitus.")
        else:
            st.write("Anda terduga tidak terkena diabetes mellitus.")

    # Display the DataFrame
    st.subheader("Input Data")
    st.dataframe(st.session_state.input_data, hide_index=True, use_container_width=True)

with tab2:
    #row 1
    col1, col2 = st.columns(2)
    with col1:
        #data
        st.subheader("Data")
        st.dataframe(df, hide_index=True)
    with col2:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(classification_report(y_test, model.predict(X_test), output_dict=True)).T
        st.dataframe(report_df, use_container_width=True)
    #row 2
    col1, col2 = st.columns(2)
    with col1:
        #correlation matrix
        st.subheader("Correlation Matrix")
        corr_matrix = df.corr()
        fig = px.imshow(corr_matrix, 
                color_continuous_scale='Mint', text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        #confusion matrix
        y_pred = model.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig = px.imshow(conf_mat, color_continuous_scale='Mint', text_auto=True)
        fig.update_layout(
            xaxis_title='Predicted labels',
            yaxis_title='True labels',
        )
        st.plotly_chart(fig)
    #row 3
    # Streamlit app
    st.subheader("Random Forest Visualization")

    # Select a tree to visualize
    tree_index = st.slider("Select a tree index:", 0, model.n_estimators - 1)

    # Highlight the selected tree
    st.subheader(f"Selected Tree: {tree_index + 1}")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_tree(model.estimators_[tree_index], feature_names=X.columns, class_names=['Normal', 'Diabetes'], ax=ax, filled=True)
    st.pyplot(fig)
    #with col1:
        #st.write(model.intercept_)
        #st.write(model.coef_)
        #classification report

    #with col2:
        #sigmoid curve
        #st.subheader('Sigmoid Curve')

with tab3:
    #row 1 metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Count of Data", len(df))
    with col2:
        st.metric("Jenis Kelamin Laki Laki", len(df[df['Jenis_Kelamin'] == 1]))
    with col3:
        st.metric("Jenis Kelamin Perempuan", len(df[df['Jenis_Kelamin'] == 0]))
    with col4:
        st.metric("Normal ", len(df[df['Hasil_Tes'] == 0])) 
    with col5:
        st.metric("Terduga diabetes", len(df[df['Hasil_Tes'] == 1]))
    st.markdown("---")
    #row 2 visual variables
    column = st.selectbox("Visualisai Variabel", df.columns)
    if column in ['Jenis_Kelamin', 'Hasil_Tes']:
        df_count = df[column].value_counts().reset_index()
        df_count.columns = [column, 'Count']
        if column == 'Jenis_Kelamin':
            jenis_kelamin_map = {0: 'Perempuan', 1: 'Laki-laki'}
            df_count[column] = df_count[column].map(jenis_kelamin_map)
        elif column == 'Hasil_Tes':
            hasil_tes_map = {0: 'Normal', 1: 'Diabetes'}
            df_count[column] = df_count[column].map(hasil_tes_map)
        fig = go.Figure(data=[go.Pie(labels=df_count[column], values=df_count['Count'], 
                                    marker_colors=['#2E4F4F', '#0E8388'], 
                                    textposition='inside', 
                                    textinfo='percent', 
                                    textfont_color='white')])
        fig.update_layout(title='Distribusi Variabel {}'.format(column))
        st.plotly_chart(fig)
    elif column in ['BMI', 'Kadar_Gula_Darah', 'Tekanan_Darah', 'HbA1c']:
        df_count = df[column].value_counts().reset_index()
        df_count.columns = [column, 'Count']
        fig = go.Figure(data=[go.Scatter(x=df_count[column], y=df_count['Count'], mode='markers', 
                                        marker_color='#0E8388')])
        fig.update_layout(title='Distribusi Variabel {}'.format(column), xaxis_title=column, yaxis_title='Count')
        st.plotly_chart(fig)
    else:
        fig = go.Figure(data=[go.Bar(x=df[column].value_counts().index, y=df[column].value_counts().values, 
                                    marker_color='#0E8388', width=0.9)])
        fig.update_layout(title='Distribusi Variabel {}'.format(column), barmode='group')
        st.plotly_chart(fig)
    #row 3 filter
    st.markdown("---")
    unique_Usia = pd.unique(df["Usia"])
    min_Usia = unique_Usia.min()
    max_Usia = unique_Usia.max()
    filter = st.slider("Filter Berdasarkan Usia", min_Usia, max_Usia, min_Usia)
    df_filter = df[df["Usia"] == filter]
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        jenis_kelamin_map = {0: 'Perempuan', 1: 'Laki-laki'}
        df_count = df_filter['Jenis_Kelamin'].value_counts().reset_index()
        df_count.columns = ['Jenis_Kelamin', 'Count']
        df_count['Jenis_Kelamin'] = df_count['Jenis_Kelamin'].map(jenis_kelamin_map)

        fig = go.Figure(data=[go.Pie(labels=df_count['Jenis_Kelamin'], values=df_count['Count'], marker_colors=['#0E8388','#2E4F4F'], textposition='inside',textinfo='percent', textfont_color='white')])
        fig.update_layout(title=f"Distribusi Variabel Jenis Kelamin pada Usia {filter}")
        st.plotly_chart(fig)
    with kpi2:
        hasil_tes_map = {0: 'Normal', 1: 'Diabetes'}
        df_count = df_filter['Hasil_Tes'].value_counts().reset_index()
        df_count.columns = ['Hasil_Tes', 'Count']
        df_count['Hasil_Tes'] = df_count['Hasil_Tes'].map(hasil_tes_map)

        fig = go.Figure(data=[go.Pie(labels=df_count['Hasil_Tes'], values=df_count['Count'], marker_colors=['#0E8388', '#2E4F4F'], textposition='inside', textinfo='percent', textfont_color='white')])
        fig.update_layout(title=f"Distribusi Variabel Hasil Tes pada Usia {filter}")
        st.plotly_chart(fig)

    st.markdown("---")
    #row 4 visual for 2 variabel
    col1, col2 = st.columns(2)
    with col1:
        select_x = st.selectbox('Pilih Variabel', options=['Usia','Jenis_Kelamin','BMI','Kadar_Gula_Darah','Tekanan_Darah','HbA1c','Hasil_Tes'], key='x_column')
    with col2:
        options_y = [option for option in ['Usia','Jenis_Kelamin','BMI','Kadar_Gula_Darah','Tekanan_Darah','HbA1c','Hasil_Tes'] if option != select_x]
        select_y = st.selectbox('Pilih Variabel', options=options_y, key='y_column')
    with st.container():
        #visuaal usia
        if (select_x == 'Usia' and select_y == 'Jenis_Kelamin') or (select_x == 'Jenis_Kelamin' and select_y == 'Usia'):
            usia_jenis_kelamin_counts = df.groupby(['Usia', 'Jenis_Kelamin']).size().reset_index(name='Count')
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=usia_jenis_kelamin_counts[usia_jenis_kelamin_counts['Jenis_Kelamin'] == 0]['Usia'],
                y=usia_jenis_kelamin_counts[usia_jenis_kelamin_counts['Jenis_Kelamin'] == 0]['Count'],
                name='Perempuan',
                marker_color='#2E4F4F'
            ))
            fig.add_trace(go.Bar(
                x=usia_jenis_kelamin_counts[usia_jenis_kelamin_counts['Jenis_Kelamin'] == 1]['Usia'],
                y=usia_jenis_kelamin_counts[usia_jenis_kelamin_counts['Jenis_Kelamin'] == 1]['Count'],
                name='Laki-laki',
                marker_color='#0E8388'
            ))
            fig.update_layout(barmode='group', xaxis_tickangle=-45, title="Hubungan Antara Usia dan Jenis Kelamin", xaxis_title="Usia", yaxis_title="Jumlah")
            st.plotly_chart(fig)

        elif (select_x == 'Usia' and select_y == 'BMI') or (select_x == 'BMI' and select_y == 'Usia'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'Usia' and select_y == 'Kadar_Gula_Darah') or (select_x == 'Kadar_Gula_Darah' and select_y == 'Usia'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'Usia' and select_y == 'Tekanan_Darah') or (select_x == 'Tekanan_Darah' and select_y == 'Usia'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'Usia' and select_y == 'HbA1c') or (select_x == 'HbA1c' and select_y == 'Usia'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'Usia' and select_y == 'Hasil_Tes') or (select_x == 'Hasil_Tes' and select_y == 'Usia'):
            usia_hasil_tes_counts = df.groupby(['Usia', 'Hasil_Tes']).size().reset_index(name='Count')
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=usia_hasil_tes_counts[usia_hasil_tes_counts['Hasil_Tes'] == 1]['Usia'],
                y=usia_hasil_tes_counts[usia_hasil_tes_counts['Hasil_Tes'] == 1]['Count'],
                name='Diabetes',
                marker_color='#0E8388'
            ))
            fig.add_trace(go.Bar(
                x=usia_hasil_tes_counts[usia_hasil_tes_counts['Hasil_Tes'] == 0]['Usia'],
                y=usia_hasil_tes_counts[usia_hasil_tes_counts['Hasil_Tes'] == 0]['Count'],
                name='Normal',
                marker_color='#2E4F4F'
            ))
            fig.update_layout(barmode='group', xaxis_tickangle=-45, title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title="Jumlah")
            st.plotly_chart(fig)

        #visual jenis kelamin
        elif (select_x == 'Jenis_Kelamin' and select_y == 'BMI') or (select_x == 'BMI' and select_y == 'Jenis_Kelamin'):
            jenis_kelamin_map = {0: 'Perempuan', 1: 'Laki-laki'}
            df_count = df.groupby(['BMI', 'Jenis_Kelamin']).size().reset_index(name='Count')
            fig = go.Figure()
            for jenis_kelamin, name in jenis_kelamin_map.items():
                df_jenis_kelamin = df_count[df_count['Jenis_Kelamin'] == jenis_kelamin]
                fig.add_trace(go.Scatter(x=df_jenis_kelamin['BMI'], y=df_jenis_kelamin['Count'], mode='markers',name=name, marker_color=['#2E4F4F', '#0E8388'][jenis_kelamin]))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title="BMI", yaxis_title="Jumlah")
            st.plotly_chart(fig)

        elif (select_x == 'Jenis_Kelamin' and select_y == 'Kadar_Gula_Darah') or (select_x == 'Kadar_Gula_Darah' and select_y == 'Jenis_Kelamin'):
            jenis_kelamin_map = {0: 'Perempuan', 1: 'Laki-laki'}
            df_count = df.groupby(['Kadar_Gula_Darah', 'Jenis_Kelamin']).size().reset_index(name='Count')
            fig = go.Figure()
            for jenis_kelamin, name in jenis_kelamin_map.items():
                df_jenis_kelamin = df_count[df_count['Jenis_Kelamin'] == jenis_kelamin]
                fig.add_trace(go.Scatter(x=df_jenis_kelamin['Kadar_Gula_Darah'], y=df_jenis_kelamin['Count'], mode='markers', name=name, marker_color=['#2E4F4F', '#0E8388'][jenis_kelamin]))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title="Kadar Gula Darah", yaxis_title="Jumlah")
            st.plotly_chart(fig)

        elif (select_x == 'Jenis_Kelamin' and select_y == 'Tekanan_Darah') or (select_x == 'Tekanan_Darah' and select_y == 'Jenis_Kelamin'):
            jenis_kelamin_map = {0: 'Perempuan', 1: 'Laki-laki'}
            df_count = df.groupby(['Tekanan_Darah', 'Jenis_Kelamin']).size().reset_index(name='Count')
            fig = go.Figure()
            for jenis_kelamin, name in jenis_kelamin_map.items():
                df_jenis_kelamin = df_count[df_count['Jenis_Kelamin'] == jenis_kelamin]
                fig.add_trace(go.Scatter(x=df_jenis_kelamin['Tekanan_Darah'], y=df_jenis_kelamin['Count'], mode='markers', name=name, marker_color=['#2E4F4F', '#0E8388'][jenis_kelamin]))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title="Tekanan Darah", yaxis_title="Jumlah")
            st.plotly_chart(fig)

        elif (select_x == 'Jenis_Kelamin' and select_y == 'HbA1c') or (select_x == 'HbA1c' and select_y == 'Jenis_Kelamin'):
            jenis_kelamin_map = {0: 'Perempuan', 1: 'Laki-laki'}
            df_count = df.groupby(['HbA1c', 'Jenis_Kelamin']).size().reset_index(name='Count')
            fig = go.Figure()
            for jenis_kelamin, name in jenis_kelamin_map.items():
                df_jenis_kelamin = df_count[df_count['Jenis_Kelamin'] == jenis_kelamin]
                fig.add_trace(go.Scatter(
                    x=df_jenis_kelamin['HbA1c'],
                    y=df_jenis_kelamin['Count'],
                    mode='markers',
                    name=name,
                    marker_color=['#2E4F4F', '#0E8388'][jenis_kelamin]
                ))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title="HbA1c", yaxis_title="Jumlah")
            st.plotly_chart(fig)

        elif (select_x == 'Jenis_Kelamin' and select_y == 'Hasil_Tes') or (select_x == 'Hasil_Tes' and select_y == 'Jenis_Kelamin'):
            jenis_kelamin_map = {0: 'Perempuan', 1: 'Laki-laki'}
            hasil_tes_map = {0: 'Normal', 1: 'Diabetes'}
            df_count = df.groupby(['Jenis_Kelamin', 'Hasil_Tes']).size().reset_index(name='Count')
            labels = ['Perempuan Normal', 'Perempuan Diabetes', 'Laki-laki Normal', 'Laki-laki Diabetes']
            values = [
                df_count[(df_count['Jenis_Kelamin'] == 0) & (df_count['Hasil_Tes'] == 0)]['Count'].values[0],
                df_count[(df_count['Jenis_Kelamin'] == 0) & (df_count['Hasil_Tes'] == 1)]['Count'].values[0],
                df_count[(df_count['Jenis_Kelamin'] == 1) & (df_count['Hasil_Tes'] == 0)]['Count'].values[0],
                df_count[(df_count['Jenis_Kelamin'] == 1) & (df_count['Hasil_Tes'] == 1)]['Count'].values[0]
            ]
            colors = ['#29A18D', '#2E4F4F', '#0E8388', '#29A19C']
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors, textposition='inside', textinfo='percent', textfont_color='white')])
            fig.update_layout(title=f"{select_x} vs {select_y}")
            st.plotly_chart(fig)

        #visual bmi
        elif (select_x == 'BMI' and select_y == 'Kadar_Gula_Darah') or (select_x == 'Kadar_Gula_Darah' and select_y == 'BMI'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'BMI' and select_y == 'Tekanan_Darah') or (select_x == 'Tekanan_Darah' and select_y == 'BMI'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'BMI' and select_y == 'HbA1c') or (select_x == 'HbA1c' and select_y == 'BMI'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'BMI' and select_y == 'Hasil_Tes') or (select_x == 'Hasil_Tes' and select_y == 'BMI'):
            hasil_tes_map = {0: 'Normal', 1: 'Diabetes'}
            df_count = df.groupby(['BMI', 'Hasil_Tes']).size().reset_index(name='Count')
            fig = go.Figure()
            for hasil_tes, name in hasil_tes_map.items():
                df_hasil_tes = df_count[df_count['Hasil_Tes'] == hasil_tes]
                fig.add_trace(go.Scatter(x=df_hasil_tes['BMI'], y=df_hasil_tes['Count'], mode='markers', name=name, marker_color=['#2E4F4F', '#0E8388'][hasil_tes]))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title="BMI", yaxis_title="Jumlah")
            st.plotly_chart(fig)

        #visual kadar gula darah
        elif (select_x == 'Kadar_Gula_Darah' and select_y == 'Tekanan_Darah') or (select_x == 'Tekanan_Darah' and select_y == 'Kadar_Gula_Darah'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'Kadar_Gula_Darah' and select_y == 'HbA1c') or (select_x == 'HbA1c' and select_y == 'Kadar_Gula_Darah'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)
        elif (select_x == 'Kadar_Gula_Darah' and select_y == 'Hasil_Tes') or (select_x == 'Hasil_Tes' and select_y == 'Kadar_Gula_Darah'):
            hasil_tes_map = {0: 'Normal', 1: 'Diabetes'}
            df_count = df.groupby(['Kadar_Gula_Darah', 'Hasil_Tes']).size().reset_index(name='Count')
            fig = go.Figure()
            for hasil_tes, name in hasil_tes_map.items():
                df_hasil_tes = df_count[df_count['Hasil_Tes'] == hasil_tes]
                fig.add_trace(go.Scatter(x=df_hasil_tes['Kadar_Gula_Darah'], y=df_hasil_tes['Count'], mode='markers', name=name, marker_color=['#2E4F4F','#0E8388'][hasil_tes]))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title="Kadar Gula Darah", yaxis_title="Jumlah")
            st.plotly_chart(fig)
        #vissual tekanan darah
        elif (select_x == 'Tekanan_Darah' and select_y == 'HbA1c') or (select_x == 'HbA1c' and select_y == 'Tekanan_Darah'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[select_x], y=df[select_y], mode='markers', marker_color='#2E4F4F'))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title=select_x, yaxis_title=select_y)
            st.plotly_chart(fig)

        elif (select_x == 'Tekanan_Darah' and select_y == 'Hasil_Tes') or (select_x == 'Hasil_Tes' and select_y == 'Tekanan_Darah'):
            hasil_tes_map = {0: 'Normal', 1: 'Diabetes'}
            df_count = df.groupby(['Tekanan_Darah', 'Hasil_Tes']).size().reset_index(name='Count')
            fig = go.Figure()
            for hasil_tes, name in hasil_tes_map.items():
                df_hasil_tes = df_count[df_count['Hasil_Tes'] == hasil_tes]
                fig.add_trace(go.Scatter(x=df_hasil_tes['Tekanan_Darah'], y=df_hasil_tes['Count'], mode='markers', name=name, marker_color=['#2E4F4F', '#0E8388'][hasil_tes]))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title="Tekanan Darah", yaxis_title="Jumlah")
            st.plotly_chart(fig)

        #visual hba1c
        elif (select_x == 'HbA1c' and select_y == 'Hasil_Tes') or (select_x == 'Hasil_Tes' and select_y == 'HbA1c'):
            hasil_tes_map = {0: 'Normal', 1: 'Diabetes'}
            df_count = df.groupby(['HbA1c', 'Hasil_Tes']).size().reset_index(name='Count')
            fig = go.Figure()
            for hasil_tes, name in hasil_tes_map.items():
                df_hasil_tes = df_count[df_count['Hasil_Tes'] == hasil_tes]
                fig.add_trace(go.Scatter(x=df_hasil_tes['HbA1c'], y=df_hasil_tes['Count'], mode='markers', name=name, marker_color=['#2E4F4F', '#0E8388'][hasil_tes]))
            fig.update_layout(title=f"{select_x} vs {select_y}", xaxis_title="HbA1c", yaxis_title="Jumlah")
            st.plotly_chart(fig)