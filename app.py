import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi YouTube Subscribers", layout="wide")

# Judul dan Deskripsi Umum Dataset
st.title("Dashboard Analisis & Prediksi Jumlah Subscribers YouTube")
st.markdown("""
### Overview Dataset
Dataset ini berisi informasi metrik performa dan profil dari berbagai channel YouTube, seperti jumlah *subscribers*, tayangan bulanan (*monthly views*), total video, rata-rata pengeluaran per video, jumlah tim, hingga durasi video. 
Selain metrik numerik, terdapat pula data kategori berupa frekuensi upload, kualitas video, tipe kreator, kelompok usia audiens, negara asal audiens, dan status video trending.

Dashboard ini dirancang untuk mengeksplorasi data awal secara mentah, serta melatih model machine learning guna memprediksi jumlah *subscribers* secara akurat.
""")

def load_raw_data():
    return pd.read_csv("youtube_channel_subscribers.csv")

def load_and_preprocess_data():
    df = pd.read_csv("youtube_channel_subscribers.csv")

    # Handle duplicates
    df = df.drop_duplicates()

    # Standardize Creator_Type
    df["Creator_Type"] = df["Creator_Type"].replace({
        "multi-niche": "Multi-Niche",
        "Multi Niche": "Multi-Niche",
        "#Multi-Niche": "Multi-Niche",
        "lifestyle": "Lifestyle",
        "#Lifestyle": "Lifestyle",
        "#General": "General",
        "general": "General"
    })

    # Trim string columns
    str_cols = ["Youtuber_Name", "Upload_Frequency", "Video_Quality", "Creator_Type",
                "Audience_Age_Group", "Audience_Country"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Impute missing values for Creator_Type and Video_Quality with mode
    df_impute = df.copy()
    if 'Creator_Type' in df_impute.columns:
        df_impute["Creator_Type"] = df_impute["Creator_Type"].fillna(df_impute["Creator_Type"].mode()[0])
    if 'Video_Quality' in df_impute.columns:
        df_impute["Video_Quality"] = df_impute["Video_Quality"].fillna(df_impute["Video_Quality"].mode()[0])

    # Impute Avg_Spend_Per_Video_USD and Avg_Video_Length_Min using grouped median
    if 'Subscribers' in df_impute.columns:
        df_impute["Subscribers_Group"] = pd.qcut(df_impute["Subscribers"], q=4, labels=False, duplicates='drop')
        if 'Avg_Spend_Per_Video_USD' in df_impute.columns:
            df_impute["Avg_Spend_Per_Video_USD"] = df_impute["Avg_Spend_Per_Video_USD"].fillna(
                df_impute.groupby("Subscribers_Group")["Avg_Spend_Per_Video_USD"].transform("median"))
        df_impute = df_impute.drop(columns=["Subscribers_Group"])

    if 'Total_Videos' in df_impute.columns:
        df_impute["Total_Videos_Group"] = pd.qcut(df_impute["Total_Videos"], q=4, labels=False, duplicates='drop')
        if 'Avg_Video_Length_Min' in df_impute.columns:
            df_impute["Avg_Video_Length_Min"] = df_impute["Avg_Video_Length_Min"].fillna(
                df_impute.groupby("Total_Videos_Group")["Avg_Video_Length_Min"].transform("median"))
        df_impute = df_impute.drop(columns=["Total_Videos_Group"])

    # Ensure no remaining NaNs in numeric columns
    numeric_cols = df_impute.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_impute[col] = df_impute[col].fillna(df_impute[col].median())

    # Handle outliers (capping)
    cols_to_treat = ['Subscribers', 'Monthly_Views', 'Avg_Spend_Per_Video_USD', 'Avg_Video_Length_Min']
    for col in cols_to_treat:
        if col in df_impute.columns:
            Q1 = df_impute[col].quantile(0.25)
            Q3 = df_impute[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_impute[col] = np.where(df_impute[col] < lower_bound, lower_bound, df_impute[col])
            df_impute[col] = np.where(df_impute[col] > upper_bound, upper_bound, df_impute[col])

    # Drop identifiers
    cols_to_drop = ['S.No', 'Youtuber_Name']
    for col in cols_to_drop:
        if col in df_impute.columns:
            df_impute = df_impute.drop(columns=[col])

    # One-hot encode categorical columns
    categorical_cols = ['Upload_Frequency', 'Video_Quality', 'Creator_Type', 'Audience_Age_Group', 'Audience_Country']
    categorical_cols_exist = [col for col in categorical_cols if col in df_impute.columns]
    df_impute = pd.get_dummies(df_impute, columns=categorical_cols_exist, drop_first=True)

    return df_impute

def train_and_evaluate_models(X_train, X_test, Y_train, Y_test):
    results = {}
    predictions = {}
    models = {}

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, Y_train)
    Y_pred_lr = lr_model.predict(X_test)
    results['Linear Regression'] = {
        'MAE': mean_absolute_error(Y_test, Y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(Y_test, Y_pred_lr)),
        'R2': r2_score(Y_test, Y_pred_lr)
    }
    predictions['Linear Regression'] = Y_pred_lr
    models['Linear Regression'] = lr_model

    # Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, Y_train)
    Y_pred_dt = dt_model.predict(X_test)
    results['Decision Tree Regressor'] = {
        'MAE': mean_absolute_error(Y_test, Y_pred_dt),
        'RMSE': np.sqrt(mean_squared_error(Y_test, Y_pred_dt)),
        'R2': r2_score(Y_test, Y_pred_dt)
    }
    predictions['Decision Tree Regressor'] = Y_pred_dt
    models['Decision Tree Regressor'] = dt_model

    # Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, Y_train)
    Y_pred_rf = rf_model.predict(X_test)
    results['Random Forest Regressor'] = {
        'MAE': mean_absolute_error(Y_test, Y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(Y_test, Y_pred_rf)),
        'R2': r2_score(Y_test, Y_pred_rf)
    }
    predictions['Random Forest Regressor'] = Y_pred_rf
    models['Random Forest Regressor'] = rf_model

    # Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, Y_train)
    Y_pred_gb = gb_model.predict(X_test)
    results['Gradient Boosting Regressor'] = {
        'MAE': mean_absolute_error(Y_test, Y_pred_gb),
        'RMSE': np.sqrt(mean_squared_error(Y_test, Y_pred_gb)),
        'R2': r2_score(Y_test, Y_pred_gb)
    }
    predictions['Gradient Boosting Regressor'] = Y_pred_gb
    models['Gradient Boosting Regressor'] = gb_model

    return results, predictions, models

def main():
    try:
        # --- Section 1: Data Mentah & Fitur Pencarian ---
        st.header('1. Eksplorasi Data Mentah (Original Dataset)')
        df_raw = load_raw_data()
        
        # Fitur Pencarian Berdasarkan Nama Youtuber
        search_query = st.text_input("Cari Channel YouTuber:", "")
        
        if search_query:
            filtered_df = df_raw[df_raw['Youtuber_Name'].str.contains(search_query, case=False, na=False)]
            st.write(f"Menampilkan hasil pencarian untuk: **\"{search_query}\"** ({len(filtered_df)} baris ditemukan)")
            st.dataframe(filtered_df)
        else:
            st.write("Menampilkan 100 data mentah pertama secara default:")
            st.dataframe(df_raw.head(100))

        # --- Section 2: Pra-pemrosesan Data ---
        st.header('2. Pre Processing Data & Modelling')
        st.write("Memproses data secara otomatis (menghapus duplikat, imputasi nilai hilang, encoding, dll)...")
        
        df_processed = load_and_preprocess_data()
        st.success("Data berhasil diproses dan siap digunakan!")
        
        with st.expander("Lihat Data Hasil Preprocessing (5 Baris Pertama)"):
            st.dataframe(df_processed.head())

        X = df_processed.drop(columns=['Subscribers'])
        Y = df_processed['Subscribers']

        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"Jumlah data latih: **{X_train.shape[0]}** baris")
        with col_info2:
            st.info(f"Jumlah data uji: **{X_test.shape[0]}** baris")

        # --- Section 3: Pelatihan & Evaluasi ---
        st.header('3. Pelatihan & Evaluasi Kinerja Model')
        model_results, model_predictions, trained_models = train_and_evaluate_models(X_train, X_test, Y_train, Y_test)

        st.subheader('Metrik Evaluasi Model')
        results_df = pd.DataFrame(model_results).T
        st.dataframe(results_df.style.format({
            'MAE': '{:,.2f}'.format,
            'RMSE': '{:,.2f}'.format,
            'R2': '{:,.4f}'.format
        }))

        best_model_name = results_df['R2'].idxmax()
        st.success(f"Model terbaik berdasarkan R-squared adalah: **{best_model_name}**")

        # --- Section 4: Visualisasi dengan Warna Berbeda ---
        st.header('4. Visualisasi Perbandingan Prediksi vs Aktual')
        st.write("Setiap model divisualisasikan dengan warna plot yang berbeda untuk keterbacaan yang optimal.")

        # Map warna untuk masing-masing tipe model agar unik
        model_colors = {
            'Linear Regression': '#3498db',         # Blue
            'Decision Tree Regressor': '#2ecc71',   # Green
            'Random Forest Regressor': '#9b59b6',   # Purple
            'Gradient Boosting Regressor': '#e67e22' # Orange
        }

        # Membuat layout visualisasi berkolom
        plot_cols = st.columns(2)
        
        for index, (model_name, y_pred) in enumerate(model_predictions.items()):
            col_target = plot_cols[index % 2]
            with col_target:
                st.subheader(f"{model_name}")
                fig, ax = plt.subplots(figsize=(8, 5))
                
                color = model_colors.get(model_name, '#7f8c8d')
                ax.scatter(Y_test, y_pred, color=color, alpha=0.5, label='Prediksi')
                
                # Menghitung batas garis diagonal
                min_val = min(Y_test.min(), y_pred.min())
                max_val = max(Y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', lw=2, label='Akurat (Y_test=Y_pred)')
                
                ax.set_xlabel("Subscribers Sebenarnya (Y_test)")
                ax.set_ylabel("Hasil Prediksi (Y_pred)")
                ax.set_title(f"{model_name} (R² = {model_results[model_name]['R2']:.4f})")
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                
                st.pyplot(fig)
                plt.close(fig)

        # --- Section 5: Simulasi Prediksi Subscribers Kanal YouTube ---
        st.header('5. Simulasi Prediksi Subscribers')
        st.markdown("""
        Gunakan panel simulasi interaktif di bawah ini untuk memasukkan metrik performa khayalan/target Anda, 
        dan lihat seberapa besar estimasi jumlah *Subscribers* yang bisa Anda dapatkan berdasarkan kecerdasan model terbaik.
        """)

        # Buat form input interaktif untuk beberapa fitur numerik utama
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            input_views = st.number_input("Monthly Views (Penayangan Bulanan)", min_value=0, max_value=50000000, value=5000000, step=10000)
            input_videos = st.number_input("Total Videos (Jumlah Video)", min_value=1, max_value=10000, value=300, step=10)
            input_team = st.slider("Team Members (Jumlah Anggota Tim)", min_value=1, max_value=50, value=5, step=1)
            
        with col_sim2:
            input_spend = st.number_input("Avg Spend Per Video (USD)", min_value=0.0, max_value=50000.0, value=1500.0, step=50.0)
            input_length = st.slider("Avg Video Length (Menit)", min_value=1.0, max_value=180.0, value=12.0, step=0.5)
            input_trending = st.checkbox("Has Trending Video (Memiliki Video Trending)", value=False)

        # Tombol Eksekusi Simulasi
        if st.button("Hitung Estimasi Subscribers"):
            # Mengambil model terbaik dari hasil evaluasi
            best_model = trained_models[best_model_name]
            
            # Siapkan baris input data baru yang strukturnya sama persis dengan X_train
            # Menginisialisasi semua kolom prediktor yang dilatih dengan nilai 0 / False
            input_df = pd.DataFrame(0, index=[0], columns=X_train.columns)
            
            # Isi nilai metrik numerik berdasarkan input pengguna
            input_df['Monthly_Views'] = float(input_views)
            input_df['Total_Videos'] = int(input_videos)
            input_df['Team_Members'] = int(input_team)
            input_df['Avg_Spend_Per_Video_USD'] = float(input_spend)
            input_df['Avg_Video_Length_Min'] = float(input_length)
            input_df['Has_Trending_Video'] = int(input_trending)
            
            # Catatan: Kolom categorical hasil one-hot encoding lainnya sengaja dibiarkan bernilai 0
            # untuk mewakili kondisi dasar/default (misal kategori baseline).
            
            # Lakukan estimasi prediksi menggunakan model terbaik
            predicted_subscribers = best_model.predict(input_df)[0]
            
            # Pastikan hasil prediksi tidak negatif karena jumlah subscribers logisnya selalu positif
            predicted_subscribers = max(0, predicted_subscribers)
            
            # Tampilkan Hasil Simulasi secara Dinamis
            st.success(f"### Estimasi Hasil Prediksi:\nJumlah *Subscribers* yang diproyeksikan adalah **{predicted_subscribers:,.0f}**")
            st.balloons()

    except FileNotFoundError:
        st.error("File 'youtube_channel_subscribers.csv' tidak ditemukan.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

if __name__ == '__main__':
    main()
