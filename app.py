import streamlit as st
import numpy as np
import rasterio
from rasterio.transform import from_origin
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
from io import BytesIO
import folium
from streamlit_folium import st_folium

# ---------- 1. Авто-генерация модели и тестовых снимков ----------
if not os.path.exists("water_ml_model.pkl"):
    st.info("🛠 Первый запуск: обучение модели Random Forest... подождите пару секунд.")
    np.random.seed(42)
    n = 300
    eps = 1e-5
    clean = {
        'B2': np.random.normal(300, 30, n),
        'B3': np.random.normal(400, 40, n),
        'B4': np.random.normal(200, 20, n),
        'B8': np.random.normal(50, 10, n)
    }
    oil = {
        'B2': np.random.normal(400, 40, n),
        'B3': np.random.normal(500, 50, n),
        'B4': np.random.normal(450, 45, n),
        'B8': np.random.normal(300, 30, n)
    }
    algae = {
        'B2': np.random.normal(250, 20, n),
        'B3': np.random.normal(800, 80, n),
        'B4': np.random.normal(300, 30, n),
        'B8': np.random.normal(900, 90, n)
    }
    B2 = np.concatenate([clean['B2'], oil['B2'], algae['B2']])
    B3 = np.concatenate([clean['B3'], oil['B3'], algae['B3']])
    B4 = np.concatenate([clean['B4'], oil['B4'], algae['B4']])
    B8 = np.concatenate([clean['B8'], oil['B8'], algae['B8']])
    NDWI = (B3 - B8) / (B3 + B8 + eps)
    NDVI = (B8 - B4) / (B8 + B4 + eps)
    X = np.stack([B2, B3, B4, B8, NDWI, NDVI], axis=1)
    y = np.concatenate([np.zeros(n), np.ones(n), np.ones(n)*2])
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "water_ml_model.pkl")
    st.success("✅ Модель обучена и сохранена!")

if not os.path.exists("test1.tif"):
    for scene_id in [1, 2]:
        height, width = 200, 200
        water_clean = {
            'B2': np.random.normal(300, 20, (height, width//2)),
            'B3': np.random.normal(400, 20, (height, width//2)),
            'B4': np.random.normal(200, 20, (height, width//2)),
            'B8': np.random.normal(50, 5, (height, width//2)),
            'B10': np.random.normal(800, 50, (height, width//2))
        }
        oil_part = {
            'B2': np.random.normal(400, 30, (height, width//4)),
            'B3': np.random.normal(500, 30, (height, width//4)),
            'B4': np.random.normal(450, 30, (height, width//4)),
            'B8': np.random.normal(300, 20, (height, width//4)),
            'B10': np.random.normal(900, 50, (height, width//4))
        }
        algae_part = {
            'B2': np.random.normal(250, 20, (height, width//4)),
            'B3': np.random.normal(800, 50, (height, width//4)),
            'B4': np.random.normal(300, 20, (height, width//4)),
            'B8': np.random.normal(900, 50, (height, width//4)),
            'B10': np.random.normal(850, 50, (height, width//4))
        }
        if scene_id == 2:
            oil_part['B8'] = np.random.normal(350, 30, (height, width//4))
            algae_part['B3'] = np.random.normal(600, 40, (height, width//4))
        B2_img = np.concatenate([water_clean['B2'], oil_part['B2'], algae_part['B2']], axis=1)
        B3_img = np.concatenate([water_clean['B3'], oil_part['B3'], algae_part['B3']], axis=1)
        B4_img = np.concatenate([water_clean['B4'], oil_part['B4'], algae_part['B4']], axis=1)
        B8_img = np.concatenate([water_clean['B8'], oil_part['B8'], algae_part['B8']], axis=1)
        B10_img = np.concatenate([water_clean['B10'], oil_part['B10'], algae_part['B10']], axis=1)
        transform = from_origin(0, 0, 10, 10)
        fname = f"test{scene_id}.tif"
        with rasterio.open(fname, 'w', driver='GTiff',
                           height=height, width=width,
                           count=5, dtype='float32',
                           crs='EPSG:32643', transform=transform,
                           descriptions=['B02_Blue','B03_Green','B04_Red','B08_NIR','B10_SWIR']) as dst:
            dst.write(B2_img.astype(np.float32), 1)
            dst.write(B3_img.astype(np.float32), 2)
            dst.write(B4_img.astype(np.float32), 3)
            dst.write(B8_img.astype(np.float32), 4)
            dst.write(B10_img.astype(np.float32), 5)
    st.success("✅ Тестовые снимки test1.tif и test2.tif созданы!")

# ---------- 2. Загрузка модели ----------
model = joblib.load("water_ml_model.pkl")

# ---------- 3. Валидация и метрики ----------
np.random.seed(123)
n_test = 200
eps = 1e-5
clean_test = {
    'B2': np.random.normal(300, 30, n_test),
    'B3': np.random.normal(400, 40, n_test),
    'B4': np.random.normal(200, 20, n_test),
    'B8': np.random.normal(50, 10, n_test)
}
oil_test = {
    'B2': np.random.normal(400, 40, n_test),
    'B3': np.random.normal(500, 50, n_test),
    'B4': np.random.normal(450, 45, n_test),
    'B8': np.random.normal(300, 30, n_test)
}
algae_test = {
    'B2': np.random.normal(250, 20, n_test),
    'B3': np.random.normal(800, 80, n_test),
    'B4': np.random.normal(300, 30, n_test),
    'B8': np.random.normal(900, 90, n_test)
}
B2_test = np.concatenate([clean_test['B2'], oil_test['B2'], algae_test['B2']])
B3_test = np.concatenate([clean_test['B3'], oil_test['B3'], algae_test['B3']])
B4_test = np.concatenate([clean_test['B4'], oil_test['B4'], algae_test['B4']])
B8_test = np.concatenate([clean_test['B8'], oil_test['B8'], algae_test['B8']])
NDWI_test = (B3_test - B8_test) / (B3_test + B8_test + eps)
NDVI_test = (B8_test - B4_test) / (B8_test + B4_test + eps)
X_test = np.stack([B2_test, B3_test, B4_test, B8_test, NDWI_test, NDVI_test], axis=1)
y_test = np.concatenate([np.zeros(n_test), np.ones(n_test), np.ones(n_test)*2])
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

# ---------- 4. Интерфейс ----------
st.set_page_config(page_title="AquaScan ML", layout="wide")
st.title("🌊 AquaScan AI — Мультиспектральный ML-анализ водоемов")

# Сайдбар
st.sidebar.success("🤖 Модель Random Forest загружена")
st.sidebar.info("💡 Настройте параметры и выберите снимок.")
st.sidebar.markdown("---")
st.sidebar.metric("🎯 Accuracy на тесте", f"{acc:.1%}")
st.sidebar.metric("🧩 F1 (macro)", f"{f1:.2f}")

st.sidebar.subheader("⚙️ Параметры детекции")
ndwi_threshold = st.sidebar.slider("Порог NDWI для водной маски", 0.0, 0.5, 0.1, 0.05)
cloud_threshold = st.sidebar.slider("Порог облачности (B10)", 1500, 3000, 2000, 100)

st.sidebar.subheader("🗂 Источник снимка")
source = st.sidebar.radio("Выберите источник:", [
    "Загрузить свой .tif",
    "Тестовый снимок 1",
    "Тестовый снимок 2",
    "Загрузить снимок Sentinel-2 (демо)"
])

# Переменные для координат карты (по умолчанию центр Астаны)
map_center = [51.16, 71.45]
map_bounds = None  # будем вычислять для тестовых снимков

uploaded_file = None
if source == "Загрузить свой .tif":
    uploaded_file = st.file_uploader("Загрузите мультиспектральный снимок (.tif)", type=["tif", "tiff"])
elif source.startswith("Тестовый снимок"):
    num = source.split()[-1]
    fname = f"test{num}.tif"
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            uploaded_file = f.read()
        # Для тестовых снимков координаты соответствуют центру Астаны
        map_center = [51.16, 71.45]
        map_bounds = [[51.15, 71.44], [51.17, 71.46]]  # приблизительно
    else:
        st.error(f"Файл {fname} не найден!")
elif source == "Загрузить снимок Sentinel-2 (демо)":
    st.info("🚀 Демо-режим: поиск снимка Sentinel-2 по координатам...")
    coords = st.text_input("Введите координаты (широта, долгота)", "51.16, 71.45")
    if st.button("Получить снимок"):
        st.warning("Заглушка: здесь будет интеграция с Sentinel Hub API. Используется тестовый снимок.")
        with open("test1.tif", "rb") as f:
            uploaded_file = f.read()
        try:
            lat, lon = map(float, coords.split(","))
            map_center = [lat, lon]
            map_bounds = [[lat-0.01, lon-0.01], [lat+0.01, lon+0.01]]
        except:
            pass
    else:
        uploaded_file = None

if uploaded_file is None:
    st.info("👆 Выберите источник снимка слева, чтобы начать анализ.")
    st.stop()

# ---------- 5. Анализ ----------
file_obj = BytesIO(uploaded_file)
with st.spinner("⏳ ИИ анализирует спектральные профили воды..."):
    with rasterio.open(file_obj) as src:
        band_names = [src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}" for i in range(src.count)]
        st.sidebar.markdown("### 📋 Каналы файла:")
        st.sidebar.write(band_names)
        if src.count < 4:
            st.error("Нужно минимум 4 канала!")
            st.stop()
        b2 = src.read(1).astype(float)
        b3 = src.read(2).astype(float)
        b4 = src.read(3).astype(float)
        b8 = src.read(4).astype(float)
        cloud_mask = src.read(5) > cloud_threshold if src.count >= 5 else np.zeros_like(b2, dtype=bool)

    eps = 1e-5
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    water_mask = ndwi > ndwi_threshold

    h, w = b2.shape
    prediction_mask = np.full((h, w), -2, dtype=int)
    confidence_map = np.zeros((h, w))
    if np.any(water_mask):
        X_water = np.stack([b2[water_mask], b3[water_mask], b4[water_mask],
                             b8[water_mask], ndwi[water_mask], ndvi[water_mask]], axis=1)
        prediction_mask[water_mask] = model.predict(X_water)
        probs = model.predict_proba(X_water)
        confidence_map[water_mask] = np.max(probs, axis=1)

    prediction_mask[cloud_mask] = -1

    valid = (prediction_mask >= 0)
    total_water = np.sum(valid)
    oil_pct = (np.sum(prediction_mask == 1) / total_water * 100) if total_water else 0
    algae_pct = (np.sum(prediction_mask == 2) / total_water * 100) if total_water else 0
    mean_conf = (np.mean(confidence_map[water_mask]) * 100) if np.any(water_mask) else 0

# ---------- Метрики на главном экране ----------
st.subheader("📊 Ключевые показатели")
col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown(
    f"<div style='text-align:center;background:#fff3e0;padding:10px;border-radius:10px;'>"
    f"<h4>🚨 Разливы</h4><h2 style='color:#d32f2f;'>{oil_pct:.2f}%</h2></div>",
    unsafe_allow_html=True
)
col2.markdown(
    f"<div style='text-align:center;background:#e8f5e9;padding:10px;border-radius:10px;'>"
    f"<h4>🦠 Цветение</h4><h2 style='color:#2e7d32;'>{algae_pct:.2f}%</h2></div>",
    unsafe_allow_html=True
)
col3.markdown(
    f"<div style='text-align:center;background:#e3f2fd;padding:10px;border-radius:10px;'>"
    f"<h4>🎯 Уверенность</h4><h2 style='color:#1565c0;'>{mean_conf:.1f}%</h2></div>",
    unsafe_allow_html=True
)
col4.markdown(
    f"<div style='text-align:center;background:#f3e5f5;padding:10px;border-radius:10px;'>"
    f"<h4>✅ Accuracy</h4><h2 style='color:#7b1fa2;'>{acc:.1%}</h2></div>",
    unsafe_allow_html=True
)
col5.markdown(
    f"<div style='text-align:center;background:#fff9c4;padding:10px;border-radius:10px;'>"
    f"<h4>🧩 F1-score</h4><h2 style='color:#f57f17;'>{f1:.2f}</h2></div>",
    unsafe_allow_html=True
)

# ---------- Спектральные профили ----------
st.subheader("🔬 Спектральные профили классов")
classes = {0: 'Чистая вода', 1: 'Нефть', 2: 'Водоросли'}
fig_profile, ax_profile = plt.subplots(figsize=(8, 4))
band_labels = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B8 (NIR)']
for cls_id, cls_name in classes.items():
    mask_cls = (prediction_mask == cls_id)
    if np.any(mask_cls):
        means = [np.mean(b2[mask_cls]), np.mean(b3[mask_cls]), np.mean(b4[mask_cls]), np.mean(b8[mask_cls])]
        ax_profile.plot(band_labels, means, marker='o', label=cls_name)
ax_profile.set_ylabel('Отражательная способность')
ax_profile.set_title('Средние спектральные сигнатуры')
ax_profile.legend()
ax_profile.grid(True, alpha=0.3)
st.pyplot(fig_profile)

# ---------- Трёхпанельная визуализация ----------
st.subheader("📊 Результаты пространственного анализа")
fig, ax = plt.subplots(1, 3, figsize=(20, 7))
rgb = np.stack([b4, b3, b2], axis=2)
rgb = np.clip(rgb / 3000, 0, 1)
ax[0].imshow(rgb)
ax[0].set_title("RGB снимок")
ax[0].axis('off')

colors = ['#D2B48C', '#C0C0C0', '#1E90FF', '#FF0000', '#228B22']
cmap = ListedColormap(colors)
ax[1].imshow(prediction_mask, cmap=cmap, vmin=-2, vmax=2)
ax[1].set_title("Спектральная карта загрязнений", fontsize=12, fontweight='bold')
ax[1].set_xlabel("Красный = нефть, Зелёный = водоросли, Синий = чистая вода")
ax[1].axis('off')
patches = [
    mpatches.Patch(color='#D2B48C', label='Суша'),
    mpatches.Patch(color='#C0C0C0', label='Облачность'),
    mpatches.Patch(color='#1E90FF', label='Чистая вода'),
    mpatches.Patch(color='#FF0000', label='Нефть/Хим. стоки'),
    mpatches.Patch(color='#228B22', label='Цветение водорослей')
]
ax[1].legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8, frameon=False)

im = ax[2].imshow(confidence_map, cmap='RdYlGn', vmin=0.5, vmax=1.0)
ax[2].set_title("Уверенность модели")
ax[2].axis('off')
fig.colorbar(im, ax=ax[2], shrink=0.6)
st.pyplot(fig)

# ---------- Карта расположения снимка ----------
st.subheader("🗺 Карта расположения снимка")
try:
    m = folium.Map(location=map_center, zoom_start=13)
    # Добавляем прямоугольник примерных границ снимка, если известны
    if map_bounds:
        folium.Rectangle(
            bounds=map_bounds,
            color='red',
            fill=True,
            fill_opacity=0.2,
            popup='Границы снимка Sentinel-2'
        ).add_to(m)
    folium.Marker(map_center, popup='Центр снимка').add_to(m)
    st_folium(m, width=700, height=400)
except Exception as e:
    st.warning("Не удалось отобразить карту. Установите folium: pip install folium streamlit-folium")

# ---------- Экспорт ----------
st.subheader("📥 Экспорт результатов")
col_down1, col_down2, col_down3 = st.columns(3)
fig.savefig("report.png", dpi=150, bbox_inches='tight')
with open("report.png", "rb") as f:
    col_down1.download_button(
        label="Скачать карту (PNG)",
        data=f,
        file_name="aquascan_report.png",
        mime="image/png"
    )

csv_data = f"Показатель,Значение\nРазливы нефти (%),{oil_pct:.2f}\nЦветение водорослей (%),{algae_pct:.2f}\nСредняя уверенность (%),{mean_conf:.1f}\nAccuracy модели,{acc:.1%}\nF1-score,{f1:.2f}"
col_down2.download_button(
    label="Скачать метрики (CSV)",
    data=csv_data,
    file_name="aquascan_metrics.csv",
    mime="text/csv"
)

with PdfPages("report.pdf") as pdf:
    pdf.savefig(fig, bbox_inches='tight')
with open("report.pdf", "rb") as f:
    col_down3.download_button(
        label="Скачать отчёт (PDF)",
        data=f,
        file_name="aquascan_report.pdf",
        mime="application/pdf"
    )