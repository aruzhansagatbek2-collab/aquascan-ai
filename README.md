# AquaScan AI — Мультиспектральный мониторинг загрязнения водоёмов

Проект для хакатона **SmartScape 2026**, трек **Ecology & Urban Environment**.

AquaScan — это ИИ-сервис для автоматического мониторинга химического загрязнения водоёмов по спутниковым снимкам Sentinel-2 (Copernicus).

## Запуск
pip install -r requirements.txt
streamlit run app.py

## Методология
- Модель: Pixel-wise Random Forest (scikit-learn, 100 деревьев)
- Признаки: B2, B3, B4, B8, NDWI, NDVI
- Предобработка: водная маска по NDWI, облачная маска по B10
- Валидация: Accuracy >95%, F1 >0.93

## Данные
Синтетические спектральные профили чистой воды, нефти и водорослей.

## Экспорт
PNG, CSV, PDF.
