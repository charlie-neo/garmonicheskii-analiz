# app_streamlit.py
# Двухпроцессная версия: Streamlit (UI) + Flask (API) — по диаграмме

import streamlit as st
import requests
import time
from pathlib import Path

from styles import GLOBAL_CSS

API_URL = "http://localhost:5001/api/analyze"
API_HEALTH = "http://localhost:5001/api/health"


# ============================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="Анализатор гармонии",
    page_icon="🎼",
    layout="wide",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ============================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================
with st.sidebar:
    # --- Логотип ---
    try:
    	# Логотип занимает ~50% ширины панели 
    	col_l, col_m, col_r = st.columns([25, 50, 25])
    	with col_m:
        	st.image("static/logo.png", use_container_width=True)
    except Exception:
        st.markdown('<div style="text-align:center;font-size:3rem;">🎼</div>',
                    unsafe_allow_html=True)

    # --- Название вуза ---
    st.markdown("""
    <div class="sidebar-univ">
        ЧОУВО «Московский университет<br>имени С.Ю. Витте»
    </div>
    """, unsafe_allow_html=True)

    # --- Цитата ---
    st.markdown("""
    <div class="quote-block">
        «Если бы только весь мир мог почувствовать силу гармонии».
        <div class="quote-author">— В. А. Моцарт</div>
    </div>
    """, unsafe_allow_html=True)

    # --- О работе ---
    st.markdown("### 📖 О работе")

    st.markdown("""
    <div class="info-block">
        <b>Тема:</b><br>
        Разработка интеллектуальной системы для гармонического анализа
        нотных текстов с использованием методов машинного обучения
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        <b>Автор:</b><br>Ольга Арсенина
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-block">
        <b>Научный руководитель:</b><br>Сергей Зайцев
    </div>
    """, unsafe_allow_html=True)

    # --- Распознаваемые типы ---
    st.markdown("### 🎵 Распознаваемые типы")

    st.markdown("""
    <div class="types-block">
        <span class="type-title">🎼 Хорал</span><br>
        <span class="type-desc">Полифонический склад, аккордовая вертикаль</span>
        <hr>
        <span class="type-title">🎹 Бас + аккорд</span><br>
        <span class="type-desc">Гомофонно-гармонический склад</span>
        <hr>
        <span class="type-title">🎶 Арпеджио</span><br>
        <span class="type-desc">Гомофонно-гармонический склад</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Статус системы ---
    st.markdown("### ⚙️ Статус системы")

    st.markdown("""
    <div class="status-block">
        <b>🟢 Система активна</b><br>
        • Модель: загружена<br>
        • Признаков: 23<br>
        • Типов: 3
    </div>
    """, unsafe_allow_html=True)

    # --- Ссылки ---
    st.markdown("### 🔗 Ссылки")

    st.markdown("""
    <div class="links-block">
        📂 <a href="https://gitflic.ru/project/arseninaoa/garmonicheskii-analiz"
              target="_blank">Исходный код (GitFlic)</a><br>
        📊 <a href="https://disk.yandex.ru/d/OhxZHtNJ88haHw"
              target="_blank">Датасет для обучения</a>
    </div>
    """, unsafe_allow_html=True)

    # --- Музыкальная заставка ---
    st.markdown("""
    <div class="music-block">🎶 🎹 🎼</div>
    """, unsafe_allow_html=True)

    # --- Проверка API ---
    st.markdown("---")
    try:
        r = requests.get(API_HEALTH, timeout=2)
        if r.status_code == 200:
            st.success("🟢 API-сервер доступен")
        else:
            st.warning(f"🟡 API вернул код {r.status_code}")
    except Exception:
        st.error("🔴 API-сервер недоступен. Запустите `python api_server.py`")


# ============================================================
# ШАПКА ОСНОВНОГО ЭКРАНА
# ============================================================
st.markdown("""
<div class="main-header">
    <div class="corner-bl"></div>
    <div class="corner-br"></div>
    <h1>🎼 Анализатор гармонии</h1>
    <p>Загрузите нотный файл — нейросеть<br>
    определит тип и проведёт гармонический анализ</p>
    <div class="badge-ai">🤖 Работает на нейросети</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# ЗАГРУЗКА ФАЙЛА
# ============================================================
st.subheader("📁 Загрузка файла")

uploaded_file = st.file_uploader(
    "Выберите или перетащите файл",
    type=["mxl", "xml", "mid", "midi"],
    help="Поддерживаются .mxl, .xml, .mid, .midi",
    label_visibility="collapsed",
)

file_valid = False
if uploaded_file is not None:
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in {'.mxl', '.xml', '.mid', '.midi'}:
        st.error(f"❌ Неверный формат файла: {ext}. "
                 f"Поддерживаются: .mxl, .xml, .mid, .midi")
    else:
        file_valid = True
        size_kb = len(uploaded_file.getvalue()) / 1024
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.success(f"✅ Файл загружен: **{uploaded_file.name}**")
        col2.metric("Размер", f"{size_kb:.1f} КБ")
        col3.metric("Формат", ext.upper())

st.divider()


# ============================================================
# КНОПКА ЗАПУСКА
# ============================================================
analyze_clicked = st.button(
    "🚀 Запустить анализ",
    type="primary",
    use_container_width=True,
    disabled=not file_valid,
)


# ============================================================
# АНАЛИЗ
# ============================================================
if analyze_clicked and file_valid:
    st.subheader("⏳ Состояние анализа")
    stage_placeholder = st.empty()
    progress = st.progress(0)

    def show_stages(active_idx: int):
        stages = [
            ("1. Извлечение 23 признаков", "🔍"),
            ("2. Классификация нейросетью", "🧠"),
            ("3. Гармонический анализ",     "🎼"),
        ]
        html = ""
        for i, (txt, emoji) in enumerate(stages):
            if i < active_idx:
                cls = "stage-done"; mark = "✅"
            elif i == active_idx:
                cls = "stage-active"; mark = "⏳"
            else:
                cls = ""; mark = "•"
            html += f'<div class="stage-box {cls}">{mark} {emoji} {txt}</div>'
        stage_placeholder.markdown(html, unsafe_allow_html=True)

    try:
        show_stages(0)
        progress.progress(15)
        time.sleep(0.4)

        show_stages(1)
        progress.progress(40)
        time.sleep(0.4)

        files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
        resp = requests.post(API_URL, files=files, timeout=120)

        if resp.status_code != 200:
            try:
                err = resp.json().get('error', 'Неизвестная ошибка')
            except Exception:
                err = resp.text or f'HTTP {resp.status_code}'
            progress.empty()
            stage_placeholder.empty()
            st.error(f"❌ Сообщение об ошибке: {err}")
            st.stop()

        show_stages(2)
        progress.progress(85)
        time.sleep(0.4)

        result = resp.json()
        progress.progress(100)
        show_stages(3)

        # ---- РЕЗУЛЬТАТ ----
        st.divider()
        st.subheader("📊 Результаты анализа")

        predicted = result.get('predicted_type', '?')
        confidence = result.get('confidence', 0)
        emoji = {'хорал': '🎵', 'бас+аккорд': '🎹', 'арпеджио': '🎶'}.get(predicted, '🎼')

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {emoji} Тип: **{predicted}**")
        with col2:
            st.metric("Уверенность НС", f"{confidence}%")

        st.markdown("#### Основные характеристики")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Тональность", result.get('tonality', '—').split('(')[0].strip())
        m2.metric("Размер", result.get('time_signature', '—'))
        m3.metric("Тактов", result.get('total_measures', 0))
        m4.metric("Аккордов", result.get('total_chords', 0))

        st.caption(f"**Тональность полностью:** {result.get('tonality', '—')}")

        st.markdown("#### 🎼 Аккордовая последовательность")
        chords = result.get('chords', '—')
        st.code(chords, language=None)

        # ---- ДЕЙСТВИЯ ----
        st.divider()
        st.markdown("### 🔄 Возможные действия")

        act1, act2 = st.columns(2)
        with act1:
            report_text = (
                f"АНАЛИЗ ГАРМОНИИ\n{'='*50}\n"
                f"Файл:              {uploaded_file.name}\n"
                f"Тип:               {predicted}\n"
                f"Уверенность НС:    {confidence}%\n"
                f"Тональность:       {result.get('tonality', '?')}\n"
                f"Размер:            {result.get('time_signature', '?')}\n"
                f"Тактов:            {result.get('total_measures', 0)}\n"
                f"Аккордов:          {result.get('total_chords', 0)}\n"
                f"\nАККОРДЫ ПО ТАКТАМ:\n{chords}\n"
            )
            st.download_button(
                "📥 Скачать отчёт (TXT)",
                data=report_text,
                file_name=f"{Path(uploaded_file.name).stem}_analysis.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with act2:
            if st.button("🔄 Загрузить новый файл", use_container_width=True):
                st.rerun()

    except requests.exceptions.ConnectionError:
        progress.empty()
        stage_placeholder.empty()
        st.error("❌ Не удалось подключиться к API-серверу. "
                 "Запустите `python api_server.py`.")
    except requests.exceptions.Timeout:
        progress.empty()
        stage_placeholder.empty()
        st.error("❌ Превышено время ожидания ответа от сервера.")
    except Exception as e:
        progress.empty()
        stage_placeholder.empty()
        st.error(f"❌ Непредвиденная ошибка: {e}")