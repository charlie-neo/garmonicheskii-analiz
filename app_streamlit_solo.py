# app_streamlit_solo.py
# Однопроцессная версия: Streamlit + анализатор в одном файле.
# Без Flask. Используется для деплоя на Streamlit Cloud.

import streamlit as st
import tempfile
import uuid
import time
from pathlib import Path

from analyzer import MusicAnalyzer
from styles import GLOBAL_CSS


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
# ЗАГРУЗКА МОДЕЛИ (один раз, кэшируется)
# ============================================================
@st.cache_resource(show_spinner="📦 Загружаю нейросеть...")
def load_analyzer():
    return MusicAnalyzer(models_dir="dataset/models")


# ============================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================
with st.sidebar:
    # --- Логотип ---
    try:
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

    st.markdown("---")
    st.success("🟢 Система готова к работе")


# ============================================================
# ЗАГРУЗКА АНАЛИЗАТОРА
# ============================================================
try:
    analyzer = load_analyzer()
except Exception as e:
    st.error(f"❌ Не удалось загрузить модель: {e}")
    st.stop()


# ============================================================
# ШАПКА ОСНОВНОГО ЭКРАНА
# ============================================================
st.markdown("""
<div class="main-header">
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
        time.sleep(0.3)

        show_stages(1)
        progress.progress(45)
        time.sleep(0.3)

        # Сохраняем во временный файл
        tmp_dir = Path(tempfile.gettempdir()) / "music_analyzer"
        tmp_dir.mkdir(exist_ok=True)
        ext = Path(uploaded_file.name).suffix.lower()
        tmp_path = tmp_dir / f"{uuid.uuid4().hex}{ext}"
        tmp_path.write_bytes(uploaded_file.getvalue())

        # Анализ
        result = analyzer.analyze(str(tmp_path))

        # Удаляем временный файл
        if tmp_path.exists():
            tmp_path.unlink()

        show_stages(2)
        progress.progress(85)
        time.sleep(0.3)

        if result.get('status') == 'error':
            progress.empty()
            stage_placeholder.empty()
            st.error(f"❌ Ошибка анализа: "
                     f"{result.get('error', 'Неизвестная ошибка')}")
            st.stop()

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

    except Exception as e:
        progress.empty()
        stage_placeholder.empty()
        st.error(f"❌ Непредвиденная ошибка: {e}")