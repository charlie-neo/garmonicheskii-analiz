# styles.py
# Единые стили для сайта анализатора гармонии
# Палитра: белый фон, красный акцент, светло-сиреневые плашки,
#          фиолетово-синий для линий, тёмно-серый текст


GLOBAL_CSS = """
<style>
    /* ============ ОСНОВА ============ */
    .stApp {
        background: #ffffff;
    }

    /* ============ ШАПКА ОСНОВНОГО ЭКРАНА ============ */
     .main-header {
        background: #e9e4f4;
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 24px;
    }

    .main-header h1 {
        color: #2b2b2b;
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }

    .main-header p {
        color: #6c757d;
        margin: 12px 0 0 0;
        font-size: 1rem;
        line-height: 1.5;
    }

    .badge-ai {
        display: inline-block;
        background: #ffffff;
        color: #d62828;
        border: 1px solid #d62828;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-top: 12px;
        font-weight: 600;
    }

    /* ============ БОКОВАЯ ПАНЕЛЬ ============ */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e9e4f4;
    }

    .sidebar-logo {
        text-align: center;
        padding: 12px 0 16px 0;
        border-bottom: 1px solid #667eea;
        margin-bottom: 16px;
    }

    .sidebar-univ {
        text-align: center;
        font-weight: 700;
        font-size: 0.95rem;
        color: #2b2b2b;
        line-height: 1.3;
        padding: 8px 0 12px 0;
    }

    .quote-block {
        background: #e9e4f4;
        padding: 14px 16px;
        border-radius: 10px;
        margin: 16px 0;
        font-style: italic;
        color: #2b2b2b;
        line-height: 1.6;
        font-size: 0.88rem;
    }

    .quote-author {
        text-align: right;
        margin-top: 8px;
        font-style: normal;
        font-weight: 600;
        color: #6c757d;
        font-size: 0.82rem;
    }

    .info-block {
        background: #e9e4f4;
        padding: 12px 14px;
        border-radius: 10px;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #2b2b2b;
        margin-top: 8px;
    }

    .info-block b {
        color: #2b2b2b;
    }

    .types-block {
        background: #e9e4f4;
        padding: 12px 14px;
        border-radius: 10px;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #2b2b2b;
    }

    .types-block .type-title {
        font-weight: 700;
        color: #2b2b2b;
    }

    .types-block .type-desc {
        font-size: 0.8rem;
        color: #6c757d;
    }

    .types-block hr {
        margin: 8px 0;
        border: none;
        border-top: 1px solid #667eea;
        opacity: 0.3;
    }

    .status-block {
        background: #e9e4f4;
        border-left: 3px solid #667eea;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #2b2b2b;
        line-height: 1.7;
    }

    .status-block b {
        color: #2b2b2b;
    }

    .links-block {
        font-size: 0.85rem;
        line-height: 1.9;
    }

    .links-block a {
        color: #d62828;
        text-decoration: none;
    }

    .links-block a:hover {
        text-decoration: underline;
    }

    .music-block {
        text-align: center;
        font-size: 2.2rem;
        padding: 12px 0;
        background: #e9e4f4;
        border-radius: 10px;
        margin-top: 12px;
    }

    /* ============ ЭТАПЫ АНАЛИЗА ============ */
    .stage-box {
        background: #e9e4f4;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 6px 0;
        font-family: monospace;
        color: #2b2b2b;
    }

    .stage-done {
        border-left-color: #667eea;
        color: #2b2b2b;
    }

    .stage-active {
        border-left-color: #d62828;
        color: #2b2b2b;
        background: #ffffff;
        border: 1px solid #d62828;
        border-left: 4px solid #d62828;
    }

    /* ============ КНОПКИ ============ */
    .stButton > button[kind="primary"] {
        background: #667eea;
        border: none;
        color: #ffffff;
        font-weight: 600;
    }

    .stButton > button[kind="primary"]:hover {
        background: #5568d3;
        color: #ffffff;
    }

    .stDownloadButton > button {
        background: #e9e4f4;
        color: #2b2b2b;
        border: 1px solid #667eea;
    }

    .stDownloadButton > button:hover {
        background: #dcd4ef;
        color: #2b2b2b;
    }

    /* ============ МЕТРИКИ ============ */
    div[data-testid="stMetric"] {
        background: #e9e4f4;
        padding: 12px 16px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        color: #6c757d;
    }

    div[data-testid="stMetricValue"] {
        color: #2b2b2b;
    }
</style>
"""