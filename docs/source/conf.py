import sys
import os

# Добавляем путь к корневой директории вашего проекта
sys.path.insert(0, os.path.abspath('../../'))

# ИЛИ добавляем путь непосредственно к папке src
sys.path.insert(0, os.path.abspath('../../src'))


project = 'autofin'
copyright = '2025, Denis Tomin'
author = 'Denis Tomin'

release = '0.0.1'  # Полная версия
version = '0.1'    # Короткая версия


# Расширения
extensions = [
    'sphinx.ext.autodoc',       # Для автоматической генерации из docstring
    'sphinx.ext.napoleon',      # Для поддержки Google/Numpy стиля docstring
    'sphinx.ext.viewcode',      # Добавляет ссылки на исходный код
    'sphinx.ext.intersphinx',   # Для ссылок на другую документацию
    'sphinx_design',            # Интерактивные компоненты
    'sphinx_autosummary_accessors',   # Для autosummary
    'sphinx_copybutton',        # Кнопка копирования в кодых блоках
    'sphinxext.opengraph',      # Метатеги Open Graph
]

html_theme = "furo"
html_title = project  # Название в заголовке браузера
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# Дополнительные опции темы (опционально)
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#FF5252",  # Основной цвет для светлой темы
        "color-brand-content": "#FF5252",  # Цвет ссылок и акцентов
    },
    "dark_css_variables": {
        "color-brand-primary": "#FF5252",  # Основной цвет для темной темы
        "color-brand-content": "#FF5252",  # Цвет ссылок и акцентов в темной теме
    },
    "sidebar_hide_name": True,  # Скрыть название в боковой панели (если есть логотип)
}

# Настройка copybutton для исключения промптов Python и IPython
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True


# Включите автоматическую генерацию заглушек
autosummary_generate = True

# Настройка opengraph
ogp_site_url = "https://autofin-docs.readthedocs.io/"  # Пример для Read the Docs
ogp_social_cards = {
    "enable": True,
    "line_color": "#FF5252",
}
# Настройка autodoc
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'

# Добавление кастомного CSS
html_static_path = ['_static']
html_css_files = ['custom.css']

autodoc_mock_imports = ['pandas', 'numpy', 'scikit-learn']

# Базовая конфигурация
ogp_site_name = project  # Название сайта (по умолчанию - project)
# ogp_image = "https://your-documentation-url.org/image.png"  # Дефолтное изображение для превью
ogp_description_length = 200  # Длина описания (символов)
ogp_type = "website"  # Тип контента

# Для генерации социальных карточек (требует matplotlib)
ogp_social_cards = {
    "enable": True,
    "line_color": "#FF5252",  # Цвет акцентов на карточке
    "font": "Noto Sans",  # Шрифт
}