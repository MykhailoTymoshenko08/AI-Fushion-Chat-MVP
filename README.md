# AI Fusion Chat (Multi-LLM Judge Aggregator)

**AI Fusion Chat** — це потужна платформа для агрегації відповідей від кількох LLM з використанням патерну "LLM-as-a-Judge". Система не лише опитує моделі паралельно, але й має можливості RAG (Retrieval-Augmented Generation) для роботи з PDF-файлами та доступ до інтернету через Tavily Search API для отримання актуальної інформації.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)

## Основні можливості

* **Multi-LLM Architecture**: Паралельні асинхронні запити до `Llama-3.3` та `Gemma-3` через OpenRouter.
* **AI Judge System**: Окрема модель (`Mistral Devstral`) аналізує відповіді та генерує фінальний, синтезований висновок.
* **RAG (PDF Analysis)**: Завантаження PDF-документів, їх векторизація (ChromaDB) та використання як контексту для відповідей.
* **Web Search**: Інтеграція з Tavily API для пошуку актуальної інформації в інтернеті.
* **Smart Context Routing**: Система автоматично обирає джерело даних (PDF, Web або загальні знання) залежно від запиту.
* **History & Analytics**: Збереження історії запитів у SQLite та перегляд статистики продуктивності моделей.

## Структура проєкту

```text
my_ai_project/
├── backend/
│   ├── server.py           # Основний файл FastAPI бекенду
│   └── my_aggregator.db    # SQLite база даних (історія)
├── frontend/
│   └── ui.py               # Streamlit інтерфейс
├── db_storage/             # Папка для векторної бази даних (ChromaDB)
├── .env                    # Змінні середовища (API ключі)
├── requirements.txt        # Залежності
├── helper.txt              # Допоміжні нотатки
└── README.md               # Документація
```

## Технологічний стек

-**Python 3.10+**
-**LangChain**
-**ChromaDB**
-**Tavily**
-**FastAPI**
-**Streamlit**
-**SQLite3**
-**OpenRouter API**
-**asyncio**
-**langchain-openai**
-**langchain-core**
-**langchain_huggingface**
-**langchain_community**
-**langchain_text_splitters**
-**pydantic**
-**shutil**
-**dotenv**
-**Uvicorn**
-**Json**
-**Typing**
-**datetime**

## Встановлення та запуск

### 1. Клонування репозиторію

```bash 
git clone <your-repo-url>
cd my_ai_project
```

### 2. Створення віртуального середовища

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Встановлення залежностей
```bash
pip install -r requirements.txt
```

### 4. Налаштування .env
Створіть файл .env у кореневій папці та додайте свої ключі:
```bash
OPENROUTER_API_KEY=sk-or-v1-......
TAVILY_API_KEY=tvly-......
```

## Використання

Для роботи системи потрібно запустити Backend та Frontend у двох окремих терміналах.

### Термінал 1: Запуск Backend (FastAPI)
Перебуваючи у кореневій папці проєкту:
```bash
cd backend
uvicorn server:app --reload
```
API буде доступне за адресою: http://localhost:8000 
Документація Swagger UI: http://localhost:8000/docs

### Термінал 2: Запуск Frontend (Streamlit)
Відкрийте новий термінал, активуйте venv та запустіть:
```bash
streamlit run frontend/ui.py
```
Інтерфейс відкриється автоматично у браузері за адресою: http://localhost:8501

## API Endpoints
**Метод   Ендпоінт	Опис**
POST    /ask	    Основний запит. Опитує 2 моделі + Judge, використовує RAG/Web контекст.
POST    /upload	    Завантаження PDF файлу для наповнення бази знань (RAG).
GET     /history	Отримання історії запитів.
GET     /stats	    Статистика швидкості та кількості запитів до моделей.
DELETE  /clear	    Очищення історії запитів.

Приклад запиту /ask
```json
{
  "question": "Як працює квантовий комп'ютер?",
  "model1": "meta-llama/llama-3.3-70b-instruct:free",
  "model2": "google/gemma-3-27b-it:free",
  "use_judge": true
}
```
Приклад відповіді
```json
{
  "status": "success",
  "final_answer": "Квантовий комп'ютер використовує кубіти...",
  "model1": { "name": "...", "duration": "3.2s" },
  "model2": { "name": "...", "duration": "2.8s" },
  "total_duration": "7.5s"
}
```

## Архітектура RAG + Judge

### Input: Користувач задає питання.
### Context:
-Пошук схожих чанків у ChromaDB (з PDF).
-Пошук в інтернеті через Tavily.
### Aggregation: Питання + Контекст надсилаються паралельно двом моделям.
### Judging: Відповіді моделей передаються третій моделі (Judge), яка формує фінальний результат.
### Output: Користувач отримує найкращу можливу відповідь.
