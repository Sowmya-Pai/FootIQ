# ⚽ FootIQ – AI-Powered Football Analytics Assistant

FootIQ is an AI-powered football analytics application that combines Machine Learning, SQL analytics and Retrieval-Augmented Generation (RAG) to help users explore football statistics, predict match outcomes and understand football concepts through a conversational assistant.

The application provides historical match analysis, team comparisons, head-to-head statistics, match predictions and AI-generated explanations through a single interface.

## 🚀 Key Features

### 1. Machine Learning Match Prediction
- Predicts football match outcomes: Home Win, Draw or Away Win.
- Uses historical English Premier League match data.
- Applies feature engineering to identify useful match patterns.
- Provides probability scores for each predicted outcome.
- Uses the trained FootIQ ML model (v5).

### 2. SQL-Based Football Analytics
- Retrieves historical team statistics using SQLite.
- Calculates wins, losses, draws and average goals scored and conceded.
- Supports team-to-team comparisons.
- Provides head-to-head analysis.
- Uses a question router to identify the type of analytical query and retrieve relevant results.

### 3. AI Football Knowledge Assistant
- Uses Retrieval-Augmented Generation (RAG).
- Retrieves relevant information from a football knowledge base.
- Explains football concepts, tactics and terminology.
- Uses ChromaDB for vector-based knowledge retrieval.
- Uses Google's Gemini API to generate conversational responses.

### 4. Prediction Explanations
- Connects match predictions with historical statistics.
- Retrieves prediction results and relevant information.
- Provides understandable explanations of model predictions.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Machine Learning | Scikit-learn, Pandas, NumPy |
| Database | SQLite |
| RAG | ChromaDB |
| Generative AI | Google Gemini API |
| Model Storage | Joblib, Pickle |
| Version Control | Git, GitHub |

## 🏗️ Project Architecture

```text
                    FOOTIQ
                       |
               Flask Web Application
                       |
          +------------+------------+
          |            |            |
      ML Engine    SQL Analytics   RAG Engine
          |            |            |
      Match         SQLite       ChromaDB
     Prediction    Statistics   Knowledge Base
          |            |            |
          +------------+------------+
                       |
                Gemini API
                       |
              AI-Generated Responses
                       |
                 Web Interface
```

## 📂 Project Structure

```text
FootIQ/
│
├── Database/
│   ├── build_database.py
│   ├── footiq.db
│   ├── sql_queries.py
│   └── sql_question_router.py
│
├── Datasets/
│   ├── advanced_features.py
│   ├── feature_engineering.py
│   ├── compare_models_v5.py
│   ├── evaluate_model_v5.py
│   ├── predict_model.py
│   ├── explain_prediction.py
│   ├── footIQ_best_model_v5.pkl
│   └── Historical match datasets
│
├── rag/
│   ├── build_knowledge_base.py
│   ├── match_extractor.py
│   ├── rag_chat.py
│   └── search_knowledge.py
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html
│
├── app.py
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sowmya-Pai/FootIQ.git
cd FootIQ
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

Install the Python packages required by the project.

```bash
pip install flask pandas numpy scikit-learn joblib chromadb google-genai python-dotenv
```

If additional packages are required by your local environment, install those as well.

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key.

**Important:** Never upload your `.env` file or expose your API key publicly.

### 5. Run the application

```bash
python app.py
```

Open the local URL displayed in the terminal in your browser.

## 📊 Dataset Information

FootIQ uses historical English Premier League match data for model training and analysis.

- The current dataset contains historical matches through **10 April 2022**.
- Predictions are based on the trained model and available historical features.
- The application does not currently use live match feeds or real-time season data.

## 🔮 Future Enhancements

- Integration of live football match data and APIs.
- Updated team and player performance statistics.
- Advanced player-level performance analytics.
- More detailed model explainability.
- Interactive football dashboards and visualizations.
- Expansion to additional football leagues.

## 👩‍💻 Developer

**Sowmya U Pai**

- GitHub: [Sowmya-Pai](https://github.com/Sowmya-Pai)

---

*FootIQ – Bringing Machine Learning, Football Statistics and AI Together.*