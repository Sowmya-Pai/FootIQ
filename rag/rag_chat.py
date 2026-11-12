
import os
import sys
import json
import chromadb

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
from match_extractor import extract_match_details

# ---------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")
DATABASE_DIR = os.path.join(BASE_DIR, "Database")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, DATASETS_DIR)
sys.path.insert(0, DATABASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

from sql_question_router import (
    process_sql_question,
    identify_question_type
)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found!")

# Import ML explainability function
from explain_prediction import explain_match


# ---------------------------------------
# 2. CONNECT TO GEMINI
# ---------------------------------------

client = genai.Client(api_key=API_KEY)

GENERATION_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"


# ---------------------------------------
# 3. CONNECT TO CHROMADB
# ---------------------------------------

DB_PATH = os.path.join(BASE_DIR, "rag", "vector_store")

chroma_client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = chroma_client.get_collection(
    name="footiq_knowledge"
)


# ---------------------------------------
# 4. IDENTIFY PREDICTION QUESTIONS
# ---------------------------------------

def is_prediction_question(question):

    question = question.lower()

    keywords = [
        "prediction",
        "predict",
        "predicted",
        "why did footiq",
        "why did the model",
        "model prediction"
    ]

    return any(
        keyword in question
        for keyword in keywords
    )

# ---------------------------------------
# 5. IDENTIFY SQL ANALYTICS QUESTIONS
# ---------------------------------------


def is_sql_question(question):

    # Prediction questions should not trigger SQL automatically
    if is_prediction_question(question):
        return False

    question_type = identify_question_type(question)

    return question_type != "unknown"



# ---------------------------------------
# 6. GET SQL ANALYTICS
# ---------------------------------------

def get_sql_data(question):

    result = process_sql_question(question)

    return result
# ---------------------------------------
# 5. GET ML PREDICTION AND EXPLANATION
# ---------------------------------------

def get_ml_explanation(question):

    # Extract match details from user question
    match_details = extract_match_details(question)

    home = match_details["home_team"]
    away = match_details["away_team"]
    match_date = match_details["match_date"]

    # Require a date for now
    if not match_date:
        raise ValueError(
            "Please include a match date in your question."
        )

    # Generate ML prediction and explanation
    explanation = explain_match(
        home,
        away,
        match_date
    )

    return explanation


# ---------------------------------------
# 6. RETRIEVE RELEVANT DOCUMENTS
# ---------------------------------------

def retrieve_documents(question):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    query_embedding = response.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_documents = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        retrieved_documents.append({
            "topic": metadata["topic"],
            "content": document,
            "distance": distance
        })

    return retrieved_documents


# ---------------------------------------
# 7. GENERATE ANSWER USING GEMINI
# ---------------------------------------

def generate_answer(
    question,
    retrieved_documents,
    ml_data=None,
    sql_data=None
):

    # Prepare football knowledge context
    context = ""

    for document in retrieved_documents:

        context += f"""
        Topic: {document['topic']}
        Information: {document['content']}
        """

    # Prepare ML prediction context
    ml_context = ""

    if ml_data is not None:

        ml_context = json.dumps(
            ml_data,
            indent=4
        )

    # Prepare SQL analytics context
    sql_context = ""

    if sql_data is not None:

        sql_context = json.dumps(
            sql_data,
            indent=4
        )


    prompt = f"""
    You are FootIQ, an intelligent football analytics assistant.

    Answer the user's question clearly and accurately.

    RETRIEVED FOOTBALL KNOWLEDGE:

    {context}

    MACHINE LEARNING PREDICTION DATA:

    {ml_context}

        HISTORICAL SQL ANALYTICS DATA:

    {sql_context}

     
    RULES:
    
    1. Use retrieved football knowledge when relevant.
    
    2. Treat the ML prediction and probabilities as the
   actual output of the prediction model.
   
   3. Every feature contains:
   - feature: technical feature name
   - meaning: definition of that feature
   - value: actual calculated feature value
   - contribution: influence on the class score
   
   4. NEVER confuse feature values with contributions.
   
   5. Positive contributions increase the corresponding
   class score.
   
   6. Negative contributions decrease the corresponding
   class score.
   
   7. Contributions are NOT probability percentages.
   
   8. Explain features using their provided meanings.
   Do not independently guess what they represent.
   
   9. Do not describe statistical associations as proof
   of football causation.
   
   10. Do not invent match statistics or events.
   
   11. Clearly distinguish general football knowledge
    from the ML model's findings.
    
    12. If information is unavailable, acknowledge it.
    
    13. When explaining a prediction, show the numerical
    feature value AND its contribution separately.
    
    14. Use this format:

    Feature name:
    Meaning:
    Feature value:
    Class score contribution:
    
    15. Do not include a separate "Knowledge Sources" section
    in your answer. The website displays retrieved sources
    automatically.
    
    16. Do not output horizontal rules or Markdown separators.
    
    17. Use only ## and ### for headings. Do not use ####.
    
    18. Format answers consistently using headings, bullet
    points, and bold text where appropriate.

    19. Explain all five positive and all five negative
    factors for the predicted outcome whenever available.
    
    20. Do not omit negative factors or replace them with
    vague summaries.
    
    21. Never describe historical match count as model
    confidence. It is simply a historical data count.
    
    22. Do not claim that one factor alone caused the
    predicted outcome.
    
    23. Explain that the highest predicted probability
    determines the predicted class, even when the
    probabilities are very close.

    24. Use SQL analytics data as the authoritative source
    for historical team statistics and match records.

    25. Never invent or modify numerical values retrieved
    from SQL.

    26. If SQL data contains an error, clearly explain
    that the requested information is unavailable.

    27. Do not confuse historical statistics with ML
    predictions.

    28. The dataset contains historical English Premier
    League matches and ends on 10 April 2022.
    Do not present these statistics as current-season data.
    
    29. Do not begin every answer with "Hello", "Hello! I am FootIQ",
    or a generic introduction.
    
    30. Answer the user's question directly from the first sentence.
    
    31. Use headings and bullet points when presenting statistics.
    
    32. Keep answers focused and avoid unnecessary closing statements.

    33. When explaining feature contributions, clarify that they
    represent contributions to the model's class decision score,
    not direct changes in predicted probability. Do not interpret
    a positive contribution as proof that a team will win.




    USER QUESTION:

    {question}

    Write a helpful and easy-to-understand answer:
    """

    try:

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1500
            )
        )

    except errors.APIError as e:

        if e.code == 503:

            print("\nPrimary Gemini model is busy.")
            print("Trying fallback model...")

            response = client.models.generate_content(
                model=FALLBACK_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=900
                )
            )

        else:
            raise

    return response.text



# ---------------------------------------
# 8. PROCESS QUESTION FOR FLASK WEBSITE
# ---------------------------------------


def process_question(question):

    # Retrieve relevant football knowledge
    retrieved_documents = retrieve_documents(question)

    ml_data = None
    sql_data = None

    # Check ML and SQL independently
    if is_prediction_question(question):
        ml_data = get_ml_explanation(question)

    if is_sql_question(question) or is_prediction_question(question):
        sql_data = get_sql_data(question)

    # Generate the final AI response
    answer = generate_answer(
        question,
        retrieved_documents,
        ml_data,
        sql_data
    )

    result = {
        "answer": answer,
        "sources": [
            {
                "topic": document["topic"],
                "distance": round(float(document["distance"]), 4)
            }
            for document in retrieved_documents
        ]
    }

    if ml_data is not None:
        result["prediction"] = {
            "home_team": ml_data["home_team"],
            "away_team": ml_data["away_team"],
            "match_date": ml_data["match_date"],
            "predicted_outcome": ml_data["predicted_outcome"],
            "probabilities": ml_data["probabilities"]
        }

    if sql_data is not None:
        result["sql_analytics"] = sql_data

    return result



# ---------------------------------------
# 8. MAIN CHAT FUNCTION
# ---------------------------------------


def chat():

    print("\n================================")
    print("       FOOTIQ AI ASSISTANT")
    print("================================")

    question = input("\nYou: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nSearching football knowledge...")

    retrieved_documents = retrieve_documents(question)

    # ---------------------------------------
    # IDENTIFY QUESTION TYPE
    # ---------------------------------------

    ml_data = None
    sql_data = None

    prediction_question = is_prediction_question(question)

    # ---------------------------------------
    # RETRIEVE ML PREDICTION
    # ---------------------------------------

    if prediction_question:

        print("\nRetrieving ML prediction and explanation...")

        ml_data = get_ml_explanation(question)

        print("\nML prediction retrieved successfully!")

        print(
            "Predicted outcome:",
            ml_data["predicted_outcome"]
        )

    # ---------------------------------------
    # RETRIEVE SQL ANALYTICS
    # ---------------------------------------

    elif is_sql_question(question):

        print("\nRetrieving historical SQL analytics...")

        sql_data = get_sql_data(question)

        if "error" in sql_data:

            print("SQL information:", sql_data["error"])

        else:

            print("\nSQL analytics retrieved successfully!")

            print("Analytics type:", sql_data["type"])

    # ---------------------------------------
    # GENERATE GEMINI RESPONSE
    # ---------------------------------------

    print("\nGenerating AI response...")

    answer = generate_answer(
        question,
        retrieved_documents,
        ml_data,
        sql_data
    )

    print("\n========== FOOTIQ RESPONSE ==========")

    print("\nFootIQ:", answer)

    # ---------------------------------------
    # DISPLAY ML RESULTS
    # ---------------------------------------

    if ml_data is not None:

        print("\n========== ML PREDICTION ==========")

        print(
            "Match:",
            ml_data["home_team"],
            "vs",
            ml_data["away_team"]
        )

        print(
            "Predicted outcome:",
            ml_data["predicted_outcome"]
        )

        print("\nProbabilities:")

        for outcome, probability in ml_data["probabilities"].items():

            print(f"{outcome}: {probability:.2f}%")

    # ---------------------------------------
    # DISPLAY RETRIEVED SOURCES
    # ---------------------------------------

    print("\n========== SOURCES RETRIEVED ==========")

    for document in retrieved_documents:

        print("\nTopic:", document["topic"])

        print(
            "Distance:",
            round(document["distance"], 4)
        )

    print("\n=======================================")



# ---------------------------------------
# 9. RUN APPLICATION
# ---------------------------------------

if __name__ == "__main__":
    chat()
