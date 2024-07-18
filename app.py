import os
import sqlite3
from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
from scipy import spatial
from dotenv import load_dotenv
import pandas as pd
import ast
from flask_cors import CORS
import logging
import uuid

app = Flask(__name__)
CORS(app, supports_credentials=True)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app.config['SECRET_KEY'] = os.getenv("APP_KEY")

API_KEY = os.getenv("API_KEY")

# dependencies
client = OpenAI(api_key=API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
FILE_PATH = "embeddings.csv"

# reading the csv
df = pd.read_csv(FILE_PATH)
df['Embedding'] = df['Embedding'].apply(ast.literal_eval)

# Database setup
DATABASE = 'sessions.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with app.app_context():
        db = get_db()
        db.execute("CREATE TABLE IF NOT EXISTS session (sessionID TEXT PRIMARY KEY, sessionTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, info TEXT)")
        db.commit()

def user_input_embedding(query: str):
    user_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_vector = user_embedding_response.data[0].embedding
    return query_vector

def strings_ranked_by_relatedness(
    query_vector,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    strings_and_relatednesses = [
        (row["Section"], relatedness_fn(query_vector, row["Embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def handle_response(query):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {'role': 'system', 'content': 'You answer questions from students and parents at Palm Beach State College. Your answers should be short (2-3 sentences) and easy to understand. Use a friendly tone of voice. Do not hallucinate! If there is a link in provided information - attach this link to the last word of the sentence before the link in HTML format like <a href="https://www.homepage.com">Visit our homepage</a>.'},
            {'role': 'user', 'content': query},
        ]
    )
    response_message = response.choices[0].message
    return response_message.content

@app.route('/', methods=['GET'])
def index():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id  # Store the session ID in the session
    
    # Insert new session record into the database
    db = get_db()
    db.execute("INSERT INTO session (sessionID, info) VALUES (?, ?)", (session_id, ""))
    db.commit()
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        session_id = session.get('session_id')  # Retrieve session_id from the session

        if not session_id:
            return jsonify({"error": "Session ID not found"}), 400

        db = get_db()

        # Perform embedding search first
        query_for_embedding = f"Answer the following question: {user_message}"
        strings, relatednesses = strings_ranked_by_relatedness(user_input_embedding(query_for_embedding), df, top_n=1)

        # Update session info
        new_info = strings[0]
        
        # Retrieve existing info
        cursor = db.execute("SELECT info FROM session WHERE sessionID = ?", (session_id,))
        row = cursor.fetchone()
        current_info = row['info'] if row else ""

        updated_info = new_info + "\n" + current_info

        # Update session info in the database
        db.execute("UPDATE session SET info = ? WHERE sessionID = ?", (updated_info, session_id))
        db.commit()

        # Construct query for OpenAI API
        query = f"""Use only the information below to answer the following question. If the answer cannot be found, write "Oops, seems like I don't know the answer to this question! Please, visit https://www.palmbeachstate.edu/"
        information:
        \"\"\"
        {updated_info}
        \"\"\"
        Question: {user_message}"""

        # Get response from OpenAI
        response_text = handle_response(query)

        message = {"response": response_text}
        return jsonify(message)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An internal server error occurred"}), 500

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.run(debug=True)
