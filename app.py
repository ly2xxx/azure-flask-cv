from flask import Flask, render_template, request
import os
import pinecone
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

def get_query_results(question):
    # Set Pinecone API and environment
    pinecone_api = os.environ["PINECONE_API"]
    pinecone_env = os.environ["PINECONE_ENV"]

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)

    # Set Pinecone index
    index_name = "pdfsearchdx"
    index = pinecone.Index(index_name)

    # Set Hugging Face Hub API token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = str(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    # Set query questions
    query_questions = [question, ]

    # Initialize Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode query questions
    query_vectors = [model.encode(str(question)).tolist() for question in query_questions]

    # Perform Pinecone query
    query_results = index.query(vector=query_vectors, top_k=2)

    return query_results

@app.route('/', methods=['GET', 'POST'])
def home():
    submitted_question = "What is my name?"
    query_results = "No results"

    if request.method == 'POST':
        submitted_question = request.form['question']

        # Check if the submitted question is not empty
        if submitted_question:
            query_results = get_query_results(submitted_question)

    return render_template('index.html', query_results=str(query_results), submitted_question=submitted_question)


if __name__ == '__main__':
    app.run(debug=True)
