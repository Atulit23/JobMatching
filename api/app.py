from flask import Flask, jsonify, request
import joblib
import spacy
from spacy.lang.en import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

vectorizer = TfidfVectorizer()

data = joblib.load('job_data.joblib')
job_vectors = vectorizer.fit_transform(data)

def preprocess_text(text):
    tokens = [token.lower() for token in text if token not in STOP_WORDS]
    return " ".join(tokens)

@app.route('/', methods=['POST', 'GET'])
def index():
    data_to_return = []

    input_data = request.json
    input_ = input_data.get("query")

    user_input = preprocess_text(input_)
    candidate_vectors_new = vectorizer.transform([user_input])

    similarity_scores_new = cosine_similarity(job_vectors, candidate_vectors_new)

    dist_final = sorted(list(enumerate(similarity_scores_new)), reverse=True, key = lambda x : x[1])

    for i in range(len(data)):
        for j in dist_final[0:5]:
            if(i == j[0]):
                data_to_return.append({
                    'title': data['Job Title'][i],
                    'description': data['Job Description'][i],
                    'salary': data['Salary Range'][i],
                    'location': data['location'][i],
                    'Country': data['Country'][i],
                    'type': data['Work Type'][i],
                    'role': data['Role'][i],
                    'benefits': data['Benefits'][i],
                    'skills': data['skills'][i],
                    'qualifications': data['Qualifications'][i],
                    'expirence': data['Experience'][i],
                })
    return jsonify(data_to_return)


if __name__ == "__main__":
    app.run(debug=True)