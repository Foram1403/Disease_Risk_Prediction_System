from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "what is heart disease",
    "how to reduce risk",
    "what is diabetes",
    "liver disease symptoms"
]

answers = [
    "Heart disease affects heart function.",
    "Exercise and healthy diet reduce risk.",
    "Diabetes is high blood sugar condition.",
    "Liver disease causes fatigue and jaundice."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    return answers[idx]
