chatbot_data = {
    "heart disease": "Heart disease involves blocked arteries or heart issues.",
    "diabetes": "Diabetes is high blood sugar due to insulin issues.",
    "liver disease": "Liver disease affects detoxification and metabolism.",
    "reduce risk": "Exercise, healthy diet, avoid smoking.",
    "high risk": "Consult doctor immediately.",
    "low risk": "Maintain healthy lifestyle."
}

def chatbot_response(user_input):
    user_input = user_input.lower()
    for key in chatbot_data:
        if key in user_input:
            return chatbot_data[key]
    return "Please consult a healthcare professional."
