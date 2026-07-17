import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Expanded Dataset (Intents, Patterns, and Responses)
intents_data = [
    {
        "tag": "greeting",
        "patterns": ["hi", "hello", "hey", "is anyone there", "good day", "greetings"],
        "responses": ["Hello! How can I help you today?", "Hi there! What can I do for you?"]
    },
    {
        "tag": "shipping",
        "patterns": ["where is my order", "track my package", "delivery status", "when will it arrive", "shipping updates"],
        "responses": ["You can track your order using the link sent to your email, or check your account dashboard."]
    },
    {
        "tag": "returns",
        "patterns": ["i want a refund", "how to return an item", "can i get my money back", "return policy", "damaged item"],
        "responses": ["We offer a 30-day money-back guarantee. Would you like me to generate a return shipping label?"]
    },
    {
        "tag": "human_agent",
        "patterns": ["talk to a human", "speak with a person", "customer service representative", "call center", "help desk"],
        "responses": ["I am transferring you to a live support agent right now. Please hold on briefly."]
    }
]

# 2. Prepare Data for Training
training_sentences = []
training_labels = []

for intent in intents_data:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern.lower())
        training_labels.append(intent["tag"])

# 3. Vectorize Text & Train the Model
# TF-IDF converts text strings into a matrix of numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_sentences)
y_train = training_labels

# FIX APPLIED HERE: C=100 reduces regularization, increasing confidence on exact matches
model = LogisticRegression(C=100, max_iter=200)
model.fit(X_train, y_train)

# 4. Chatbot Logic Layer
def predict_intent(user_text):
    """Processes input text and returns the predicted tag and confidence score."""
    # Transform user input using the trained vectorizer
    transformed_text = vectorizer.transform([user_text.lower()])
    
    # Get probability distribution across all tags
    probabilities = model.predict_proba(transformed_text)[0]
    max_prob_index = probabilities.argmax()
    
    predicted_tag = model.classes_[max_prob_index]
    confidence_score = probabilities[max_prob_index]
    
    return predicted_tag, confidence_score

def get_response(tag):
    """Finds a random response matching the predicted tag."""
    for intent in intents_data:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I encountered an internal error."

# 5. Live Conversational Loop
def run_chatbot():
    print("====================================================")
    print("SupportBot v2.1 (NLP Classifier Active - High Confidence)")
    print("Type 'quit' to exit the chat.")
    print("====================================================\n")
    
    CONFIDENCE_THRESHOLD = 0.60  # Keeping the safe threshold
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("SupportBot: Thank you for reaching out. Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Machine learning prediction step
        tag, confidence = predict_intent(user_input)
        
        # Fallback mechanism if confidence is too low
        if confidence < CONFIDENCE_THRESHOLD:
            bot_response = "I'm not completely sure I understand. Could you rephrase your question, or would you like to speak to an agent?"
            log_tag = "fallback_triggered"
        else:
            bot_response = get_response(tag)
            log_tag = tag
            
        print(f"SupportBot: {bot_response}")
        
        # Mock Database Logger 
        print(f"   [LOG: Intent='{log_tag}' | Confidence={confidence:.2f}]\n")

if __name__ == "__main__":
    run_chatbot()