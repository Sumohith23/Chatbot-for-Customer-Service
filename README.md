# Chatbot-for-Customer-Service
This Python script implements a machine learning-based conversational agent, designated in the code as "SupportBot v2.1". It serves as a foundational customer support chatbot that uses Natural Language Processing (NLP) to classify user input and return appropriate, predefined responses.

Here is a breakdown of its core components and functionality:

1. Intent-Based DatasetThe bot relies on a hardcoded, dictionary-based dataset intents_data containing specific support categories, or "tags" (e.g., greeting, shipping, returns, and human_agent). Each tag includes:  
Patterns: Example phrases a user might type (e.g., "where is my order", "i want a refund").  
Responses: A list of appropriate replies the bot can select from.py

2. Text Vectorization and Model TrainingThe script prepares the data by extracting and lowercasing all patterns to train the model.  
Vectorization: It utilizes TfidfVectorizer from the sklearn library to convert the textual patterns into a numerical matrix (TF-IDF) so the algorithm can process it. Classification: The core brain is a LogisticRegression model. A specific fix is applied by setting the inverse regularization parameter C=100, which increases the model's confidence when encountering exact or near-exact pattern matches. 

3. Prediction and Response LogicWhen a user submits a message, the script processes it through two main functions:
predict_intent: Transforms the user's text through the vectorizer and asks the Logistic Regression model to predict the intent. It calculates the probability distribution across all known tags and returns both the most likely tag and its associated confidence score.  
get_response: Takes the predicted tag and randomly selects one of the predefined responses from the intents_data dictionary.  

4. Interactive Chat Loop & Fallback ProtocolThe run_chatbot function powers the live, command-line interface.  
Threshold: It implements a CONFIDENCE_THRESHOLD set to 0.60 (60%).  
Fallback: If the model's confidence score falls below this threshold, the bot rejects its own prediction and triggers a fallback response, asking the user to rephrase their query or offering to connect them to a human agent.  
Logging: After every interaction, the system prints a simulated mock database log detailing the matched intent (or fallback_triggered) and the exact confidence score of the prediction.  
