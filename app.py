from flask import Flask, request, jsonify
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from waitress import serve

app = Flask(__name__)

# Function to check if embeddings file exists
def embeddings_file_exists(file_path):
    return os.path.exists(file_path)

# Function to save embeddings
def save_embeddings(file_path, embeddings):
    np.save(file_path, embeddings)

# Function to load embeddings
def load_embeddings(file_path):
    return np.load(file_path, allow_pickle=True)

# Load the JSON file
with open('combined_avantages_absences_by_intent.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# File paths
embeddings_file = 'embeddings.npy'

# Extract sentences and intents
sentences = []
intents = []
for parent_intent, child_intents in data.items():
    for child_intent, details in child_intents.items():
        for sentence in details['Questions']:
            sentences.append(sentence)
            intents.append(child_intent)

# Load or compute embeddings
if embeddings_file_exists(embeddings_file):
    print("Loading embeddings from file...")
    embeddings = load_embeddings(embeddings_file)
    print("Embeddings loaded successfully.")
else:
    print("Loading the model...")
    model = SentenceTransformer("OrdalieTech/Solon-embeddings-large-0.1")
    print("Model loaded successfully.")
    # Encode the sentences and save the embeddings
    print("Encoding sentences...")
    embeddings = model.encode(sentences)
    print("Sentences encoded successfully.")
    save_embeddings(embeddings_file, embeddings)

def find_most_similar_intents(input_sentence, num_intents=3):
    input_sentence = "query: " + input_sentence

    # Encode the new sentence
    new_embedding = model.encode([input_sentence])

    # Flatten the embedding if necessary
    if len(new_embedding.shape) == 3:
        new_embedding = new_embedding.squeeze()

    # Calculate similarity scores
    similarity_scores = cosine_similarity(new_embedding.reshape(1, -1), embeddings)

    # Get the indices of the most similar sentences and their confidence scores
    sorted_indices = np.argsort(similarity_scores[0])[::-1]
    top_intents = []
    unique_intents = set()

    for i in sorted_indices:
        intent = intents[i]
        confidence = float(similarity_scores[0][i])
        
        if intent not in unique_intents:
            top_intents.append({"intent": intent, "confidence": confidence})
            unique_intents.add(intent)

        if len(top_intents) >= num_intents:
            break

    return top_intents

@app.route('/get_intent', methods=['GET'])
def get_intent():
    input_sentence = request.args.get('sentence')
    if not input_sentence:
        return jsonify({"error": "No sentence provided"}), 400

    top_intents = find_most_similar_intents(input_sentence)
    return jsonify({"intents": top_intents})

if __name__ == '__main__':
   # Load the model once during startup if not already loaded
    if 'model' not in globals():
        print("Loading the model for the first time...")
        model = SentenceTransformer("OrdalieTech/Solon-embeddings-large-0.1")
        print("Model loaded successfully.")

    # Start the application with Waitress
    serve(app, host='0.0.0.0', port=5000)
