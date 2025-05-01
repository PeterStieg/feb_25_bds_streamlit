import streamlit as st
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import Normalizer

st.title("Word2Vec Model")

# Load dictionaries and vocabulary size
try:
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)

    with open("idx2word.pkl", "rb") as f:
        idx2word = pickle.load(f)

    # Calculate vocabulary size from the dictionaries
    vocab_size = len(word2idx) + 1  # +1 for padding/OOV token

    st.success(f"Successfully loaded vocabulary with {vocab_size:,} words")

except Exception as e:
    st.error(f"Error loading vocabulary files: {e}")
    st.info(
        "Make sure files 'word2idx.pkl' and 'idx2word.pkl' exist and are in app's directory."
    )
    st.stop()

# Try two different approaches to load the model
try:
    # Approach 1: Load the entire model
    try:
        model = load_model("word2vec.h5")
        st.success("Successfully loaded complete model")
    except:
        # Approach 2: Create model and load weights
        embedding_dim = 300
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(vocab_size, activation="softmax"))

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Load weights
        model.load_weights("word2vec.h5")
        st.success("Successfully loaded model weights")

    # Extract the word vectors
    vectors = model.layers[0].get_weights()[0]

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info(
        "Make sure the 'word2vec.h5' file exists and matches the model architecture"
    )
    st.stop()


def dot_product(vec1, vec2):
    return np.sum((vec1 * vec2))


def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2) / np.sqrt(
        dot_product(vec1, vec1) * dot_product(vec2, vec2)
    )


def find_closest(word_index, vectors, number_closest):
    list1 = []
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])
    return np.asarray(sorted(list1, reverse=True)[:number_closest])


def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1 = []
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector = normalizer.fit_transform([query_vector], "l2")
    query_vector = query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])
    return np.asarray(sorted(list1, reverse=True)[:number_closest])


# Create a Streamlit function to display similar words
def st_display_closest(word, number=10):
    if word in word2idx:
        index_closest_words = find_closest(word2idx[word], vectors, number)
        results = []
        for i, index_word in enumerate(index_closest_words):
            word_result = idx2word[int(index_word[1])]
            similarity = float(index_word[0])
            results.append(
                {"#": i + 1, "Word": word_result, "Similarity": f"{similarity:.4f}"}
            )
        return results
    else:
        return None


# UI Section 1: Find similar words
st.header("Find Similar Words")
example_words = ["movie", "actor", "zombie", "love", "horror", "comedy"]
word_input = st.selectbox(
    "Select a word or type your own:", options=[""] + example_words, index=0
)

# Allow custom word input
custom_word = st.text_input("Or type a custom word:", "")
if custom_word:
    word_input = custom_word

number_words = st.slider("Number of similar words to display:", 5, 30, 10)

if word_input and st.button("Find Similar Words"):
    if word_input in word2idx:
        st.write(f"Words most similar to '{word_input}':")
        results = st_display_closest(word_input, number_words)
        st.table(results)
    else:
        st.error(f"'{word_input}' not found in vocabulary. Try another word.")

# UI Section 2: Word Arithmetic
st.header("Word Arithmetic")
st.write("Explore semantic relationships (e.g., king - man + woman ≈ queen)")

col1, col2, col3 = st.columns(3)
with col1:
    word1 = st.text_input("Positive word 1:", "king")
with col2:
    word2 = st.text_input("Negative word:", "man")
with col3:
    word3 = st.text_input("Positive word 2:", "woman")

num_results = st.slider("Number of results:", 1, 20, 5)

if st.button("Calculate"):
    missing_words = []
    if word1 not in word2idx:
        missing_words.append(word1)
    if word2 not in word2idx:
        missing_words.append(word2)
    if word3 not in word2idx:
        missing_words.append(word3)

    if missing_words:
        st.error(
            f"The following words are not in the vocabulary: {', '.join(missing_words)}"
        )
        st.info("Try using different words that might be in the movie reviews dataset.")
    else:
        st.write(f"Results for: {word1} - {word2} + {word3}")
        results = compare(
            word2idx[word1], word2idx[word2], word2idx[word3], vectors, num_results
        )

        # Create a table for results
        data = []
        for i, result in enumerate(results):
            word = idx2word[int(result[1])]
            similarity = float(result[0])
            data.append({"#": i + 1, "Word": word, "Similarity": f"{similarity:.4f}"})

        st.table(data)

# Add explanatory information in the sidebar
st.sidebar.header("About Word2Vec")
st.sidebar.write(
    """
Word2Vec is a technique for learning word embeddings - vector representations of words 
that capture their semantic meaning and relationships.

This model was trained on movie reviews, so the vocabulary and learned 
relationships reflect language used in that domain.
"""
)

st.sidebar.header("How it works")
st.sidebar.write(
    """
The model represents each word as a 300-dimensional vector. Words with similar 
meanings or that appear in similar contexts will have vectors that are close 
to each other in this 300-dimensional space.

The 'Similar Words' function finds words whose vectors are closest to the 
target word's vector using cosine similarity.

The 'Word Arithmetic' function performs vector operations that preserve semantic 
relationships. For example, "king - man + woman ≈ queen" works because the model 
learns the gender relationship between words.
"""
)
