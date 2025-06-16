"""
Topic Modeling Script

This script performs topic modeling on text data extracted from an SQLite database.
It uses BERTopic for topic modeling, SentenceTransformers for embeddings, and
allows for customization through command-line arguments.

Key functionalities:
- Loads messages from an SQLite database.
- Filters messages by a minimum word count.
- Uses a specified sentence embedding model.
- Allows custom stopwords.
- Performs topic modeling to identify key themes.
- Generates an interactive HTML visualization of topics over time.
- Supports GPU (CUDA) for faster processing if available.

To run the script, use the command line and specify arguments as needed.
Example:
    python topic_modeling.py --db-file "my_data.db" --min-word-count 5 --output-file "my_topics.html"

For help with command-line arguments:
    python topic_modeling.py --help
"""
import sqlite3
try:
    import pandas as pd
except ImportError as e:
    print("Error: Failed to import pandas. This might be due to an issue with pandas itself or one of its dependencies (like pyarrow).")
    print(f"Please check your pandas installation. Original error: {e}")
    import sys
    sys.exit(1)

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import torch
import argparse

# --- Function Definitions ---

def load_data(db_file: str, min_wc: int) -> pd.DataFrame:
    """
    Loads and preprocesses data from the specified SQLite database.
    Filters messages by a minimum word count.

    Args:
        db_file (str): The path to the SQLite database file.
                       Expected to contain a 'messages' table with
                       'text_content', 'create_time', and 'word_count' columns.
        min_wc (int): The minimum word count for messages to be included. Messages
                      with fewer words than this value will be excluded.

    Returns:
        pd.DataFrame: A DataFrame with 'text_content' and 'timestamp' (converted from 'create_time'),
                      containing only messages that meet the minimum word count.

    Raises:
        FileNotFoundError: If the database file does not exist at the given path.
        sqlite3.OperationalError: If the 'word_count' column is missing or other SQL errors occur.
    """
    if not os.path.exists(db_file):
        print(f"Error: Database file not found at '{db_file}'")
        raise FileNotFoundError(f"Database file not found at '{db_file}'")

    print(f"Connecting to {db_file} and loading messages with at least {min_wc} words...")
    with sqlite3.connect(db_file) as conn:
        query = f"SELECT text_content, create_time FROM messages WHERE text_content IS NOT NULL AND word_count >= {min_wc}"
        try:
            df = pd.read_sql_query(query, conn)
        except sqlite3.OperationalError as e:
            print(f"SQL Error: {e}. Ensure 'messages' table has 'text_content', 'create_time', and 'word_count' columns.")
            raise
    df['timestamp'] = pd.to_datetime(df['create_time'], unit='s')
    print(f"Successfully loaded {len(df)} messages meeting the word count criteria of >= {min_wc} words.")
    return df

def load_stopwords(stopwords_path: str) -> list[str]:
    """
    Loads stop words from a specified text file. Each stop word should be on a new line.

    Args:
        stopwords_path (str): The path to the file containing stopwords.

    Returns:
        list[str]: A list of stop words. Returns an empty list if the file is not found
                   or if an error occurs during reading, along with a printed warning.
    """
    if not os.path.exists(stopwords_path):
        print(f"⚠️ Warning: Stopwords file not found at '{stopwords_path}'. Proceeding without custom stopwords.")
        return []
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        print(f"Successfully loaded {len(stopwords)} stopwords from '{stopwords_path}'.")
        return stopwords
    except Exception as e:
        print(f"Error loading stopwords from '{stopwords_path}': {e}. Proceeding without custom stopwords.")
        return []

def initialize_models(dewey_model_path: str, stopwords_path: str, device: str) -> tuple[SentenceTransformer, CountVectorizer]:
    """
    Initializes and returns the sentence embedding model and the CountVectorizer.
    Custom stop words are loaded. The embedding model is loaded onto the specified device.

    Args:
        dewey_model_path (str): The file path to the pre-trained SentenceTransformer model.
        stopwords_path (str): The file path to the text file containing custom stopwords.
        device (str): The device to load the SentenceTransformer model onto (e.g., 'cuda' or 'cpu').
                      This determines where the embedding model computations will run.

    Returns:
        tuple[SentenceTransformer, CountVectorizer]: A tuple containing the initialized
                                                     SentenceTransformer model and the
                                                     CountVectorizer model.

    Raises:
        FileNotFoundError: If the Dewey model file does not exist at `dewey_model_path`.
    """
    if not os.path.exists(dewey_model_path):
        print(f"Error: Dewey model not found at '{dewey_model_path}'")
        raise FileNotFoundError(f"Dewey model not found at '{dewey_model_path}'")

    print(f"Loading embedding model from '{dewey_model_path}' onto device '{device}'...")
    embedding_model = SentenceTransformer(dewey_model_path, device=device)
    print("Embedding model loaded successfully.")

    stop_words_list = load_stopwords(stopwords_path)

    print(f"Initializing vectorizer...")
    vectorizer_model = CountVectorizer(stop_words=stop_words_list if stop_words_list else None)
    if stop_words_list:
        print("Vectorizer initialized with custom stop words.")
    else:
        print("Vectorizer initialized. No custom stop words were loaded (file might be missing, empty, or error during load).")

    return embedding_model, vectorizer_model

def perform_topic_modeling(df: pd.DataFrame, embedding_model: SentenceTransformer,
                           vectorizer_model: CountVectorizer, device: str) -> tuple[list[int], list[float], BERTopic]:
    """
    Performs topic modeling on the provided data using BERTopic.
    Disables PyTorch gradients before model fitting for efficiency, especially on GPU.

    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
                           Requires a column named 'text_content' for the documents.
        embedding_model (SentenceTransformer): The pre-trained sentence embedding model.
        vectorizer_model (CountVectorizer): The initialized CountVectorizer model.
        device (str): The device being used (e.g., 'cuda' or 'cpu'). This argument is noted,
                      and torch gradients are disabled globally before fitting, which is
                      particularly beneficial for GPU operations.

    Returns:
        tuple[list[int], list[float], BERTopic]: A tuple containing:
            - topics (list[int]): A list of topic assignments for each document.
            - probs (list[float]): A list of probabilities for the topic assignments.
            - topic_model (BERTopic): The fitted BERTopic model instance.
    """
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        verbose=True # BERTopic will show its progress.
    )

    print("\nStarting BERTopic model fitting. This may take a significant amount of time depending on data size and hardware...")
    # Disable gradients for efficiency during inference/embedding generation.
    # This is a global PyTorch setting.
    torch.set_grad_enabled(False)
    print("PyTorch gradients disabled for embedding generation phase.")

    topics, probs = topic_model.fit_transform(df['text_content'])

    # Optional: Re-enable gradients if other parts of the script needed them.
    # For this script, it's generally not necessary to re-enable them after this point.
    # torch.set_grad_enabled(True)

    print(f"Topic modeling complete. Found {topic_model.get_topic_info()['Topic'].max() + 1} topics (excluding potential outliers identified as topic -1).")
    return topics, probs, topic_model

def visualize_and_save(topic_model: BERTopic, df: pd.DataFrame, topics: list[int], output_html_file: str) -> None:
    """
    Generates a visualization of topics over time and saves it as an HTML file.

    Args:
        topic_model (BERTopic): The fitted BERTopic model instance.
        df (pd.DataFrame): The DataFrame used for topic modeling.
                           Requires 'text_content' and 'timestamp' columns.
        topics (list[int]): The list of topic assignments for each document in `df`.
        output_html_file (str): The path and filename for the output HTML visualization.
    """
    print(f"\nGenerating topics over time visualization...")
    topics_over_time_data = topic_model.topics_over_time(df['text_content'], topics, df['timestamp'])

    fig = topic_model.visualize_topics_over_time(
        topics_over_time=topics_over_time_data,
        title="<b>Topics Over Time (Refined)</b>"
    )

    fig.write_html(output_html_file)
    print(f"\n✅ Success! Interactive visualization saved to '{output_html_file}'")
    print("You can now open this HTML file in your web browser to explore the topics.")

# --- Main Execution ---
def run_topic_modeling(db_file_path: str, model_path_str: str, output_html_path: str,
                       stopwords_file_path: str, min_word_count: int) -> None:
    """
    Main orchestrator for the topic modeling pipeline.
    Handles GPU/CPU device selection, data loading (with word count filtering),
    model initialization (with device selection for embedding model),
    topic modeling (with gradient disabling), and visualization.

    Args:
        db_file_path (str): Path to the SQLite database file.
        model_path_str (str): Path to the sentence embedding model.
        output_html_path (str): Path for the output HTML visualization.
        stopwords_file_path (str): Path to the stopwords text file.
        min_word_count (int): Minimum word count for messages to be included in processing.
    """
    # Determine compute device (GPU if available, else CPU)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda":
        print(f"✅ GPU found: {torch.cuda.get_device_name(0)}. Topic modeling will utilize the GPU for embedding model.")
    else:
        print("⚠️ Warning: No GPU found. BERTopic can run on CPU, but embedding generation will be significantly slower.")

    try:
        # Step 1: Load data from the database, applying word count filter.
        data_df = load_data(db_file_path, min_word_count)

        # Step 2: Initialize embedding and vectorizer models. Embedding model uses selected device.
        embedding_model, vectorizer_model = initialize_models(model_path_str, stopwords_file_path, device_str)

        # Step 3: Perform the core topic modeling process. Gradients are disabled within this function.
        topics, probabilities, topic_model_instance = perform_topic_modeling(data_df, embedding_model, vectorizer_model, device_str)

        # Step 4: Generate and save the visualization of topics over time.
        visualize_and_save(topic_model_instance, data_df, topics, output_html_path)

    except FileNotFoundError as e:
        print(f"Process aborted due to a missing critical file: {e}")
        print("Please ensure all configured file paths (database, model) are correct and accessible.")
    except sqlite3.OperationalError as e:
        print(f"Database error occurred: {e}")
        print("Ensure your database schema is correct (e.g., 'word_count' column exists in 'messages' table if using --min-word-count > 0).")
    except Exception as e:
        print(f"An unexpected error occurred during the topic modeling process: {e}")
        print("Review the error message and stack trace for more details.")

if __name__ == "__main__":
    # --- Argument Parsing ---
    # Sets up command-line argument parsing for script configuration.
    parser = argparse.ArgumentParser(
        description="Run topic modeling on text data from an SQLite database. Creates an interactive HTML visualization of topics over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message.
    )

    parser.add_argument("--db-file",
                        default="2years.db",
                        help="Path to the SQLite database file.")
    parser.add_argument("--model-path",
                        default=os.path.expanduser("~/llama.cpp/downloads/dewey_en_beta"),
                        help="Path to the SentenceTransformer embedding model directory.")
    parser.add_argument("--output-file",
                        default="topics_over_time_refined.html",
                        help="Filename for the output HTML visualization.")
    parser.add_argument("--stopwords-file",
                        default="stopwords.txt",
                        help="Path to a text file containing custom stopwords (one per line).")
    parser.add_argument("--min-word-count",
                        type=int,
                        default=4,
                        help="Minimum word count for messages to be included in the analysis.")

    args = parser.parse_args()

    # --- Main Process Execution ---
    # Calls the main function with parsed command-line arguments.
    run_topic_modeling(args.db_file,
                       args.model_path,
                       args.output_file,
                       args.stopwords_file,
                       args.min_word_count)
