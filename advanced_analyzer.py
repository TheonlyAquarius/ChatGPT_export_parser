import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer - Tfidf not explicitly requested for merge
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import DBSCAN # Not explicitly requested for merge from "corrected"
# from sklearn.decomposition import PCA # Not explicitly requested for merge
from sklearn.preprocessing import StandardScaler # Used in semantic drift example
from scipy import stats
from scipy.spatial.distance import jensenshannon
from collections import Counter, defaultdict
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px # For more plot types
import os
import torch
import argparse
from datetime import datetime # Not explicitly listed but often used with timestamps
import warnings
import re # For feature engineering
import json # For saving structured data like interactions

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='hdbscan')
warnings.filterwarnings("ignore", category=FutureWarning, module='hdbscan')
warnings.filterwarnings("ignore", category=FutureWarning, module='torch')


class AdvancedTopicAnalyzer:
    def __init__(self, embedding_model_path: str = "all-MiniLM-L6-v2", device: str = "auto"):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"Using device: {self.device}")

        try:
            self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device)
        except Exception as e:
            print(f"Error loading SentenceTransformer model from {embedding_model_path} on device {self.device}: {e}")
            print("Attempting to load on CPU as a fallback...")
            try:
                self.device = 'cpu'
                self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device)
                print(f"Successfully loaded model on CPU: {embedding_model_path}")
            except Exception as e_cpu:
                print(f"Fatal error: Could not load SentenceTransformer model on CPU either: {e_cpu}")
                raise

        self.topic_model = None
        self.df = None
        self.topics = None
        self.probabilities = None # BERTopic can return these
        self.embeddings = None
        self.hierarchical_topics = None

    def load_enhanced_data(self, db_file: str, min_wc: int = 0, query: str = None):
        if not os.path.exists(db_file):
            raise FileNotFoundError(f"Database file not found: {db_file}")

        print(f"Connecting to database: {db_file}")
        conn = sqlite3.connect(db_file)

        if query is None:
            base_query = "SELECT text_content, create_time, word_count FROM messages WHERE text_content IS NOT NULL"
            # SQL Safety Fix for min_wc
            if min_wc > 0:
                actual_query = f"{base_query} AND CAST(word_count AS INTEGER) >= ?"
                params = (min_wc,)
            else:
                actual_query = base_query
                params = ()
        else: # Custom query
            actual_query = query
            params = (min_wc,) if '?' in actual_query and min_wc > 0 else ()

        print(f"Executing query: {actual_query} with params: {params if params else 'None'}")
        try:
            df = pd.read_sql_query(actual_query, conn, params=params)
        except sqlite3.OperationalError as e:
            print(f"Error executing SQL query: {e}. Ensure table and columns exist.")
            conn.close()
            raise
        conn.close()

        if df.empty:
            print("Warning: Loaded DataFrame is empty.")
            self.df = pd.DataFrame() # Ensure df is an empty DataFrame not None
            return

        print(f"Loaded {len(df)} messages.")
        df['timestamp'] = pd.to_datetime(df['create_time'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name() # Full name
        df['month'] = df['timestamp'].dt.month_name() # Full name
        df['text_length'] = df['text_content'].apply(len)
        df['sentence_count'] = df['text_content'].apply(lambda x: len(re.findall(r'[.!?]+', str(x))) + 1)
        df['avg_word_length'] = df['text_content'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
        df['question_marks'] = df['text_content'].apply(lambda x: str(x).count('?'))
        df['exclamation_marks'] = df['text_content'].apply(lambda x: str(x).count('!'))
        df['caps_ratio'] = df['text_content'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
        print("Feature engineering complete.")
        self.df = df

    def perform_hierarchical_topic_modeling(self, stopwords_path: str = None):
        if self.df is None or self.df.empty:
            print("DataFrame is not loaded. Call load_enhanced_data() first.")
            return
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized.")

        print("Performing hierarchical topic modeling...")
        stop_words_list = []
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f: # Ensure encoding
                stop_words_list = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(stop_words_list)} custom stopwords from {stopwords_path}")

        vectorizer = CountVectorizer(
            stop_words=stop_words_list if stop_words_list else 'english',
            max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.95
        )
        print("CountVectorizer configured.")

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer,
            verbose=True, min_topic_size=10, nr_topics='auto',
            calculate_probabilities=True # Good addition
        )

        with torch.no_grad(): # Ensure PyTorch gradient handling
            print("Generating embeddings (within torch.no_grad())...")
            self.embeddings = self.embedding_model.encode(self.df['text_content'].tolist(), show_progress_bar=True)

            print("Fitting BERTopic model (within torch.no_grad())...")
            self.topics, self.probabilities = self.topic_model.fit_transform(self.df['text_content'].tolist(), self.embeddings)
            self.df['topic'] = self.topics

            print("Generating hierarchical topic structure (within torch.no_grad())...")
            self.hierarchical_topics = self.topic_model.hierarchical_topics(self.df['text_content'].tolist())

        print(f"Discovered {len(self.topic_model.get_topic_info())-1} topics (excluding outliers).")
        if self.hierarchical_topics is not None: print("Hierarchical topic structure generated.")

    def analyze_topic_lifecycle(self):
        print("Analyzing topic lifecycle...")
        if self.df is None or 'topic' not in self.df.columns or self.topic_model is None:
            print("Data or topic model not available. Perform modeling first.")
            return None

        self.df['date'] = self.df['timestamp'].dt.to_period('M') # Group by month
        topic_counts_over_time = self.df.groupby(['date', 'topic']).size().reset_index(name='count')

        lifecycle_data = {}
        for topic_id in self.df['topic'].unique():
            if topic_id == -1: continue
            topic_data = topic_counts_over_time[topic_counts_over_time['topic'] == topic_id]
            if not topic_data.empty:
                # Simple trend: emerging if increasing, fading if decreasing, stable if relatively constant
                # This is a very basic heuristic
                slope = np.polyfit(np.arange(len(topic_data)), topic_data['count'], 1)[0] if len(topic_data) > 1 else 0
                status = "emerging" if slope > 0.5 else "fading" if slope < -0.5 else "stable"
                lifecycle_data[topic_id] = {"status": status, "trend": topic_data[['date', 'count']].to_dict('records')}
        print("Topic lifecycle analysis complete.")
        return lifecycle_data

    def detect_topic_anomalies(self, z_thresh: float = 3.0): # Corrected method from previous steps
        print(f"Detecting topic anomalies with Z-score threshold: {z_thresh}...")
        if self.df is None or 'topic' not in self.df.columns or self.topic_model is None:
            print("Data or topic model not available. Perform modeling first.")
            return {'sudden_spikes': [], 'other_anomalies': []} # Ensure all expected keys are present

        anomalies = {'sudden_spikes': []} # Focused on sudden_spikes as per previous fix

        # Ensure 'timestamp' is datetime for dt accessor
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
             self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Group by date and topic to count daily occurrences
        # Using .dt.date ensures we group by calendar date, not specific timestamp
        daily_topic_counts = self.df.groupby([self.df['timestamp'].dt.date, 'topic']).size().reset_index(name='count')
        daily_topic_counts.rename(columns={'timestamp': 'date'}, inplace=True)


        for topic_id in self.df['topic'].unique():
            if topic_id == -1: continue # Skip outlier topic

            topic_counts_df = daily_topic_counts[daily_topic_counts['topic'] == topic_id].copy()

            if topic_counts_df.empty or len(topic_counts_df) < 2:
                continue

            counts_array = topic_counts_df['count'].values
            mean_count = np.mean(counts_array)
            std_count = np.std(counts_array)

            if std_count == 0: continue

            topic_counts_df['z_score'] = (counts_array - mean_count) / std_count
            spike_indices = np.where(np.abs(topic_counts_df['z_score']) > z_thresh)[0]

            if len(spike_indices) > 0:
                spike_dates = topic_counts_df.iloc[spike_indices]['date'].astype(str).tolist()
                spike_counts_values = counts_array[spike_indices].tolist()

                for i, date_str in enumerate(spike_dates):
                    anomalies['sudden_spikes'].append({
                        'topic': topic_id, 'date': date_str,
                        'count': spike_counts_values[i],
                        'z_score': topic_counts_df['z_score'].iloc[spike_indices[i]]
                    })
        print(f"Anomaly detection complete. Found {len(anomalies['sudden_spikes'])} spikes.")
        return anomalies


    def analyze_semantic_drift(self, target_topic_id, window_size=30, step_size=7):
        print(f"Analyzing semantic drift for topic {target_topic_id}...")
        if self.df is None or self.embeddings is None or 'topic' not in self.df.columns or self.topic_model is None:
            print("Data, embeddings, or topics not available.")
            return None

        topic_docs_indices = self.df[self.df['topic'] == target_topic_id].index
        if len(topic_docs_indices) < window_size : # Need enough docs for meaningful analysis
            print(f"Not enough documents for topic {target_topic_id} to analyze drift.")
            return None

        topic_embeddings = self.embeddings[topic_docs_indices]
        topic_timestamps = self.df.loc[topic_docs_indices, 'timestamp'].sort_values()

        # Simplified: compare centroid of first window to subsequent windows
        # A more robust approach would involve more sophisticated distribution comparison
        drift_scores = []
        window_dates = []

        # Ensure enough data for at least one window comparison
        if len(topic_timestamps) < window_size * 2:
            print(f"Not enough documents spread over time for topic {target_topic_id} for drift analysis with window size {window_size}.")
            return {"drift_scores": [], "timestamps": []}

        first_window_indices = topic_timestamps.iloc[:window_size].index
        first_window_centroid = np.mean(self.embeddings[self.df.loc[first_window_indices].index], axis=0) # Use main embeddings

        for i in range(window_size, len(topic_timestamps) - window_size + 1, step_size):
            current_window_indices = topic_timestamps.iloc[i : i + window_size].index
            current_window_centroid = np.mean(self.embeddings[self.df.loc[current_window_indices].index], axis=0)

            # Cosine distance (1 - cosine_similarity)
            distance = 1 - cosine_similarity(first_window_centroid.reshape(1, -1), current_window_centroid.reshape(1, -1))[0,0]
            drift_scores.append(distance)
            window_dates.append(topic_timestamps.iloc[i + window_size // 2]) # Midpoint of current window

        print("Semantic drift analysis complete.")
        return {"drift_scores": drift_scores, "timestamps": window_dates}

    def analyze_topic_interactions(self, method='correlation', threshold=0.1):
        print(f"Analyzing topic interactions using {method} method...")
        if self.df is None or self.probabilities is None or self.topic_model is None:
            print("Probabilities or topic model not available.")
            return None

        if self.probabilities.shape[1] <= 1: # Need at least 2 topics for interaction
            print("Not enough topics to analyze interactions.")
            return pd.DataFrame()

        if method == 'correlation':
            # Ensure probabilities is a dense array for correlation
            prob_df = pd.DataFrame(self.probabilities, columns=[f"Topic_{i}" for i in range(self.probabilities.shape[1])])
            # Exclude outlier topic if present (assuming it's the last column if nr_topics was fixed, or find by name)
            # This is complex if nr_topics='auto'. For simplicity, correlate all.
            interaction_matrix = prob_df.corr()
        else: # Placeholder for other methods like co-occurrence
            print(f"Method {method} not fully implemented. Using correlation as fallback.")
            interaction_matrix = pd.DataFrame(self.probabilities).corr()

        # Filter by threshold and build graph for visualization (optional)
        # For now, just return the matrix or significant pairs
        significant_interactions = interaction_matrix[abs(interaction_matrix) > threshold].stack().reset_index()
        significant_interactions = significant_interactions[significant_interactions['level_0'] != significant_interactions['level_1']]
        significant_interactions.columns = ['Topic1', 'Topic2', 'Strength']
        print("Topic interaction analysis complete.")
        return significant_interactions

    def create_advanced_visualizations(self, output_dir: str):
        print(f"Creating advanced visualizations in {output_dir}...")
        if self.topic_model is None:
            print("Topic model not available.")
            return
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Standard BERTopic visualizations
            if hasattr(self.topic_model, 'visualize_topics'):
                fig = self.topic_model.visualize_topics()
                fig.write_html(os.path.join(output_dir, "intertopic_distance_map.html"))
            if hasattr(self.topic_model, 'visualize_hierarchy') and self.hierarchical_topics is not None:
                fig = self.topic_model.visualize_hierarchy(hierarchical_topics=self.hierarchical_topics)
                fig.write_html(os.path.join(output_dir, "hierarchical_topic_tree.html"))
            if hasattr(self.topic_model, 'visualize_heatmap') and self.probabilities is not None:
                 # This might need specific setup or might be too slow for large datasets
                try:
                    fig = self.topic_model.visualize_heatmap()
                    fig.write_html(os.path.join(output_dir, "topic_similarity_heatmap.html"))
                except Exception as e_heatmap:
                    print(f"Could not generate heatmap: {e_heatmap}")

            if hasattr(self.topic_model, 'visualize_barchart'):
                fig = self.topic_model.visualize_barchart(top_n_topics=10)
                fig.write_html(os.path.join(output_dir, "topic_term_scores_barchart.html"))

            # Custom: Topic trends over time (if lifecycle data was generated)
            lifecycle_data = self.analyze_topic_lifecycle() # Re-call or pass data
            if lifecycle_data:
                for topic_id, data in lifecycle_data.items():
                    if data['trend']:
                        df_trend = pd.DataFrame(data['trend'])
                        df_trend['date_str'] = df_trend['date'].astype(str) # Convert Period to string for Plotly
                        fig = px.line(df_trend, x='date_str', y='count', title=f"Trend for Topic {topic_id} ({data['status']})")
                        fig.write_html(os.path.join(output_dir, f"topic_{topic_id}_trend.html"))

            print(f"Advanced visualizations saved in {output_dir}")
        except Exception as e:
            print(f"Error during visualization generation: {e}")


    def generate_insights_report(self, lifecycle_data, anomalies, semantic_drift_results, interactions_data, output_dir: str):
        # Signature matches inferred "original" script
        report_path = os.path.join(output_dir, "advanced_topic_analysis_report.md")
        print(f"Generating insights report at {report_path}...")
        report_parts = ["# Advanced Topic Analysis Report\n"]

        report_parts.append("## 📝 Key Insights Summary\n")
        if self.topic_model:
            num_topics = len(self.topic_model.get_topic_info()) -1 # Exclude -1 outlier
            report_parts.append(f"- Discovered {num_topics} distinct topics.\n")
        else:
            report_parts.append("- Topic modeling not performed or failed.\n")

        report_parts.append("\n## 📊 Detailed Topic Findings\n")
        if self.topic_model:
            try:
                top_topics = self.topic_model.get_topic_info().head(11) # Top 10 + outlier
                for _, row in top_topics.iterrows():
                    report_parts.append(f"- **Topic {row['Topic']}**: {row['Name']} (Count: {row['Count']})\n")
            except Exception as e: report_parts.append(f"- Could not retrieve top topics: {e}\n")

        report_parts.append("\n## 📈 Topic Lifecycle Analysis\n")
        if lifecycle_data:
            for topic_id, data in lifecycle_data.items():
                report_parts.append(f"- Topic {topic_id}: Status - {data['status']}\n")
        else: report_parts.append("- Lifecycle data not available.\n")

        report_parts.append("\n## 🚨 Anomalies and Unexpected Patterns\n") # Corrected section name
        if anomalies and anomalies.get('sudden_spikes'):
            report_parts.append("### ⚡ Sudden Topic Spikes Detected\n")
            for spike in anomalies['sudden_spikes']:
                report_parts.append(f"- Topic {spike['topic']} spiked on {spike['date']} (Count: {spike['count']}, Z-score: {spike.get('z_score', 'N/A'):.2f}).\n")
        else: report_parts.append("- No significant sudden topic spikes detected.\n")

        report_parts.append("\n## 🌳 Hierarchical Topic Structure\n") # Added as per previous step
        if self.hierarchical_topics is not None and not self.hierarchical_topics.empty:
            report_parts.append("This tree shows how specific sub-topics are related to broader parent themes.\n\n")
            report_parts.append("```\n")
            try: report_parts.append(self.topic_model.get_topic_tree(self.hierarchical_topics))
            except Exception as e: report_parts.append(f"Could not generate topic tree visualization: {e}")
            report_parts.append("\n```\n")
        else: report_parts.append("- Hierarchical topic data not available or empty.\n")

        report_parts.append("\n## 🌊 Semantic Drift Analysis\n")
        if semantic_drift_results and semantic_drift_results.get('drift_scores'):
            report_parts.append(f"- Semantic drift for analyzed topic(s) shows scores: {semantic_drift_results['drift_scores']}\n") # Simplified
        else: report_parts.append("- Semantic drift analysis not performed or no significant drift detected.\n")

        report_parts.append("\n## 🔗 Topic Interactions\n")
        if interactions_data is not None and not interactions_data.empty:
            report_parts.append("Significant topic interactions (e.g., correlation > threshold):\n")
            report_parts.append("```\n" + interactions_data.to_string(index=False) + "\n```\n")
        else: report_parts.append("- No significant topic interactions found or analysis not performed.\n")

        final_report_str = "\n".join(report_parts)
        try:
            with open(report_path, "w", encoding="utf-8") as f: # Ensure encoding
                f.write(final_report_str)
            print(f"Insights report successfully saved to: {report_path}")
        except IOError as e:
            print(f"Error saving report to file {report_path}: {e}")

# --- Helper Function for Orchestration (as per original script structure) ---
def run_advanced_analysis(db_file: str,
                          output_dir: str,
                          embedding_model_path: str = "all-MiniLM-L6-v2",
                          stopwords_file: str = None,
                          min_word_count: int = 5,
                          device: str = "auto"):
    print("Starting Advanced Topic Analysis Pipeline...")
    os.makedirs(output_dir, exist_ok=True)

    analyzer = AdvancedTopicAnalyzer(embedding_model_path=embedding_model_path, device=device)
    try:
        analyzer.load_enhanced_data(db_file=db_file, min_wc=min_word_count)
        if analyzer.df is None or analyzer.df.empty: return
    except Exception as e:
        print(f"Error in data loading: {e}"); return

    analyzer.perform_hierarchical_topic_modeling(stopwords_path=stopwords_file)
    if analyzer.topic_model is None:
        print("Aborting further analysis as topic modeling failed."); return

    lifecycle_data = analyzer.analyze_topic_lifecycle()
    anomalies = analyzer.detect_topic_anomalies() # Uses corrected version
    # For simplicity, analyze drift for a few top topics if available
    semantic_drift_results = {}
    top_topic_ids = [info['Topic'] for info in analyzer.topic_model.get_topic_info().to_dict('records') if info['Topic'] != -1][:2]
    for topic_id in top_topic_ids:
        semantic_drift_results[topic_id] = analyzer.analyze_semantic_drift(target_topic_id=topic_id)

    interactions_data = analyzer.analyze_topic_interactions()

    vis_output_dir = os.path.join(output_dir, "visualizations")
    analyzer.create_advanced_visualizations(output_dir=vis_output_dir)

    analyzer.generate_insights_report(lifecycle_data, anomalies, semantic_drift_results, interactions_data, output_dir) # Pass output_dir for report path

    print("Advanced Topic Analysis Pipeline Completed.")
    print(f"Report and visualizations saved in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced Topic Analysis CLI")
    parser.add_argument("--db-file", type=str, required=True, help="Path to the SQLite database file.")
    parser.add_argument("--output-dir", type=str, default="topic_analysis_output", help="Directory to save analysis results.")
    parser.add_argument("--model-path", type=str, default="all-MiniLM-L6-v2", help="Path or HuggingFace name of SentenceTransformer model.")
    parser.add_argument("--stopwords-file", type=str, default=None, help="Optional path to a custom stopwords file.")
    parser.add_argument("--min-word-count", type=int, default=5, help="Minimum word count for messages.")
    # Added arguments from inferred "original script" context
    parser.add_argument("--device", type=str, default="auto", choices=['auto', 'cuda', 'cpu'], help="Device for computations.")
    # Potentially more args from the original script could be added here

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_advanced_analysis(
        db_file=args.db_file,
        output_dir=args.output_dir,
        embedding_model_path=args.model_path,
        stopwords_file=args.stopwords_file,
        min_word_count=args.min_word_count,
        device=args.device
    )
```
