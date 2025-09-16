import pandas as pd
import spacy
import matplotlib.pyplot as plt
import textstat
from collections import Counter
import numpy as np
import pandas as pd
import yaml
from nltk.corpus import stopwords
import nltk

"""
Linguistic and YAML-metadata comparison between human-written and model-generated sentences.

This module:
- Loads human and model datasets (Excel/CSV).
- Parses YAML-structured fields to count entities, properties, relations, and parsing errors.
- Computes a suite of sentence-level statistics (lengths, n-gram repetition, readability, POS distribution).
- Produces side-by-side normalized bar charts for YAML metrics, text statistics, and POS distributions.
- Prints top words (with and without stopwords) for each corpus.

Dependencies:
    pandas, numpy, matplotlib, spacy, textstat, pyyaml, nltk
    spaCy model: `en_core_web_sm`
    NLTK resources: 'punkt', 'stopwords'

Data expectations:
    - human_df: columns ['sentence', 'YAML']
    - model_df: columns ['sentence', 'query'] where 'query' contains a YAML string

Notes:
    - File paths are currently hard-coded under ./data/ .
    - NLTK corpora are downloaded at runtime if missing.
"""

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load datasets
human_df = pd.read_excel("data/gold_annotations_05112024_old.xlsx")
# model_df = pd.read_csv("data/gpt_generations_dataset_v17_2_10k_yaml.csv")
model_df = pd.read_csv("data/gpt_generations_dataset_v17_newPrompt_10k_yaml.csv")
original_count = len(model_df)
model_df = model_df[model_df['sentence'] != "UNREALISTIC COMBINATION"]
filtered_count = len(model_df)
print(f"Removed {original_count - filtered_count} rows with 'UNREALISTIC COMBINATION'")

# Extract sentences
human_sentences = human_df['sentence'].dropna().astype(str).tolist()
model_sentences = model_df['sentence'].dropna().astype(str).tolist()

# Extract and drop missing YAML
human_YAML_raw = human_df['YAML'].dropna().to_dict()
model_YAML_raw = model_df['query'].dropna().to_dict()

both_yamls = [human_YAML_raw, model_YAML_raw]
results = []

# Process separately
for id, curr_yaml in enumerate(both_yamls):
    # Counters
    entity_count = 0
    property_count = 0
    relation_count = 0
    errors = []

    for idx, yaml_str in curr_yaml.items():
        try:
            parsed = yaml.safe_load(yaml_str)
            entities = parsed.get("entities", [])
            relations = parsed.get("relations", [])

            entity_count += len(entities)
            relation_count += len(relations)

            for ent in entities:
                props = ent.get("properties", [])
                property_count += len(props)
        except Exception as e:
            errors.append((idx, str(e)))

    # Normalize
    sample_count = len(curr_yaml)
    results.append({
        "Entities": entity_count / sample_count,
        "Properties": property_count / sample_count,
        "Relations": relation_count / sample_count,
        "Parsing Errors": len(errors) / sample_count
    })
    print(">>Current data source:", ["Human", "Model"][id])
    # print(">Values raw:")
    # print("Entities:", entity_count)
    # print("Properties:", property_count)
    # print("Relations:", relation_count)
    # print("Parsing errors:", len(errors))
    print(">Values normalized by number of samples:")
    print("Entities:", entity_count/sample_count)
    print("Properties:", property_count/sample_count)
    print("Relations:", relation_count/sample_count)
    print("Parsing errors:", len(errors)/sample_count)

df_plot = pd.DataFrame(results, index=["Human", "Model"]).T.reset_index()
df_plot.columns = ["Feature", "Human_norm", "Model_norm"]

# Plotting
def plot_normalized_comparison(df):
    """
    Plot a side-by-side bar chart of normalized values for Human vs. Model.

    Args:
        df (pandas.DataFrame): A dataframe with columns:
            - 'Feature': str feature names
            - 'Human_norm': float normalized value for the human set
            - 'Model_norm': float normalized value for the model set

    Returns:
        None: Displays a matplotlib figure.

    Notes:
        This is the stand-alone (axes-owning) variant that uses the global pyplot
        state to create a new figure. The module later defines an axes-based
        variant with the same name that accepts an Axes object.
    """
    x = range(len(df))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x, df['Human_norm'], width=width, label='Human')
    plt.bar([i + width for i in x], df['Model_norm'], width=width, label='Model')
    plt.xticks([i + width/2 for i in x], df['Feature'], rotation=45, ha='right')
    plt.title("Normalized Comparison (Entities, Properties, Relations, Errors)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_ngrams(tokens, n):
    """
    Generate n-grams from a sequence of tokens.

    Args:
        tokens (Sequence[str]): Tokenized text.
        n (int): Size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
        iterator[tuple[str, ...]]: A zip-based iterator yielding n-length tuples.
    """
    return zip(*[tokens[i:] for i in range(n)])

def analyze_sentences(sentences):
    """
    Compute linguistic statistics, frequency info, readability, and POS distribution.

    Args:
        sentences (list[str]): Collection of raw sentence strings.

    Returns:
        dict: A dictionary with the following keys:
            - 'avg_sentence_length_words' (float)
            - 'avg_sentence_length_chars' (float)
            - 'avg_word_length' (float)
            - 'type_token_ratio' (float)
            - 'flesch_reading_ease' (float)
            - 'flesch_kincaid_grade' (float)
            - 'gunning_fog_index' (float)
            - 'repeated_bigrams' (float): ratio of repeated bigrams over total bigrams
            - 'repeated_trigrams' (float): ratio of repeated trigrams over total trigrams
            - 'top_words' (list[tuple[str, int]]): top 20 tokens (with stopwords)
            - 'top_words_no_stopwords' (list[tuple[str, int]]): top 20 tokens (no stopwords)
            - 'pos_distribution' (dict[str, float]): normalized POS counts over total tokens

    Notes:
        - Uses spaCy for tokenization and POS tagging.
        - Uses textstat for readability metrics over the full concatenated text.
        - Only alphabetic tokens (token.is_alpha) are considered for many metrics.
    """
    stop_words = set(stopwords.words('english'))

    docs = list(nlp.pipe(sentences))
    num_sentences = len(docs)

    word_counts = [len([token for token in doc if token.is_alpha]) for doc in docs]
    char_counts = [len(doc.text) for doc in docs]
    word_lengths = [len(token.text) for doc in docs for token in doc if token.is_alpha]
    all_tokens = [token.text.lower() for doc in docs for token in doc if token.is_alpha]
    filtered_tokens = [word for word in all_tokens if word.lower() not in stop_words]

    unique_tokens = set(all_tokens)
    token_count = len(all_tokens)

    # POS Tag counts (normalized)
    pos_counts = {}
    for doc in docs:
        for token in doc:
            if token.is_alpha:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    total_words = len(all_tokens)
    normalized_pos = {k: v / total_words for k, v in pos_counts.items()}

    # N-gram repetition
    bigrams = list(get_ngrams(all_tokens, 2))
    trigrams = list(get_ngrams(all_tokens, 3))
    bigram_counter = Counter(bigrams)
    trigram_counter = Counter(trigrams)
    repeated_bigrams = sum(1 for c in bigram_counter.values() if c > 1)
    repeated_trigrams = sum(1 for c in trigram_counter.values() if c > 1)

    # Readability
    full_text = " ".join(sentences)
    flesch = textstat.flesch_reading_ease(full_text)
    fk_grade = textstat.flesch_kincaid_grade(full_text)
    fog = textstat.gunning_fog(full_text)


    return {
        "avg_sentence_length_words": sum(word_counts) / num_sentences,
        "avg_sentence_length_chars": sum(char_counts) / num_sentences,
        "avg_word_length": sum(word_lengths) / len(word_lengths),
        "type_token_ratio": len(unique_tokens) / token_count,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fk_grade,
        "gunning_fog_index": fog,
        "repeated_bigrams": repeated_bigrams / len(bigrams) if bigrams else 0,
        "repeated_trigrams": repeated_trigrams / len(trigrams) if trigrams else 0,
        "top_words": Counter(all_tokens).most_common(20),
        "top_words_no_stopwords": Counter(filtered_tokens).most_common(20),
        "pos_distribution": normalized_pos
    }

# Analyze both datasets
human_stats = analyze_sentences(human_sentences)
model_stats = analyze_sentences(model_sentences)

# Comparison Table
comparison_df = pd.DataFrame({
    "Feature": [
        "Avg. Sentence Length (words)",
        "Avg. Sentence Length (chars)",
        "Avg. Word Length",
        "Type-Token Ratio",
        "Flesch Reading Ease",
        "Flesch-Kincaid Grade Level",
        "Gunning Fog Index",
        "Repeated Bigrams (ratio)",
        "Repeated Trigrams (ratio)"
    ],
    "Human": [
        human_stats["avg_sentence_length_words"],
        human_stats["avg_sentence_length_chars"],
        human_stats["avg_word_length"],
        human_stats["type_token_ratio"],
        human_stats["flesch_reading_ease"],
        human_stats["flesch_kincaid_grade"],
        human_stats["gunning_fog_index"],
        human_stats["repeated_bigrams"],
        human_stats["repeated_trigrams"]
    ],
    "Model": [
        model_stats["avg_sentence_length_words"],
        model_stats["avg_sentence_length_chars"],
        model_stats["avg_word_length"],
        model_stats["type_token_ratio"],
        model_stats["flesch_reading_ease"],
        model_stats["flesch_kincaid_grade"],
        model_stats["gunning_fog_index"],
        model_stats["repeated_bigrams"],
        model_stats["repeated_trigrams"]
    ]
})

def normalize_global(df):
    """
    Apply global min-max normalization across both Human and Model columns.

    Args:
        df (pandas.DataFrame): DataFrame with numeric columns 'Human' and 'Model'.

    Returns:
        pandas.DataFrame: Copy of the input with two added columns:
            - 'Human_norm': normalized 'Human' values in [0, 1]
            - 'Model_norm': normalized 'Model' values in [0, 1]

    Raises:
        ValueError: If the global range is zero (all values equal), normalization is undefined.
    """
    norm_df = df.copy()
    all_values = df[['Human', 'Model']].values.flatten()
    global_min = all_values.min()
    global_max = all_values.max()

    norm_df['Human_norm'] = (df['Human'] - global_min) / (global_max - global_min)
    norm_df['Model_norm'] = (df['Model'] - global_min) / (global_max - global_min)
    return norm_df

fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

comparison_df_norm = normalize_global(comparison_df)
print("\n=== Comparison Table ===")
print(comparison_df_norm.round(3))



# Bar chart: Raw
# def plot_comparison(df):
#     x = range(len(df))
#     width = 0.35
#     plt.figure(figsize=(10, 6))
#     plt.bar(x, df['Human'], width=width, label='Human')
#     plt.bar([i + width for i in x], df['Model'], width=width, label='Model')
#     plt.xticks([i + width/2 for i in x], df['Feature'], rotation=45, ha='right')
#     plt.title("Global Comparison of Linguistic Features")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def plot_normalized_comparison(ax, df, title):
    """
    Plot normalized Human vs. Model values for a set of features on a provided Axes.

    Args:
        ax (matplotlib.axes.Axes): Target axes to draw the plot on.
        df (pandas.DataFrame): DataFrame with:
            - 'Feature' (str), 'Human_norm' (float), 'Model_norm' (float)
        title (str): Title for the subplot.

    Returns:
        None: Draws bars on the provided axes.
    """
    x = range(len(df))
    width = 0.35
    ax.bar(x, df['Human_norm'], width=width, label='Human')
    ax.bar([i + width for i in x], df['Model_norm'], width=width, label='Model')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(df['Feature'], rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel("Normalized Value")
    ax.legend()

#
# def plot_spider(df):
#     norm_df = normalize_global(df)
#
#     categories = norm_df["Feature"].tolist()
#     human_values = norm_df["Human_norm"].tolist()
#     model_values = norm_df["Model_norm"].tolist()
#
#     angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
#     human_values += [human_values[0]]
#     model_values += [model_values[0]]
#     angles += [angles[0]]
#     categories += [categories[0]]
#
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
#     ax.plot(angles, human_values, label='Human', marker='o')
#     ax.fill(angles, human_values, alpha=0.25)
#     ax.plot(angles, model_values, label='Model', marker='o')
#     ax.fill(angles, model_values, alpha=0.25)
#
#     ax.set_title("Radar Plot: Global Min-Max Normalized Features", size=14)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(categories[:-1], fontsize=10)
#     ax.set_yticklabels([])
#     ax.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()


def plot_pos_distribution(ax, human_pos, model_pos):
    """
    Plot a side-by-side POS tag distribution for human vs. model on a provided Axes.

    Args:
        ax (matplotlib.axes.Axes): Target axes to draw the plot on.
        human_pos (dict[str, float]): Normalized POS distribution for human text.
        model_pos (dict[str, float]): Normalized POS distribution for model text.

    Returns:
        None: Draws bars on the provided axes.

    Notes:
        - Assumes keys in `human_pos` are the POS tag universe; missing keys in
          `model_pos` are treated as zero.
    """
    labels = list(human_pos.keys())
    x = range(len(labels))
    width = 0.35

    human_vals = [human_pos[k] for k in labels]
    model_vals = [model_pos.get(k, 0) for k in labels]

    ax.bar(x, human_vals, width=width, label='Human')
    ax.bar([i + width for i in x], model_vals, width=width, label='Model')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title("POS Distribution")
    ax.legend()


# 1st plot
plot_normalized_comparison(axs[0], df_plot, "YAML Metrics")
# 2nd plot (comparison_df needs to be defined)
plot_normalized_comparison(axs[1], comparison_df_norm, "Statistics of Text")
# 3rd plot
plot_pos_distribution(axs[2], human_stats["pos_distribution"], model_stats["pos_distribution"])

plt.tight_layout()
plt.show()

# # Run plots
# plot_normalized_comparison(df_plot)
# # plot_comparison(comparison_df)
# plot_normalized_comparison(comparison_df)
# # plot_spider(comparison_df)
# plot_pos_distribution(human_stats["pos_distribution"], model_stats["pos_distribution"])

# Top words printout
print("\nTop 20 Words - Human:\n", human_stats["top_words"])
print("\nTop 20 Words - Model:\n", model_stats["top_words"])
print("\nTop 20 Words - Human w/o stopwords:\n", human_stats["top_words_no_stopwords"])
print("\nTop 20 Words - Model w/o stopwords:\n", model_stats["top_words_no_stopwords"])
