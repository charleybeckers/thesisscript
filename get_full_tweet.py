import json
from tqdm import tqdm

# Define file paths based on language
LANGUAGE = 'dutch'  # 'english' or 'dutch'

if LANGUAGE == 'english':
    input_file_path = 'uk_tbcov_shuffled_clean_3M.jsonl'
    rd_output_file_path = 'final_rd_uk_tweets.txt'
    updated_rd_output_file_path = 'final_updated_rd_uk_tweets.txt'
elif LANGUAGE == 'dutch':
    input_file_path = 'dutch_tbcov_shuffled_clean.jsonl'
    pq_output_file_path = 'final_rd_nl_tweets.txt'
    updated_pq_output_file_path = 'final_updated_rd_nl_tweets.txt'

def load_output_file(file_path):
    """Load the sentences from the output file."""
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                tweet_number, tweet_text = line.strip().split(": ", 1)
                tweet_number = int(tweet_number.split()[-1])  # Extract the number after 'Tweet'
                sentences.append((tweet_number, tweet_text))
            except ValueError as e:
                print(f"Skipping line due to error: {e}. Line: {line}")
    return sentences

def find_full_tweet(tweet_number, input_file_path):
    """Find the full tweet text by tweet number from the input file."""
    with open(input_file_path, 'r') as input_file:
        for current_number, line in enumerate(input_file, start=1):
            if current_number == tweet_number:
                tweet = json.loads(line)
                return tweet.get('full_text_cleaned', '')
    return None

# Load the RD sentences
rd_sentences = load_output_file(pq_output_file_path)

# Update the RD output file with full tweet text
with open(updated_pq_output_file_path, 'w') as updated_rd_file:
    for tweet_number, rd_text in tqdm(rd_sentences, desc="Updating PQ output file"):
        full_tweet = find_full_tweet(tweet_number, input_file_path)
        if full_tweet:
            updated_rd_file.write(f"{rd_text}\n{tweet_number}: {full_tweet}\n")
