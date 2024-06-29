import json
import spacy
from html import escape
from tqdm import tqdm
import statistics

# Global variable to set language
LANGUAGE = 'english'  # 'english' or 'dutch'

# Load the SpaCy model based on the language
if LANGUAGE == 'english':
    nlp = spacy.load("en_core_web_sm")
    input_file_path = 'uk_tbcov_shuffled_clean_3M.jsonl'
    rd_output_file_path = 'final_rd_uk_tweets.txt'
    pq_output_file_path = 'final_pq_uk_tweets.txt'
    rejected_output_file_path = 'final_rejected_sentences_en.txt'
    wh_words = {'what', 'who', 'why', 'how', 'where', 'when', 'which', 'whose', 'whom', 'wtf', 'WTF'}
    tag_questions = {'does he', 'does she', 'does it'}
elif LANGUAGE == 'dutch':
    nlp = spacy.load("nl_core_news_sm")
    input_file_path = 'dutch_tbcov_shuffled_clean.jsonl'
    rd_output_file_path = 'final_rd_nl_tweets.txt'
    pq_output_file_path = 'final_pq_nl_tweets.txt'
    rejected_output_file_path = 'final_rejected_sentences_nl.txt'
    wh_words = {'wat', 'hoe', 'waarom', 'wanneer', 'wie', 'welke', 'welk', 'waar', 'waarheen', 'hoeveel', 'hoezo', 'wtf', 'WTF22'}
    tag_questions = {'toch'}

def sentence_tokenizer(text):
    """Tokenizes the text into sentences."""
    sentences = []
    current_sentence = ''
    for char in text:
        current_sentence += char
        if char in ['.', ':', '?', '!', '-']:
            sentences.append(current_sentence.strip())
            current_sentence = ''
    if current_sentence:
        sentences.append(current_sentence.strip())
    return sentences

def get_main_clause_subject(doc):
    """Finds the main clause subject."""
    for token in doc:
        if token.dep_ == 'nsubj':
            return token
    return None

def get_main_clause_auxiliary(doc):
    """Finds the main clause auxiliary verb and ensures it is finite."""
    for token in doc:
        if token.dep_ == 'aux' and token.head.dep_ == 'ROOT' and token.morph.get('VerbForm') == ['Fin']:
            return token
    return None

def get_main_clause_finite_verb(doc):
    """Finds the main clause finite verb."""
    for token in doc:
        if token.dep_ == 'ROOT' and token.pos_ == 'VERB' and token.morph.get('VerbForm') == ['Fin']:
            return token
    return None

def rd_or_pq(sentence):
    """Checks whether a sentence is a rising declarative (RD) or a polar question (PQ)."""
    doc = nlp(sentence)
    if any(tag in sentence.lower() for tag in tag_questions):
        with open(rejected_output_file_path, 'a') as rejected_file:
            rejected_file.write(f"Rejected due to tag question: {sentence.strip()}\n")
        return None, None

    if len(doc) < 15 and sentence.strip().endswith('?'):
        if any(token.text.lower() in wh_words for token in doc):
            return None, None

        subject = get_main_clause_subject(doc)
        auxiliary = get_main_clause_auxiliary(doc)
        finite_verb = get_main_clause_finite_verb(doc)

        main_verb = auxiliary if auxiliary else finite_verb

        if subject and main_verb:
            if subject.i < main_verb.i:
                return 'rd', sentence.strip()
            elif main_verb.i < subject.i:
                return 'pq', sentence.strip()
        else:
            with open(rejected_output_file_path, 'a') as rejected_file:
                rejected_file.write(f"Rejected due to no subject or main verb: {sentence.strip()}\n")
    else:
        with open(rejected_output_file_path, 'a') as rejected_file:
            rejected_file.write(f"Rejected due to length or non-question: {sentence.strip()}\n")
    return None, None

def format_output(tweet_counter, sentence, sentence_type, full_tweet):
    """Formats the output with tweet counter, sentence type, and the entire tweet."""
    return f"Tweet {tweet_counter}: [{sentence_type.upper()}] {escape(sentence)}\nFull tweet: {escape(full_tweet)}\n\n"

def analyze_dataset_statistics(sentences):
    """Analyze dataset statistics."""
    sentence_lengths = [len(nlp(sentence)) for sentence in sentences]
    word_count = sum(sentence_lengths)
    return {
        'total_sentences': len(sentences),
        'total_words': word_count,
        'avg_sentence_length': statistics.mean(sentence_lengths) if sentence_lengths else 0
    }

def print_statistics(statistics, description):
    """Print statistics in a formatted manner."""
    print(f"{description}:")
    for key, value in statistics.items():
        print(f"{key}: {value}")
    print()

# Process tweets and analyze RD and PQ sentences
with open(input_file_path, 'r') as input_file, open(rd_output_file_path, 'w') as rd_output_file, open(pq_output_file_path, 'w') as pq_output_file:
    tweet_counter = 0
    rd_count = 0
    pq_count = 0
    total_sentences = 0
    rd_sentences = []
    pq_sentences = []
    all_sentences = []
    question_sentences = []

    # Use tqdm to track progress of main loop
    for line in tqdm(input_file, desc="Processing tweets", unit="tweets"):
        tweet_counter += 1
        tweet = json.loads(line)
        full_text_cleaned = tweet.get('full_text_cleaned', '')
        sentences = sentence_tokenizer(full_text_cleaned)
        for sentence in sentences:
            total_sentences += 1
            all_sentences.append(sentence)
            if sentence.strip().endswith('?'):
                question_sentences.append(sentence)
            sentence_type, highlighted_sentence = rd_or_pq(sentence)
            if sentence_type:
                if sentence_type == 'rd':
                    rd_count += 1
                    rd_sentences.append(sentence)
                    rd_output_file.write(format_output(tweet_counter, highlighted_sentence, sentence_type, full_text_cleaned))
                elif sentence_type == 'pq':
                    pq_count += 1
                    pq_sentences.append(sentence)
                    pq_output_file.write(format_output(tweet_counter, highlighted_sentence, sentence_type, full_text_cleaned))

    # Calculate statistics
    dataset_stats = analyze_dataset_statistics(all_sentences)
    question_stats = analyze_dataset_statistics(question_sentences)
    rd_stats = analyze_dataset_statistics(rd_sentences) if rd_sentences else {}
    pq_stats = analyze_dataset_statistics(pq_sentences) if pq_sentences else {}

    # Print statistics
    print(f"Total sentences analyzed: {total_sentences}")
    print(f"Total RDs found: {rd_count}")
    print(f"Total PQs found: {pq_count}")

    print_statistics(dataset_stats, "Dataset Statistics")
    print_statistics(question_stats, "Question Sentences Statistics")
    if rd_stats:
        print_statistics(rd_stats, "RD Statistics")
    if pq_stats:
        print_statistics(pq_stats, "PQ Statistics")
