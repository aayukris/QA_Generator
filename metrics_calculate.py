from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

t5_weights_path = ''
t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
t5_tokenizer.add_special_tokens({"sep_token": "<sep>"})
t5_model.load_state_dict(torch.load(t5_weights_path, map_location=torch.device('cpu')))

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# Load SQuAD dataset
squad = load_dataset("squad")


# Load T5 model and tokenizer

# Extract unique contexts from the first 2000 examples in the SQuAD dataset
# unique_contexts = set(example["entity_pages"]["wiki_context"] for example in triviaqa["train"])
unique_contexts = set(example["entity_pages"]["wiki_context"] if not isinstance(example["entity_pages"]["wiki_context"], list) else ' '.join(example["entity_pages"]["wiki_context"]) for example in triviaqa["train"])

# Function to generate questions for a given context
def generate_question(context):
    inputs = t5_tokenizer(context, return_tensors="pt")
    outputs = t5_model.generate(**inputs, max_length=100)
    question_answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(t5_tokenizer.pad_token, "").replace(t5_tokenizer.eos_token, "")
    # question, answer = question_answer.split(t5_tokenizer.sep_token)
    # return question
    try:
        question, answer = question_answer.split(t5_tokenizer.sep_token)
        return question
    except ValueError:
        print(f"Skipping context due to split error: {context}")
        return None

# Function to calculate cosine similarity between two strings
def calculate_similarity(str1, str2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([str1, str2])
    similarity = cosine_similarity(vectors)
    return similarity[0, 1]

# Initialize lists to store generated questions and most similar questions
generated_questions = []
similar_squad_questions = []
bert_scores = []
rouge_scores = []

# Instantiate ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# Iterate through each unique context and generate questions
i=0
counter = 0
for context in unique_contexts:
    i+=1
    # Generate question for the context
    generated_question = generate_question(context)

    if generated_question is None:
        continue

    # Filter SQuAD dataset for the same context
    similar_examples = [example for example in triviaqa["train"] if example["entity_pages"]["wiki_context"] == context]

    # Find the most similar question from the SQuAD dataset
    most_similar_example = max(similar_examples, key=lambda x: calculate_similarity(x["question"], generated_question))

    # Add generated question and most similar SQuAD question to lists
    generated_questions.append(generated_question)
    similar_squad_questions.append(most_similar_example["question"])
    _, _, bert_f1 = bert_score([generated_question], [most_similar_example["question"]], lang="en", model_type="bert-base-uncased")
    bert_scores.append(bert_f1.item())

    # Calculate ROUGE score
    rouge_score = scorer.score(generated_question, most_similar_example["question"])['rougeL'].fmeasure
    rouge_scores.append(rouge_score)

    print(i)
    counter += 1

    # If counter reaches 100, calculate metrics and reset lists
    if counter == 200:
        # Calculate and print metrics
        # You can calculate BLEU score here if needed
        print("Metrics after processing " + str(i) + " contexts:")
        print(f"Average BERT Score: {sum(bert_scores) / len(bert_scores)}")
        print(f"Average ROUGE Score: {sum(rouge_scores) / len(rouge_scores)}\n")
        bleu_1 = corpus_bleu([[question] for question in similar_squad_questions], generated_questions, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu_2 = corpus_bleu([[question] for question in similar_squad_questions], generated_questions, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu_3 = corpus_bleu([[question] for question in similar_squad_questions], generated_questions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method1)
        bleu_4 = corpus_bleu([[question] for question in similar_squad_questions], generated_questions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)

        print(f"BLEU-1 Score: {bleu_1}")
        print(f"BLEU-2 Score: {bleu_2}")
        print(f"BLEU-3 Score: {bleu_3}")
        print(f"BLEU-4 Score: {bleu_4}")
        
        bleu_score = corpus_bleu([[question] for question in similar_squad_questions], generated_questions)
        print(f"BLEU Score: {bleu_score}")
        counter = 0
