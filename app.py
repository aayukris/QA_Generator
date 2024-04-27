from flask import Flask, render_template, request
import torch
import requests
import re
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
cleaned_answers = []
cleaned_answers_f = []
# Azure necessary things
azure_endpoint = ""
url = ""
headers = {
    "Content-type": "application/json",
    "Ocp-apim-subscription-key": ""
}

generated_qna = []
distractors_str = []
question = ""
answer = ""
# qa generation

t5_weights_path = 't5-base-Race-QA-Generation-version0-step30000.pt'
t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
t5_tokenizer.add_special_tokens({"sep_token": "<sep>"})
t5_model.load_state_dict(torch.load(t5_weights_path, map_location=torch.device('cpu')))

# distractor generation
distractor_weights_path = 'distractor'
distractor_config_path = 'distractor/config.json'
distractor_model = AutoModelForSeq2SeqLM.from_pretrained(distractor_weights_path, config=distractor_config_path)
distractor_tokenizer = AutoTokenizer.from_pretrained(distractor_weights_path)
distractor_tokenizer.add_special_tokens({"sep_token": "<sep>"})

# Use regex to replace special characters between words with a space
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

# Use regex to remove URLs and image tags
def remove_urls_and_special_characters(text):
    text_without_urls = re.sub(r'https?://\S+', '', text)
    text_without_img_tags = re.sub(r'!\[Img\]', '', text_without_urls)
    return remove_special_characters(text_without_img_tags)

# insert URL
def put_url(url):
    body = [
        {
            "op": "add",
            "value": {
                "displayName": "source1",
                "sourceUri": url,
                "sourceKind": "url",
                "source": url
            }
        }
    ]

    try:
        # Post the URL to the portal
        response_post = requests.patch(azure_endpoint, headers=headers, json=body)
        if response_post.status_code == 202:
            print("INSERT request successful.")
            print(url)

        else:
            print(f"INSERT request failed with status code: {response_post.status_code}")
            print("Response content:")
            print(response_post.json())
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# delete url
def del_url(url):
    body = [
        {
            "op": "delete",
            "value": {
                "displayName": "source1",
                "sourceUri": url,
                "sourceKind": "url",
                "source": url
            }
        }
    ]

    try:
        # Delete the URL to the portal
        response_post = requests.patch(azure_endpoint, headers=headers, json=body)
        if response_post.status_code == 202:
            print("DELETE request successful.")
            print(url)
            time.sleep(30)

        else:
            print(f"DELETE request failed with status code: {response_post.status_code}")
            print("Response content:")
            print(response_post.json())
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# get answer and generation
def fetch_answers():
    global cleaned_answers
    global distractors_str
    global cleaned_answers_f
    global generated_qna
    
    # Reset variables
    cleaned_answers = []
    distractors_str = []
    cleaned_answers_f = []
    generated_qna = []
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for entry in data.get('value', []):
                answer = entry.get('answer')
                if answer is not None:
                    answer_cleaned = remove_urls_and_special_characters(answer)
                    cleaned_answers.append(answer_cleaned)
            combined_answers = []
            context = ' '.join(cleaned_answers)
            for i in range(0, len(cleaned_answers), 3):
                group = cleaned_answers[i:i + 3]
                combined_text = ' '.join(group)
                combined_answers.append(combined_text)
            non_empty_combined_answers = [answer for answer in combined_answers if answer.strip()]
            non_empty_combined_answers = [phrase for phrase in non_empty_combined_answers if len(phrase.split()) > 50]
            cleaned_answers_f.extend(non_empty_combined_answers)

            # Generate QnA for each cleaned answer
            generated_qna = []
            distractors_dict = {}  # Keep track of distractors for each question
            for cleaned_answer in cleaned_answers_f:
                input_ids_t5 = t5_tokenizer.encode(cleaned_answer, return_tensors='pt')
                output_ids_t5 = t5_model.generate(input_ids_t5, max_length=100)
                question_answer = t5_tokenizer.decode(output_ids_t5[0], skip_special_tokens=False)
                question_answer = question_answer.replace(t5_tokenizer.pad_token, "").replace(t5_tokenizer.eos_token, "")
                question, answer = question_answer.split(t5_tokenizer.sep_token)
                generated_qna.append({'question': question, 'answer': answer})
                
                # Generate distractors
                input_text_distractor = " ".join([question, distractor_tokenizer.sep_token, answer, distractor_tokenizer.sep_token,context])
                inputs_distractor = distractor_tokenizer(input_text_distractor, return_tensors="pt")
                outputs_distractor = distractor_model.generate(**inputs_distractor, max_new_tokens=128)
                distractors_list = distractor_tokenizer.decode(outputs_distractor[0], skip_special_tokens=False)
                distractors_list = distractors_list.replace(distractor_tokenizer.pad_token, "").replace(distractor_tokenizer.eos_token, "")
                distractors_list = [y.strip() for y in distractors_list.split(distractor_tokenizer.sep_token)]
                print(distractors_list)
                # Filter out duplicate distractors for each question
                unique_distractors = list(set(distractors_list) - set([answer]))
                distractors_dict[question] = unique_distractors
                print(unique_distractors)
            
            # Create the final distractors_str list
            for question_answer in generated_qna:
                question_text = question_answer['question']
                distractors_str.append({'qna': {'question': question_text, 'answer': question_answer['answer']},
                                       'distractors': distractors_dict.get(question_text, [])})
        else:
            print(f"GET request failed with status code: {response.status_code}")
            print("Response content:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url')
        put_url(url)
        time.sleep(60)
        fetch_answers()
        del_url(url)
    return render_template('index.html', cleaned_answers=cleaned_answers_f, generated_qna=generated_qna,
                           distractors=distractors_str)


if __name__ == '__main__':
    app.run(debug=True)
