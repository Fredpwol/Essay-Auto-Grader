import spacy

import numpy as np
from flask import Flask, request, current_app
from tf_socre_grader import text_distance
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language

from collections import defaultdict

THRESHOLD = 0.4

app = Flask(__name__)
nlp = spacy.load('en_core_web_md')

# Define custom pipeline components
websites = ['www.', 'http', '.com', '.org', '.net', '.gov']
stopwords = STOP_WORDS


@Language.component('remove_punctuations')
def remove_punctuations(doc):
    # Remove punctuation tokens from a Doc object
    new_tokens = []
    for token in doc:
        if not token.is_punct:
            new_tokens.append(token)
    return spacy.tokens.Doc(nlp.vocab,
                            words=[token.text for token in new_tokens])


@Language.component("remove_websites")
def remove_websites(doc):
    # Remove tokens that contain website URLs from a Doc object
    new_tokens = []
    for token in doc:
        if not any(website in token.text for website in websites):
            new_tokens.append(token)
    return spacy.tokens.Doc(nlp.vocab,
                            words=[token.text for token in new_tokens])


@Language.component("remove_stopwords")
def remove_stopwords(doc):
    # Remove stop words from a Doc object
    new_tokens = []
    for token in doc:
        if not token.is_stop:
            new_tokens.append(token)
    return spacy.tokens.Doc(nlp.vocab,
                            words=[token.text for token in new_tokens])


# Add the custom pipeline components to the pipeline
nlp.add_pipe('remove_punctuations', first=True)
nlp.add_pipe('remove_websites', after='remove_punctuations')
nlp.add_pipe('remove_stopwords', last=True)

# Process a text string with the custom pipeline
# doc = nlp("This is an example sentence. It contains some punctuations! And also a URL: www.example.com")
# print([token.text for token in doc])


def compare_words(a, b):  # Can Improve this by using a sysnet
    if a.text.lower() == b.text.lower():
        return 1.0

    return a.similarity(b)


def score_answer(data):
    if "correct_answer" not in data or "user_answer" not in data:
        return {
            "status": "error",
            "message": "correct_answer and user_answer are required fields"
        }

    correct = data.get("correct_answer").strip().lower()
    answer = data.get("user_answer").strip().lower()
    strict_mode = data.get("strict", False)

    if not bool(correct) and bool(answer):
        return {
            "status": "error",
            "message": "input can not be an empty or null"
        }

    kw = data.get("keywords", dict())

    if not (0 <= sum(kw.values()) <= 1):
        return {
            "status": "error",
            "message": "Keyword penalty can not be below 0 or go above 1"
        }, 400

    score = text_distance(answer, correct) * 100

    if not strict_mode and score > 10:
        range_ = np.linspace(0, 100, 4)
        for thres in range_:
            if score < thres:
                score = thres

    score = min(100.0, score + 10.0)

    penality_score = 0

    answer_tokens = [tok for tok in nlp(answer)]

    res = defaultdict(list)

    for key, weight in kw.items():
        key_tok = nlp(key)
        for token in answer_tokens:
            sim = compare_words(key_tok, token)
            print("smilarity score", sim, key_tok.text, token)
            if sim > THRESHOLD:
                res[key].append(token.text)
        if len(res[key]) == 0:
            res[key].append("No Match Found!")
            penality_score += weight

    score -= score * penality_score

    response = {
        "status": "success",
        "data": {
            "score": score,
            "penalty_loss": penality_score,
            "matches": dict(res)
        }
    }

    return response


@app.route('/get-score', methods=["POST"])
def get_score():
    """
    Gets the score of a question by comparing the actucal answer and the submitted answer, with an optional keyword checker.
    POST 
        {
            "correct_answer": string,
            "user_answer": string,
            "strict": boolean,
            "keywords": {
                "name": weight: int
            }

        }
    """

    try:
        data = request.get_json()

        response = score_answer(data)

        if response.get("status", "") == "error":
            return response, 400
        return response, 200

    except Exception as ex:
        current_app.logger.critical("Internal Error:", exc_info=1)
        return {
            "status": "error",
            "message": "Critical Internal Server Error",
            "error": str(ex)
        }, 500
    

@app.route('/get-score-bulk', methods=["POST"])
def get_score_bulk():
    """
    Gets the score of a question by comparing the actucal answer and the submitted answer, with an optional keyword checker.
    POST 
        {"answers": [
            {
            "correct_answer": string,
            "user_answer": string,
            "keywords": {
                "name": weight: int
            }

        }],
        "strict": booleeean
        }
    """

    try:
        data = request.get_json()

        if 'answers' not in data or type(data.get("answers")) != list:
            return {
                "status": "error",
                "message": "answer is a required field and must be an array type"
            }

        total_score = 0
        total_penalty = 0
        n = len( data['answers'])
        scores = []
        for answer in data['answers']:
            answer['strict'] = data['strict']
            response = score_answer(answer)

            if response.get("status", "") == "error":
                return response, 400
            
            score = response['data']['score']
            total_score += score
            total_penalty += response['data']['penalty_loss']
            scores.append(score)

        response = {
            "status": "success",
            "data": {
                "score": total_score / n,
                "penalty_loss": total_penalty,
                "scores": scores
            }
        }
        return response, 200

    except Exception as ex:
        current_app.logger.critical("Internal Error:", exc_info=1)
        return {
            "status": "error",
            "message": "Critical Internal Server Error",
            "error": str(ex)
        }, 500



if __name__ == "__main__":
    app.run(debug=True)
