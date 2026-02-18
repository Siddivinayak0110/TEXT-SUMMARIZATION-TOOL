import heapq
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def summarize(text, summary_length=3):

    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))

    freq = {}
    for word in words:
        if word not in stop_words and word.isalnum():
            freq[word] = freq.get(word, 0) + 1

    if not freq:
        return "Not enough content."

    max_freq = max(freq.values())
    for word in freq:
        freq[word] = freq[word] / max_freq

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq[word]

    summary = heapq.nlargest(summary_length, sentence_scores, key=sentence_scores.get)

    return " ".join(summary)
