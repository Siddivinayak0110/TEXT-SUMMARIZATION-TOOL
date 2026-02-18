from flask import Flask, request, render_template
import nltk
import heapq
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import PyPDF2

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

# Summarizer Function
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
        return "Not enough content to summarize."

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


@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""

    if request.method == "POST":

        # If PDF uploaded
        if "pdf_file" in request.files:
            file = request.files["pdf_file"]

            if file.filename != "":
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page in pdf_reader.pages:
                    text += page.extract_text()

                summary = summarize(text)

        # If text pasted
        else:
            article = request.form.get("article")
            summary = summarize(article)

    return render_template("index.html", summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
