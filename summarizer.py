# File: summarizer.py

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_article(article_text, num_sentences=3):
    parser = PlaintextParser.from_string(article_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    summary_text = " ".join([str(sentence) for sentence in summary])
    return summary_text
