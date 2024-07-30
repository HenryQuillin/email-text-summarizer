import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def load_nlp_model():
    return spacy.load("en_core_web_sm")

def remove_introduction(text):
    comma_index = text.find(',')
    if comma_index != -1:
        return text[comma_index + 1:].lstrip()
    else:
        return text

def compute_word_frequencies(doc):
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1
    return word_frequencies

def normalize_frequencies(word_frequencies):
    max_frequency = max(word_frequencies.values())
    return {word: freq / max_frequency for word, freq in word_frequencies.items()}

def compute_sentence_scores(doc, normalized_freq):
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in normalized_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + normalized_freq[word.text]
    return sentence_scores

def get_top_sentences(sentence_scores, n_sentences):
    summarized_sentences = nlargest(n_sentences, sentence_scores, key=sentence_scores.get)
    return sorted(summarized_sentences, key=lambda s: s.start)

def summarize_email(text, n_sentences=2):
    text = remove_introduction(text)

    nlp = load_nlp_model()
    doc = nlp(text)

    word_frequencies = compute_word_frequencies(doc)
    normalized_freq = normalize_frequencies(word_frequencies)
    sentence_scores = compute_sentence_scores(doc, normalized_freq)
    top_sentences = get_top_sentences(sentence_scores, n_sentences)

    summary = ' '.join([sent.text for sent in top_sentences])
    return summary


# Example usage
if __name__ == "__main__":
    email_text = """
    There are many techniques available to generate extractive summarization to keep it simple, I will be using an unsupervised learning approach to find the sentences similarity and rank them. Summarization can be defined as a task of producing a concise and fluent summary while preserving key information and overall meaning. One benefit of this will be, you don't need to train and build a model prior start using it for your project. It's good to understand Cosine similarity to make the best use of the code you are going to see. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. Its measures cosine of the angle between vectors. The angle will be 0 if sentences are similar.
    """
    print(summarize_email(email_text))