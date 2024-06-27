import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


def summarize_email(text, n_sentences=4):
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    # Compute  frequency
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    # Normalize frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    # Copute the score for each sentence
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text]
                else:
                    sentence_scores[sent] += word_frequencies[word.text]

    # Select the n most important sentences
    summarized_sentences = nlargest(n_sentences, sentence_scores, key=sentence_scores.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)

    return summary


email_text = """
The Statue of Liberty, a hollow colossus composed of thinly pounded copper sheets over a steel framework, stands on an island at the entrance to New York Harbor. It was designed by sculptor Frédéric Bartholdi in collaboration with engineer Gustave Eiffel, and was a gift from France on the centenary of American independence in 1876. Its design and construction were recognized at the time as one of the greatest technical achievements of the 19th century and hailed as a bridge between art and engineering. Atop its pedestal (designed by American architect Richard Morris Hunt), the Statue has welcomed millions of immigrants to the United States since it was dedicated in 1886.

The Statue is a masterpiece of colossal statuary, which found renewed expression in the 19th century, after the tradition of those of antiquity, but with intimations of Art Nouveau. Drawing on classical elements and iconography, it expressed modern aspirations. The interior iron framework is a formidable and intricate piece of construction, a harbinger of the future in engineering, architecture, and art, including the extensive use of concrete in the base, the flexible curtain-wall type of construction that supports the skin, and the use of electricity to light the torch. Édouard René de Laboulaye collaborated with Bartholdi for the concept of the Statue to embody international friendship, peace, and progress, and specifically the historical alliance between France and the United States. Its financing by international subscription was also significant. Highly potent symbolic elements of the design include the United States Declaration of Independence, which the Statue holds in her left hand, as well as the broken shackles from which she steps. 

"""
print(summarize_email(email_text))
