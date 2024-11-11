import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = '''Samsung was founded by Lee Byung-chul in 1938 as a trading company. Over the next three decades, the group diversified into
 areas including food processing, textiles, insurance, securities, and retail. Samsung entered the electronics industry in the late 1960s 
 and the construction and shipbuilding industries in the mid-1970s; these areas would drive its subsequent growth. Following Lee's death in 
 1987, Samsung was separated into five business groups – Samsung Group, Shinsegae Group, CJ Group, Hansol Group, and JoongAng Group.
Samsung industrial affiliates include Samsung Electronics, Samsung Heavy Industries, Samsung Engineering and Samsung C&T Corporation. Other
 subsidiaries include Samsung Life Insurance and Cheil Worldwide. Notable Samsung industrial affiliates include Samsung Electronics 
 
 Samsung Heavy Industries (the world's second largest shipbuilder measured by 2010 revenues),[7] and Samsung Engineering and Samsung 
 C&T Corporation (respectively the world's 13th and 36th largest construction companies).[8] Other notable subsidiaries include Samsung 
 Life Insurance (the world's 14th largest life insurance company),[9] Samsung Everland (operator of Everland Resort, the oldest theme park 
 in South Korea)[10] and Cheil Worldwide (the world's 15th largest advertising agency, as measured by 2012 revenues).[11][12]'''

def summarizer(rawdocs):
    stopwords = list(STOP_WORDS)
    #print(stopwords)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)
    #print(doc)

    tokens = [token.text for token in doc]
    #print(tokens)
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] +=1

    #print(word_freq)

    max_freq = max(word_freq.values())
    #print(max_freq)

    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq

    #print(word_freq)

    sent_tokens = [sent for sent in doc.sents]
    #print(sent_tokens)

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    #print(sent_scores)

    select_len = int(len(sent_tokens ) * 0.3)
    #print(select_len)

    summary = nlargest(select_len, sent_scores, key = sent_scores.get)
    #print(summary)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    #print(text)
    #print(summary)
    #print("Length of original text ", len(text.split(' ')))
    #print("Length of summary text ", len(summary.split(' ')))

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))



