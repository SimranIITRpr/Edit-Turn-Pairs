#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:00:01 2020

@author: simransetia
"""
from os import listdir
import re
from os.path import isfile, join 
mypath='/Users/simransetia/Downloads/ETPgold_v1/corresponding/'
plist = [f for f in listdir(mypath) if isfile(join(mypath, f))]



import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

selfClosingTags = ('br', 'hr', 'nobr', 'ref', 'references', 'nowiki','strong','strike')

placeholder_tags = {'math': 'formula', 'code': 'codice'}
# Match selfClosing HTML tags
selfClosing_tag_patterns = [
    re.compile(r'<\s*%s\b[^>]*/\s*>' % tag, re.DOTALL | re.IGNORECASE) for tag in selfClosingTags
    ]

# Match HTML placeholder tags
placeholder_tag_patterns = [
    (re.compile(r'<\s*%s(\s*| [^>]+?)>.*?<\s*/\s*%s\s*>' % (tag, tag), re.DOTALL | re.IGNORECASE),
     repl) for tag, repl in placeholder_tags.items()
    ]
def dropSpans(spans, text):
    """
    Drop from text the blocks identified in :param spans:, possibly nested.
    """
    spans.sort()
    res = ''
    offset = 0
    for s, e in spans:
        if offset <= s:         
            if offset < s:
                res += text[offset:s]
            offset = e
    res += text[offset:]
    return res
def clean(text):

        spans = []

        # Drop self-closing tags
        for pattern in selfClosing_tag_patterns:
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end()))
        text = dropSpans(spans, text)

        # Expand placeholders
        for pattern, placeholder in placeholder_tag_patterns:
            index = 1
            for match in pattern.finditer(text):
                text = text.replace(match.group(), '%s_%d' % (placeholder, index))
                index += 1

        text = text.replace('<<', '«').replace('>>', '»')


        # Cleanup text
        text = text.replace('\t', ' ')
        text = re.sub(' (,:\.\)\]»)', r'\1', text)
        text = re.sub('(\[\(«) ', r'\1', text)
        text = re.sub(r'\n\W+?\n', '\n', text, flags=re.U)  # lines with only punctuations
        text = text.replace(',,', ',').replace(',.', '.')
        text=text.replace('&lt',"")
        text=text.replace('&gt',"")
        text=text.replace('ref',"")
        text=text.replace('/ref',"")
        text=text.replace('<strong>',"")
        text=text.replace('</strong>',"")
        text=text.replace('<strike>',"")
        text=text.replace('</strike>',"")
        match=re.search(r'{{.*}}',text)
        if match!=None:
            text=text.replace(match.group(0),"")
        #print(match.group(0))
        
        return text

text1=''
text11=''
Edit_text=''
Turn_topic_text=''
for p in plist:
    with open (mypath+p) as f:
        x=f.readlines()
    for each in x:
        if '<edit_comment>' in each:
            Edit_comment=each.replace('<edit_comment>',"")
            Edit_comment=Edit_comment.replace('</edit_comment>',"")
        if '<edit_text>' in each:
            Edit_text=each.replace('<edit_text>',"")
            Edit_text=Edit_text.replace('</edit_text>',"")
        if '<turn_topicname>' in each:
            Turn_topic=each.replace('<turn_topicname>',"")
            Turn_topic=Turn_topic.replace('</turn_topicname>',"")
        if '<turn_topictext>' in each:
            Turn_topic_text=each.replace('<turn_topictext>',"")
            Turn_topic_text=Turn_topic_text.replace('</turn_topictext>',"")
        if '<turn_text>' in each:
            Turn_text=each.replace('<turn_text>',"")
            Turn_text=Turn_text.replace('</turn_text>',"")
        text=clean(Edit_text)
        text1=clean(Turn_topic_text)
    text1=text1+text
    text11=text11+text1
token = nltk.word_tokenize(text1)
bigrams = ngrams(token,2)
trigrams = ngrams(token,3)

unigrams_edittext=Counter(token)
unigrams_edittext=dict(unigrams_edittext)
unigrams_edittext_top100=sorted(unigrams_edittext)[:100]


bigrams_edittext=Counter(bigrams)
bigrams_edittext=dict(bigrams_edittext)
bigrams_edittext_top100=sorted(bigrams_edittext)[:100]

trigrams_edittext=Counter(trigrams)
trigrams_edittext=dict(trigrams_edittext)
trigrams_edittext_top100=sorted(trigrams_edittext)[:100]

token1 = nltk.word_tokenize(text11)
bigrams1 = ngrams(token,2)
trigrams1 = ngrams(token,3)

unigrams_turntext=Counter(token1)
unigrams_turntext=dict(unigrams_turntext)
unigrams_turntext_top100=sorted(unigrams_turntext)[:100]

bigrams_turntext=Counter(bigrams1)
bigrams_turntext=dict(bigrams_turntext)
bigrams_turntext_top100=sorted(bigrams_turntext)[:100]

trigrams_turntext=Counter(trigrams1)
trigrams_turntext=dict(trigrams_turntext)
trigrams_turntext_top100=sorted(trigrams_turntext)[:100]      
