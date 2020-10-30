#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:04:42 2020

@author: simransetia
"""
#from uni_bi_tri import unigrams_edittext_top100,bigrams_edittext_top100,trigrams_edittext_top100,unigrams_turntext_top100,bigrams_turntext_top100,trigrams_turntext_top100
from uni_bi_tri import *
import nltk
import csv
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from os import listdir
from os.path import isfile, join 
from textblob import TextBlob
import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument 

from nltk.util import ngrams
import re
rows=[[]]
mypath='/Users/simransetia/Downloads/ETPgold_v1/corresponding_new/'
plist = [f for f in listdir(mypath) if isfile(join(mypath, f))]

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
i=0
t=[53,20,144,0,600,0,87,144,72,96,18,16,0,0,3,936,9,10,12,9,9,1148,1,28,960,648,20,144,0,10,120,13,3,4]
#voc_vec = word2vec.Word2Vec(vocab, min_count=2)
for p in plist:
    print(p)
    row=[]
    with open (mypath+p) as f:
        x=f.readlines()
    for each in x:
        if '<article_title>' in each:
            article_name=each.replace('<article_title>',"")
            article_name=article_name.replace('</article_title>',"")
        if '<edit_user>' in each:
            edit_user=each.replace('<edit_user>',"")
            edit_user=edit_user.replace('</edit_user>',"")
        if '<edit_time>' in each:
            edit_time=each.replace('<edit_time>',"")
            edit_time=edit_time.replace('</edit_time>',"")
            edit_time=edit_time.replace(" CEST","")
            edit_time=edit_time.replace(" CET","")
            edit_time=edit_time.replace("\n","")
        if '<edit_comment>' in each:
            Edit_comment=each.replace('<edit_comment>',"")
            Edit_comment=Edit_comment.replace('</edit_comment>',"")
        if '<edit_text>' in each:
            Edit_text=each.replace('<edit_text>',"")
            Edit_text=Edit_text.replace('</edit_text>',"")
        if '<turn_user>' in each:
            turn_user=each.replace('<turn_user>',"")
            turn_user=turn_user.replace('</turn_user>',"")
        if '<turn_time>' in each:
            turn_time=each.replace('<turn_time>',"")
            turn_time=turn_time.replace('</turn_time>',"")
            turn_time=turn_time.replace(" CEST","")
            turn_time=turn_time.replace(" CET","")
            turn_time=turn_time.replace("\n","")
        if '<turn_topicname>' in each:
            Turn_topic=each.replace('<turn_topicname>',"")
            Turn_topic=Turn_topic.replace('</turn_topicname>',"")
        if '<turn_topictext>' in each:
            Turn_topic_text=each.replace('<turn_topictext>',"")
            Turn_topic_text=Turn_topic_text.replace('</turn_topictext>',"")
        if '<turn_text>' in each:
            Turn_text=each.replace('<turn_text>',"")
            Turn_text=Turn_text.replace('</turn_text>',"")


    j=0

    row.append(article_name)
    j=j+1
    if edit_user==turn_user:
      
        row.append(0)
    else:
        
        row.append(1)
    j=j+1
    #Convert the edit and turn time to date time objects
    et=datetime.datetime.strptime(edit_time, '%A, %B %d, %Y %I:%M:%S %p')
    tt=datetime.datetime.strptime(turn_time, '%A, %B %d, %Y %I:%M:%S %p')
    #Covert the time diffrence to hours
    row.append(str((tt-et).seconds//3600))
    row.append(t[i])
    
    j=j+1
    edit=[]
    edit.append(Edit_comment)
    edit.append (Edit_text)
    turn=[]
    turn.append(Turn_topic_text)
    turn.append(Turn_text)
    tokenized_sent = []
    
    
    
    edit_text_clean=clean(Edit_text)
    edit_comment_clean=clean(Edit_comment)
    turntopic_text_clean=clean(Turn_topic_text)
    turn_text_clean=clean(Turn_text)
    #Similarity between edit text and turn text
    tokenized_sent.append(word_tokenize(Edit_text.lower()))
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
    model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
    test_doc = word_tokenize(Turn_text.lower())
    test_doc_vector = model.infer_vector(test_doc)
    print(model.docvecs.most_similar(positive = [test_doc_vector]))
    s=model.docvecs.most_similar(positive = [test_doc_vector])
    #sheet1.write(i,j,s[0][1])
    row.append(s[0][1])
    j=j+1
    
    tokenized_sent=[]
    #Similarity between clean edit text and clean turn text
    if(edit_text_clean!='\n'):
        tokenized_sent.append(word_tokenize(edit_text_clean.lower()))
        
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
        model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
        test_doc = word_tokenize(turn_text_clean.lower())
        test_doc_vector = model.infer_vector(test_doc)
        print(model.docvecs.most_similar(positive = [test_doc_vector]))
        s=model.docvecs.most_similar(positive = [test_doc_vector])
        
        row.append(s[0][1])
        j=j+1
    else:
        row.append(1)
        
    tokenized_sent=[]
    #Similarity between edit comment and turn text
    if (Edit_comment!='NA' or Edit_comment!=''or Edit_comment!='\\n'):
        tokenized_sent.append(word_tokenize(Edit_comment.lower()))
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
        model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
        test_doc = word_tokenize(Turn_text.lower())
        test_doc_vector = model.infer_vector(test_doc)
        print(model.docvecs.most_similar(positive = [test_doc_vector]))
        s=model.docvecs.most_similar(positive = [test_doc_vector])
       
        row.append(s[0][1])
        j=j+1
    else:
        row.append(1)
    #Similarity between edit comment and clean turn text
    if (Edit_comment!='NA' or Edit_comment!=''or Edit_comment!='\\n'):
        tokenized_sent=[]
        tokenized_sent.append(word_tokenize(edit_comment_clean.lower()))
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
        model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
        test_doc = word_tokenize(turn_text_clean.lower())
        test_doc_vector = model.infer_vector(test_doc)
        print(model.docvecs.most_similar(positive = [test_doc_vector]))
        s=model.docvecs.most_similar(positive = [test_doc_vector])
        
        row.append(s[0][1])
        j=j+1
    else:
        row.append(1)
    #Similarity between edit comment and turn topic
    if (Edit_comment!='NA' or Edit_comment!=''or Edit_comment!='\\n'):
        tokenized_sent=[]
        tokenized_sent.append(word_tokenize(Edit_comment.lower()))
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
        model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
        test_doc = word_tokenize(Turn_topic.lower())
        test_doc_vector = model.infer_vector(test_doc)
        print(model.docvecs.most_similar(positive = [test_doc_vector]))
        s=model.docvecs.most_similar(positive = [test_doc_vector])
        
        row.append(s[0][1])
        j=j+1
    else:
        row.append(1)
    #Similarity between edit text and turn topic
    tokenized_sent=[]
    tokenized_sent.append(word_tokenize(Edit_text.lower()))
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
    model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
    test_doc = word_tokenize(Turn_topic.lower())
    test_doc_vector = model.infer_vector(test_doc)
    print(model.docvecs.most_similar(positive = [test_doc_vector]))
    s=model.docvecs.most_similar(positive = [test_doc_vector])
   
    row.append(s[0][1])
    j=j+1
    #Similarity between edit text and turn topic text
    tokenized_sent = []
    tokenized_sent.append(word_tokenize(Edit_text.lower()))
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
    model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
    test_doc = word_tokenize(Turn_topic_text.lower())
    test_doc_vector = model.infer_vector(test_doc)
    print(model.docvecs.most_similar(positive = [test_doc_vector]))
    s=model.docvecs.most_similar(positive = [test_doc_vector])
    
    row.append(s[0][1])
    j=j+1
    #Similarity between clean edit text and clean turn topic text
    if(edit_text_clean!='\n'):
        tokenized_sent = []
        tokenized_sent.append(word_tokenize(edit_text_clean.lower()))
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
        model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
        test_doc = word_tokenize(turntopic_text_clean.lower())
        test_doc_vector = model.infer_vector(test_doc)
        print(model.docvecs.most_similar(positive = [test_doc_vector]))
        s=model.docvecs.most_similar(positive = [test_doc_vector])
        row.append(s[0][1])
        j=j+1
    else:
        row.append(0)
    
    
    row.append(len(Edit_text))
    j=j+1
    
    row.append(len(Turn_text))
    j=j+1
    #Subjectivity of turn text
    sub=TextBlob(Edit_text).sentiment.subjectivity
    
    row.append(sub)
    j=j+1
    
    stop_words = set(stopwords.words('english')) 
    wordsList = nltk.word_tokenize(edit_text_clean)
    wordsList = [w for w in wordsList if w not in stop_words]
    tagged = nltk.pos_tag(wordsList)
    #Calculation of frequency of each of POS tags in eit text and turn text
    POS_tags=['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']
    tagfreq={}
    p=0
    if(tagged!=[]):
        for each in POS_tags:
            if each==tagged[p][1]:
                if each in tagfreq.keys():
                    tagfreq[each]=tagfreq[each]+1
                else:
                    tagfreq[each]=1
            else:
                tagfreq[each]=0
                
        for each in tagfreq.values():
           
            row.append(each)
            j=j+1
    else:
        for i in range(len(POS_tags)):
            row.append(0)
    # Calculation of frequency of unigrams, bigrams and trigrams 
    token = nltk.word_tokenize(clean(Edit_text))
    bigrams = ngrams(token,2)
    trigrams = ngrams(token,3)
    for each in unigrams_edittext_top100:
        if each in token:
            
            row.append(1)
        else:
            
            row.append(0)
        j=j+1
    for each in bigrams_edittext_top100:
        if each in bigrams:
            
            row.append(1)
        else:
            
            row.append(0)
        j=j+1
    for each in trigrams_edittext_top100:
        if each in token:
            
            row.append(1)
        else:
           
            row.append(0)
        j=j+1
        
    token = nltk.word_tokenize(clean(Turn_text))
    bigrams = ngrams(token,2)
    trigrams = ngrams(token,3)
    for each in unigrams_turntext_top100:
        if each in token:
            row.append(1)
        else:
            
            row.append(0)
        j=j+1
    for each in bigrams_turntext_top100:
        if each in bigrams:
            
            row.append(1)
        else:
            
            row.append(0)
        j=j+1
    for each in trigrams_turntext_top100:
        if each in token:
            row.append(1)
        else:
            
            row.append(0)
        j=j+1
    row.append(1)
    rows.append(row)
    i=i+1
with open('features', 'a') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)
csvfile.close()
