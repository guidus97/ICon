# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:36:48 2019

@author: utente
"""

import re
import random
import os
from nltk.corpus import wordnet
from fast_autocomplete import AutoComplete


def split_kb(filename):
   
    owe=[]
    ultimate=[]
    
    with open(filename) as pl:
        for line in pl:
            owe.append(line)
            
    for i in range(0,len(owe)):
        ultimate.append(re.split(':-|,',owe[i]))
    
    return ultimate


def deduction(filename):
    C=set()
    kb=split_kb(filename)
    
    for i in range(0,len(kb)):
       for atom in kb[i][1:]:
           if atom in C and kb[i][0] not in C :
               C.add(kb[i][0])
           if len(kb[i]) == 2:
               C.add(kb[i][0])
               
    return C


def words(filename):
    dictionary=dict()
    
    for x in deduction(filename):
        dictionary[x]={}
        
    return dictionary
    
    
    
def foundname(search):
   filename=random.choice([x for x in os.listdir(os.path.join('Images',search)) if os.path.isfile(os.path.join(os.path.join('Images',search),x))])
   filename=os.path.join(os.path.join('Images',search),filename)
   return filename


def synset(text):
    definition=wordnet.synsets(text)
    return definition
        
   
def autocomplete(token,filename):
    autocomplete=AutoComplete(words=words(filename))
    sugg=autocomplete.search(word=token)
    return sugg
        

    