# Text Technologies for Data Science: Assignment 1
import re
from typing_extensions import Concatenate
from nltk import text
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
import os.path
import math
import operator

class IR_tool(object):

    docnums = []

    def parser(self, path):

        with open(path, "r") as f:

            if '.xml' in path: # If .xml we need to parse
                tree = ET.parse(path)
                myroot = tree.getroot()
                headlines = []
                texts = []
                for a in myroot.iter('DOCNO'):
                    self.docnums.append(a.text)
                for x in myroot.iter('HEADLINE'):
                    headlines.append(x.text)
                for y in myroot.iter('TEXT'):
                    texts.append(y.text)

                # self.docnums = docnums
                zippy = zip(headlines, texts)

                # For each tuple in the list we 'join' them together to make a string of headline + text
                # Then we want to further join each element in that list to make our full corpus we can tokenise :0
                lstcorpus = [''.join(x) for x in list(zippy)]

                i = 0
                IDdict = {}
                for corpus in lstcorpus: # Creating dictionary for for docID
                    IDdict[self.docnums[i]] = corpus # TODO - replace with docnums[i]
                    i +=1
                
                return IDdict

            else: 
                doc = f.read()
                f.close()

                Pattern = "\n" 

                lst = re.split(Pattern, doc)
                # lst = doc.split()

                return [elem for elem in lst if elem != ''] # Regex searches for singles chars so we remove empty strs.

    # Tokenisation, splits string on whitespace.
    def tokenisation(self, IDdict):

        if type(IDdict) is dict:
            for key in IDdict: # Tokenise each corpus (value) for each doc id (key)
                # IDdict[key] = re.split(Pattern, IDdict[key])
                IDdict[key] = re.split('[^A-Z^a-z\d]', IDdict[key]) # Use josh regex for sam
                # # IDdict[key] = IDdict[key].split()
                IDdict[key] = [elem.lower() for elem in IDdict[key] if elem != '']


                # and not(elem.isnumeric()) and 
                #                 not(any(char.isdigit() for char in elem)) 

            return IDdict

    # Can be used to remove stop words
    # (path, path) -> (list1, list2) -> list (list1 - list2)
    def removeWords(self, lst, words): # NOTE: This calls tokenisation. No need to call both on path.

        if type(lst) is dict: 

            for key in lst: # Through each document and remove stop words
                lst[key] = [elem for elem in lst[key] if not(elem in words)]

            return lst # Actually a dictionary

    def Pstem(self, corpuses):
        ps = PorterStemmer()

        if type(corpuses) is dict:
            for key in corpuses:
                corpuses[key] = [ps.stem(word) for word in corpuses[key]]

            return corpuses

        elif type(corpuses) is list: # For stemming queries
            return [ps.stem(word).lower() for word in corpuses]

        elif type(corpuses) is str:
            return ps.stem(corpuses).lower()

    def pre_process(self, docs, stopwords): # Works
        tokens = self.tokenisation(docs)
        doc_token_remove = self.removeWords(docs, stopwords)
        stem = self.Pstem(doc_token_remove)

        return stem

    def pre_process_query(self, queryterms): # For pre processing queries.
        return self.Pstem(queryterms)
        
    def pi_index(self, dictionary):

        # initialise positional indec
        pos_index = {}

        for doc in dictionary: # Itterate through each document 
            index_num = 0 # Keeps track of the index number for each word

            for term in dictionary[doc]: # Itterate through each word to add to index. Is this already in index or not?
                if term in pos_index: # If so, we must update the table as follows

                    if doc in pos_index[term][1]: # Has the word already occured in current document -> add position
                        pos_index[term][1][doc].append(index_num) # Simply add the index to current dict

                    else:
                        pos_index[term][1][doc] = [index_num] # Initialise the key with list including current index
                        pos_index[term][0]+=1 # First occurance of some term in some document so we increment.

                else: # If term has not yet been encountered then we must initialise it in the pos_index
                    pos_index[term] = [] # Initialise for adding rest of the information
                    pos_index[term].append(1) # Set initial counter to 1
                    pos_index[term].append({}) # Initialise dictionary for docnum and positions
                    pos_index[term][1][doc] = [index_num] # Place first doc and position instance.

                index_num+=1

        return pos_index

    def findphrase(self, words, pi_index): # Only works for two terms in phrase

        words = self.Pstem(words)

        docID = {} # list of documents in which the phrase apears.

        phrase_doc = {}
        for word in words:
            phrase_doc[word] = pi_index[word][1]
        
        for key1 in phrase_doc[words[0]].keys():
            for key2 in phrase_doc[words[1]].keys():

                if key1 == key2:
                    for pos1 in phrase_doc[words[0]][key1]:
                        for pos2 in phrase_doc[words[1]][key2]:
                            if pos1 == pos2-1:
                                
                                if key1 in docID:
                                    docID[key1].append(pos2)
                                else:
                                    docID[key1] = [pos2]

        return docID

    def proximity_hits(self, instances, proximity): # This takes two dictionaries (the pi_index values for each term) and checks for proximity hits.
        
        docID = []
        for key1 in instances[0]:
            for key2 in instances[1]:
                for pos1 in instances[0][key1]:
                    for pos2 in instances[1][key2]:
                        if key1 == key2 and abs(pos1 - pos2) <= proximity:
                            docID.append(key1) # Each doc where proximity hits.

        return list(set(docID))

    def bool_search(self, queries, pi_index): # Query is a string 
        results = ""

        qnum = 1
        for query in queries:
            strip_num = str([int(s) for s in query.split() if s.isdigit()][0]) + " " # Query num plus space
            query = query.removeprefix(strip_num) # Left with query only
            query_docs = {}

            if 'AND' in query:
                if ' AND NOT ' in query: # term1 AND NOT term2
                    query = query.split(" AND NOT ") # [term1, term2]
                    for term in query:
                        if " " in term: # phrase
                            phrase_words = re.findall(r'\w+', term)
                            query_docs[term] = list(self.findphrase(phrase_words, pi_index).keys())
                        else:
                            Pterm = self.Pstem(term)
                            query_docs[term] = pi_index[Pterm][1].keys()

                    return list(set([x for x in list(query_docs[query[0]]) if x not in list(query_docs[query[1]])]))               
                
                else:
                    query = query.split(" AND ") # [term1, term2]
                    for term in query:
                        if " " in term: # phrase
                            phrase_words = re.findall(r'\w+', term)
                            query_docs[term] = list(self.findphrase(phrase_words, pi_index).keys())
                        
                        else:
                            Pterm = self.Pstem(term)
                            query_docs[term] = pi_index[Pterm][1].keys()

                    return list(set([x for x in list(query_docs[query[0]]) if x in list(query_docs[query[1]])]))

            if 'OR' in query:
                if 'NOT' in query: # term1 OR NOT term2
                    query = query.split(" OR NOT ") # [term1, term2]
                    for term in query:
                        if " " in term: # phrase
                            phrase_words = re.findall(r'\w+', term)
                            query_docs[term] = list(self.findphrase(phrase_words, pi_index).keys())
                        
                        else:
                            Pterm = self.Pstem(term)
                            query_docs[term] = pi_index[Pterm][1].keys()

                    return list(set([x for x in self.docnums if x not in list(query_docs[query[1]]) or x in list(query_docs[query[0]])])) # Replace with docnums TODO

                else:
                    query = query.split(" OR ") # [term1, term2]
                    for term in query:
                        if " " in term: # phrase
                            phrase_words = re.findall(r'\w+', term)
                            query_docs[term] = list(self.findphrase(phrase_words, pi_index).keys())
                        
                        else:
                            Pterm = self.Pstem(term)
                            query_docs[term] = pi_index[Pterm][1].keys()

                    return list(set([x for x in self.docnums if x in list(query_docs[query[1]]) or x in list(query_docs[query[0]])]))

            if '#' in query: # Proximity search

                num = ''
                for c in query:
                    if c.isdigit():
                        num = num + c

                num = int(num)
                
                terms = re.findall(r'[\w\s]+', query)[1:]
                terms[1] = terms[1][1:]

                if '' in terms:
                    terms.remove('') # Random empty string apearing. TODO: FIND THIS PROBLEM.

                freq = [] # Frequency for each term.

                for term in terms:
                    if ' ' in term: # Phrase
                        phrase_words = re.findall(r'\w+', term)
                        result = self.findphrase(phrase_words, pi_index)    
                        freq.append(result)                 
                    
                    else:
                        term = self.Pstem(term)
                        result = pi_index[term][1]
                        freq.append(result)
            
                return self.proximity_hits(freq, num)


            else: # Single term
                if " " in query: # phrase
                    phrase_words = re.findall(r'\w+', query)
                    return list(set(list(self.findphrase(phrase_words, pi_index).keys())))

                else:
                    
                    word = re.findall(r'\w+', query)
                    Pterm = self.Pstem(word[0])
                    return list(set(list(pi_index[Pterm][1].keys())))

    def ranked_retreval(self, query, pi_index):
        query = query.split() # Nothing more complext required for these types of queries.
        query = query[1:] # Index 0 is qnum
        query = self.Pstem(query)

        scores = {}
        # Now for each term we gotta calc weight for each term then sum them up for each document
        for document in self.docnums: # TODO - REPLACE WITH DOCNUMS WHEN SORTED OUT
            score = 0
            for term in query:
                
                tf = 1
                df = 1
                if term in pi_index and document in pi_index[term][1]:
                    tf = len(pi_index[term][1][document])
                    df = pi_index[term][0] 
                    weight = (1 + math.log(tf, 10))*math.log((5000/df), 10)
                else:
                    weight = 0

                score += weight

            scores[document] = score

        return scores

if __name__ == "__main__":
    ir = IR_tool()

    stopwords = ir.parser('Collections/stopwords.txt')
    collection = ir.parser('Collections/trec.5000.xml')
    bool_queries = ir.parser('Collections/queries.boolean.txt')
    ranked_queries = ir.parser('Collections/queries.ranked.txt')

    ps = ir.pre_process(collection, stopwords)
    pi = ir.pi_index(ps)

    index = ""
    for term in pi:
        index+=term + ":" + str(pi[term][0]) + "\n"
        for docID in pi[term][1]:
            index+= "\t" + str(docID) + ": "
            for pos in pi[term][1][docID]:
                index+= str(pos) + ", "
            index = index.removesuffix(", ")
            index+= "\n"

    with open("index.txt", "w") as f:
        f.write(index)
        f.close()

    results = ""
    qnum = 1
    for bool_querie in bool_queries:

        output = ir.bool_search([bool_querie], pi)

        int_output = []
        for r in output:
            int_output.append(int(r))

        int_output.sort()

        for r in int_output:
            results += str(qnum) + ", " + str(r) + "\n"
        qnum+=1

    with open("boolean.results.txt", "w") as f:
        f.write(results)
        f.close

    results = ""
    qnum = 1

    for rq in ranked_queries:
        ranked = ir.ranked_retreval(rq, pi)
        sort_orders = sorted(ranked.items(), key=lambda x: x[1], reverse=True)

        for elem in sort_orders[:150]:
            results+= str(qnum) + ", " + str(elem[0]) + ", " + str(round(elem[1], 4)) + "\n"

        qnum +=1

    with open("results.ranked.txt", "w") as f:
        f.write(results)
        f.close()


