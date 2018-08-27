
# coding: utf-8

# # Software Design Using ML&AI nWave
# 
# 
# # 1. Setup
# 
# To prepare your environment, you need to install some packages
# 
# # 1.1 Install the necessary packages
# 
# You need the latest versions of these packages:<br>
#  
# 
# ** Pandas for dataframe.<br>
# ** stop_words: **List of common stop words.<br>
# ** python-boto3:** is a python client for the Boto3 API used for communicating to AWS.<br>
# ** websocket-client: ** is a python client for the Websockets.<br>
# ** pyorient: ** is a python client for the Orient DB.<br><br>
# 
# 

# ** Install NLTK: **

# In[1]:


get_ipython().system('pip install --upgrade nltk')
get_ipython().system('pip install --upgrade pyorient')


# **Install Boto3 client for AWS communication thorugh CLI **

# In[2]:


get_ipython().system('pip install boto3 ')


# ** Install stop_words **

# In[3]:


get_ipython().system('pip install stop-words')


# ** Install websocket client: **

# In[4]:


get_ipython().system('pip install websocket-client')


# ** Install pyorient: **

# In[5]:


get_ipython().system(' pip install awscli')
get_ipython().system(' pip install pyorient --user')


# # 1.2 Import packages and libraries 
# 
# Import the packages and libraries that you'll use:

# In[6]:


import json

import nltk
from nltk.cluster.util import cosine_distance
from stop_words import get_stop_words
import numpy
import re
import boto3
from botocore.client import Config

import websocket
import _thread
import time

from io import BytesIO
import pandas as pd
import json
import sys


# # 2. Configuration
# 
# Add configurable items of the notebook below
# ## 2.1 Add your service credentials if any required( this is where you need to add credentials of infrastructure you are using to store data etc)
# 
# 
# Run the cell.

# In[7]:


### This is the section to provide credentials for AWS S3 account
### While sharing the notebook remove them -- will try to make this cell hidden later

## Console URL :::  https://awstestconsole-swaroop.signin.aws.amazon.com/console
## Account Id: 
## Username : 
## Password : 
## Then Navigate to the S3 section


# ## 2.2 Add your service credentials for S3
# 
# You must create S3 bucket service on AWS. To access data in a file in Object Storage, you need the Object Storage authentication credentials. Insert the Object Storage authentication credentials as credentials_1 in the following cell after removing the current contents in the cell.

# In[8]:


# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'ACCESS_KEY_ID': 'AKIAJW2Q23KX7U6DQMCQ',
    'ACCESS_SECRET_KEY': 'qqgJ9L1Y8BepDnh/v5BwIvxlUUPbWrPJ6OoDchls',
    'BUCKET': 'banking-brd-bucket'
}


# # 3. Text Classification  ( this section will be required if we use spacy for machine learning)
# 
# Write the classification related utility functions in a modularalized form.
# 
# ## 3.1  REQUIREMENT Classification
# 

# In[9]:


def chunk_sentence(text):
    """ Tag the sentence using chunking.
    """
    grammar = """
      Action: {<VB.?><NN.?>+}
      Action: {<VB.?><CLAUSE1><NN.?>+}
               }<CLAUSE1>{
      Action: {<VB.?><CLAUSE1><CLAUSE1><NN.?>+}
               }<CLAUSE1>{
      Action: {<VB.?><CLAUSE1><CLAUSE1><CLAUSE1><NN.?>+}
               }<CLAUSE1>{  
      CLAUSE1: {<DT|PRP.?|IN|JJ>}
      
      
      """  
    parsed_cp = nltk.RegexpParser(grammar,loop=2)
    pos_cp = parsed_cp.parse(text)
    return pos_cp


# ## 3.2 Augumented Classification
# 
# Custom classification utlity fucntions for augumenting the results by using the Grammar rule defined in JSON file  

# In[10]:


def split_sentences(text):
    """ Split text into sentences.
    """
    sentence_delimiters = re.compile(u'[\\[\\]\n.!?]')
    sentences = sentence_delimiters.split(text)
    return sentences

def split_into_tokens(text):
    """ Split text into tokens.
    """
    tokens = nltk.word_tokenize(text)
    return tokens
    
def POS_tagging(text):
    """ Generate Part of speech tagging of the text.
    """
    POSofText = nltk.tag.pos_tag(text)
    return POSofText


def keyword_tagging(tag,tagtext,text):
    """ Tag the text matching keywords.
    """
    if (text.lower().find(tagtext.lower()) != -1):
        return text[text.lower().find(tagtext.lower()):text.lower().find(tagtext.lower())+len(tagtext)]
    else:
        return 'UNKNOWN'
    
def regex_tagging(tag,regex,text):
    """ Tag the text matching REGEX.
    """    
    p = re.compile(regex, re.IGNORECASE)
    matchtext = p.findall(text)
    regex_list=[]    
    if (len(matchtext)>0):
        for regword in matchtext:
            regex_list.append(regword)
    return regex_list

def BRD_chunk_tagging(tag,chunk,text):
    """ Tag the text using chunking.
    """
    parsed_cp = nltk.RegexpParser(chunk)
    pos_cp = parsed_cp.parse(text)
    #pos_cp = chunk_sentence(text) #*** use this for getting refined output after chinking but extra entities in output
    chunk_list=[]
    for root in pos_cp:
        if isinstance(root, nltk.tree.Tree):               
            if root.label() == tag:
                chunk_word = ''
                for child_root in root:
                    chunk_word = chunk_word +' '+ child_root[0]
                chunk_list.append(chunk_word)
    return chunk_list

def chunk_tagging(tag,chunk,text):
    """ Tag the text using chunking.
    """
    parsed_cp = nltk.RegexpParser(chunk)
    pos_cp = parsed_cp.parse(text)
    chunk_list=[]
    for root in pos_cp:
        if isinstance(root, nltk.tree.Tree):               
            if root.label() == tag:
                chunk_word = ''
                for child_root in root:
                    chunk_word = chunk_word +' '+ child_root[0]
                chunk_list.append(chunk_word)
    return chunk_list
    
def augument_SpResponse(responsejson,updateType,text,tag):
    """ Update the output JSON with augumented classifications.
    """
    if(updateType == 'keyword'):
        if not any(d.get('text', None) == text for d in responsejson['Keywords']):
            responsejson['Keywords'].append({"User":text})
    else:
        if not any(d.get('text', None) == text for d in responsejson['Entities']):
            responsejson['Entities'].append({"type":tag,"text":text}) 

def classify_BRD_text(text, config):
    """ Perform augumented classification of the text for BRD specifically for getting the output with action.
    """
    
    #will be used for storing initial value of response json, this is from nlu earlier
    with open('output_format_BRD.json') as f:
        responsejson = json.load(f)
    
    sentenceList = split_sentences(text) #
    
    tokens = split_into_tokens(text)
    
    postags = POS_tagging(tokens)
    
    configjson = json.loads(config)#load would take a file-like object, read the data from that object, and use that string to create an object:
    
    no_of_items = 0
    
    for stages in configjson['configuration']['classification']['stages']:
        # print('Stage - Performing ' + stages['name']+':')
        for steps in stages['steps']:
            # print('    Step - ' + steps['type']+':')
            if (steps['type'] == 'keywords'):
                for keyword in steps['keywords']:
                        wordtag = tokens[0]
                augument_SpResponse(responsejson,'keyword',wordtag,keyword['tag'])
            elif(steps['type'] == 'd_regex'):
                for regex in steps['d_regex']:
                    for word in sentenceList:
                        regextags = regex_tagging(regex['tag'],regex['pattern'],word)
                        if (len(regextags)>0):
                            for words in regextags:
                                #print('      '+regex['tag']+':'+words)
                                augument_SpResponse(responsejson,'Action',words,regex['tag'])
            elif(steps['type'] == 'chunking'):
                for chunk in steps['chunk']:
                    chunktags = BRD_chunk_tagging(chunk['tag'],chunk['pattern'],postags)
                    if (len(chunktags)>0):
                        for words in chunktags:
                            #print('      '+chunk['tag']+':'+words)
                            if (no_of_items <1):
                                augument_SpResponse(responsejson,'Action',words,chunk['tag'])
                                no_of_items = no_of_items + 1
            else:
                print('UNKNOWN STEP')
    
    
    
    return responsejson



def classify_text(text, config):
    """ Perform augumented classification of the text.
    """
    
    #will be used for storing initial value of response json, this is from nlu earlier
    with open('sample.json') as f:
        responsejson = json.load(f)
    
    sentenceList = split_sentences(text) #
    
    tokens = split_into_tokens(text)
    
    postags = POS_tagging(tokens)
    
    configjson = json.loads(config)#load would take a file-like object, read the data from that object, and use that string to create an object:
    
    for stages in configjson['configuration']['classification']['stages']:
        # print('Stage - Performing ' + stages['name']+':')
        for steps in stages['steps']:
            # print('    Step - ' + steps['type']+':')
            if (steps['type'] == 'keywords'):
                for keyword in steps['keywords']:
                    for word in sentenceList:
                        wordtag = keyword_tagging(keyword['tag'],keyword['text'],word)
                        if(wordtag != 'UNKNOWN'):
                            #print('      '+keyword['tag']+':'+wordtag)
                            augument_SpResponse(responsejson,'keyword',wordtag,keyword['tag'])
            elif(steps['type'] == 'd_regex'):
                for regex in steps['d_regex']:
                    for word in sentenceList:
                        regextags = regex_tagging(regex['tag'],regex['pattern'],word)
                        if (len(regextags)>0):
                            for words in regextags:
                                #print('      '+regex['tag']+':'+words)
                                augument_SpResponse(responsejson,'entities',words,regex['tag'])
            elif(steps['type'] == 'chunking'):
                for chunk in steps['chunk']:
                    chunktags = chunk_tagging(chunk['tag'],chunk['pattern'],postags)
                    if (len(chunktags)>0):
                        for words in chunktags:
                            #print('      '+chunk['tag']+':'+words)
                            augument_SpResponse(responsejson,'entities',words,chunk['tag'])
            else:
                print('UNKNOWN STEP')
    
    
    return responsejson


def replace_unicode_strings(response):
    """ Convert dict with unicode strings to strings.
    """
    if isinstance(response, dict):
        return {replace_unicode_strings(key): replace_unicode_strings(value) for key, value in response.iteritems()}
    elif isinstance(response, list):
        return [replace_unicode_strings(element) for element in response]
    elif isinstance(response, str):
        return response.encode('utf-8')
    else:
        return response


# # 4. Correlate text content

# In[11]:


stopWords = get_stop_words('english')
# List of words to be ignored for text similarity
stopWords.extend(["The","This","That",".","!","?"])

def compute_text_similarity(text1, text2, text1tags, text2tags):
    """ Compute text similarity using cosine
    """
    #stemming is the process for reducing inflected (or sometimes derived) words to their stem, base or root form
    stemmer = nltk.stem.porter.PorterStemmer()
    sentences_text1 = split_sentences(text1)
    sentences_text2 = split_sentences(text2)
    tokens_text1 = []
    tokens_text2 = []
    
    for sentence in sentences_text1:
        tokenstemp = split_into_tokens(sentence.lower())
        tokens_text1.extend(tokenstemp)
    
    for sentence in sentences_text2:
        tokenstemp = split_into_tokens(sentence.lower())
        tokens_text2.extend(tokenstemp)
    if (len(text1tags) > 0):  
        tokens_text1.extend(text1tags)
    if (len(text2tags) > 0):    
        tokens_text2.extend(text2tags)
    
    tokens1Filtered = [stemmer.stem(x) for x in tokens_text1 if x not in stopWords]
    
    tokens2Filtered = [stemmer.stem(x) for x in tokens_text2 if x not in stopWords]
    
    #  remove duplicate tokens
    tokens1Filtered = set(tokens1Filtered)
    tokens2Filtered = set(tokens2Filtered)
   
    tokensList=[]

    text1vector = []
    text2vector = []
    
    if len(tokens1Filtered) < len(tokens2Filtered):
        tokensList = tokens1Filtered
    else:
        tokensList = tokens2Filtered

    for token in tokensList:
        if token in tokens1Filtered:
            text1vector.append(1)
        else:
            text1vector.append(0)
        if token in tokens2Filtered:
            text2vector.append(1)
        else:
            text2vector.append(0)  

    cosine_similarity = 1-cosine_distance(text1vector,text2vector)
    if numpy.isnan(cosine_similarity):
        cosine_similarity = 0
    
    return cosine_similarity


# # 5. Persistence and Storage
# ## 5.1 Configure Object Storage Client

# In[12]:


s3 = boto3.client('s3',
                    aws_access_key_id=credentials_1['ACCESS_KEY_ID'],
                    aws_secret_access_key=credentials_1['ACCESS_SECRET_KEY'],
                    config=Config(signature_version='s3v4')
                     )
#Enter the path where you want to store data downlaoded from S3


def get_file(filename,Location):
    s3.download_file(Bucket=credentials_1['BUCKET'],Key=filename,Filename=Location)
    #t="abc"

#def load_string(fileobject):
#    '''Load the file contents into a Python string'''
#    text = fileobject.read()
#    return text

#def load_df(fileobject,sheetname):
#    '''Load file contents into a Pandas dataframe'''
#    excelFile = pd.ExcelFile(fileobject)
#    df = excelFile.parse(sheetname)
#    return df

#def put_file(filename, filecontents):
#    '''Write file to Cloud Object Storage'''
#    resp = s3.put_object(Bucket=credentials_1['BUCKET'], Key=filename, Body=filecontents)
    #resp = s3.Bucket(Bucket=credentials_1['BUCKET']).put_object(Key=filename, Body=filecontents)
#    return resp


# In[13]:


# Name of the excel file with data in S3 Storage
BrdFileName = "Banking-BRD.xlsx"
# Choose or get as an input as to which Domain it belongs to i.e banking, healthcare etc
Domain = "Banking"

# Name of the config files in Object Storage. Rule_brd will be used specifically for parsing requirement document
configFileName = "sample_config.txt"
BRD_configFileName = "Rule_BRD.txt"
# Config contents
config = None;

Path = ".//temp/"
# Output excell

# Requirements dataframe
requirements_file_name = "Requirements.xlsx"
#requirements_sheet_name = "".join((Domain,"-Requirements"))
#requirements_df = None

# Domain/UseCase dataframe
domain_file_name = "Domain.xlsx"
#domain_sheet_name = "".join((Domain,"-Domain"))
#domain_df = None
rule_file_name = "Rule_BRD.txt"

# DataElements dataframe
dataelements_file_name ="DataElements.xlsx"
#dataelements_sheet_name ="".join((Domain,"-Dataelements"))
#dataelements_df = None

def load_artifacts():
    Location = "".join((Path,requirements_file_name))
    get_file(requirements_file_name,Location)
    Location = "".join((Path,domain_file_name))
    get_file(domain_file_name,Location)
    Location = "".join((Path,dataelements_file_name))
    get_file(dataelements_file_name,Location)
    Location = "".join((Path,rule_file_name))
    get_file(rule_file_name,Location)

load_artifacts()


# ## 5.2 OrientDB client - functions to connect, store and retrieve data

# ** Connect to OrientDB **

# In[14]:


import pyorient
client = pyorient.OrientDB(host="localhost", port=2424)
user = "root"
passw = "root"
session_id = client.connect(user, passw)


# ** OrientDB Core functions **

# In[15]:


def create_database(dbname, username, password):
    """ Create a database
    """
    client.db_create( dbname, pyorient.DB_TYPE_GRAPH, pyorient.STORAGE_TYPE_MEMORY )
    print(dbname  + " created and opened successfully")
        
def drop_database(dbname):
    """ Drop a database
    """
    if client.db_exists( dbname, pyorient.STORAGE_TYPE_MEMORY ):
        client.db_drop(dbname)
    
def create_class(classname):
    """ Create a class
    """
    command = "create class "+classname + " extends V"
    client.command(command)
    
def create_record(classname, entityname, attributes):
    """ Create a record
    """
    command = "insert into " + classname + " set " 
    attrstring = ""
    for index,key in enumerate(attributes):
        attrstring = attrstring + key + " = '"+ attributes[key] + "'"
        if index != len(attributes) -1:
            attrstring = attrstring +","
    command = command + attrstring
    client.command(command)
    
def create_domain_dataelements_edge(domainid, dataelementid, attributes):
    """ Create an edge between a domain n dataelement 
    """
    command = "create edge linkeddataelements from (select from Domains where ID = " + "'" + domainid + "') to (select from DataElements where ID = " + "'" + dataelementid + "')" 
    if len(attributes) > 0:
        command = command + " set "
    attrstring = ""
    for index,key in enumerate(attributes):
        val = attributes[key]
        if not isinstance(val, str):
            val = str(val)
        attrstring = attrstring + key + " = '"+ val + "'"
        if index != len(attributes) -1:
            attrstring = attrstring +","
    command = command + attrstring
    print(command)
    client.command(command)    
    
def create_dataelements_requirement_edge(testcaseid, reqid, attributes):
    """ Create an edge between a testcase and a requirement
    """
    command = "create edge linkedrequirements from (select from DataElements where ID = "+ "'" + testcaseid+"') to (select from Requirements where ID = "+"'"+reqid+"')" 
    if len(attributes) > 0:
        command = command + " set "
    attrstring = ""
    for index,key in enumerate(attributes):
        val = attributes[key]
        if not isinstance(val, str):
            val = str(val)
        attrstring = attrstring + key + " = '"+ val + "'"
        if index != len(attributes) -1:
            attrstring = attrstring +","
    command = command + attrstring
    client.command(command)  

    
def create_requirement_domain_edge(reqid, functionalityid, attributes):
    """ Create an edge between a requirement and a domain
    """
    command = "create edge linkeddomains from (select from Requirements where ID = "+ "'" + reqid+"') to (select from Domains where ID = "+"'"+functionalityid+"')" 
    
    if len(attributes) > 0:
         command = command + " set "
    attrstring = ""
    for index,key in enumerate(attributes):
        val = attributes[key]
        if not isinstance(val, str):
            val = str(val)
        attrstring = attrstring + key + " = '"+ val + "'"
        if index != len(attributes) -1:
            attrstring = attrstring +","
    command = command + attrstring
    print(command)
    client.command(command) 
    
def execute_query(query):
    """ Execute a query
    """
    return client.query(query)


# ** OrientDB Insights **

# In[16]:


def get_related_domaincases(reqid):
    """ Get the related domaincases for a requirements
    """
    domaincasesQuery = "select * from ( select expand( out('linkeddomains')) from Requirements where ID = '" + reqid +"' )"
    domaincases = execute_query(domaincasesQuery)
    scoresQuery = "select expand(out_linkeddomains) from Requirements where ID = '"+reqid+"'"
    scores = execute_query(scoresQuery)
    domaincaseList =[]
    scoresList= []
    for domaincase in domaincases:
        domaincaseList.append(domaincase.ID)
    for score in scores:
        scoresList.append(score.score)
    result = {}
    length = len(domaincaseList)
    for i in range(0, length):
        result[domaincaseList[i]] = scoresList[i]
    #print(result)
    return result

def get_related_dataelements(domaincaseid):
    """ Get the related requirements for a testcase
    """
    dataelementsQuery = "select * from ( select expand( out('linkeddataelements') ) from Domains where ID = '" + domaincaseid +"' )"
    dataelements = execute_query(dataelementsQuery)
    #print(dataelements)
    scoresQuery = "select expand(out_linkeddataelements) from Domains where ID = '"+domaincaseid+"'"
    scores = execute_query(scoresQuery)
    dataelementsList =[]
    scoresList= []
    for dataelement in dataelements:
        dataelementsList.append(dataelement.ID)
    for score in scores:
        scoresList.append(score.score)
    result = {}
    length = len(dataelementsList)
    #print requirementsList, scoresList
    for i in range(0, length):
        result[dataelementsList[i]] = scoresList[i]
    return result


def get_related_defects(reqid):
    """ Get the related defects for a requirement
    """
    defectsQuery = "select * from ( select expand( out('linkeddefects')) from Requirement where ID = '" + reqid +"' )"
    defects = execute_query(defectsQuery)
    scoresQuery = "select expand(out_linkeddefects) from Requirement where ID = '"+reqid+"'"
    scores = execute_query(scoresQuery)
    defectsList =[]
    scoresList= []
    for defect in defects:
        defectsList.append(defect.ID)
    for score in scores:
        scoresList.append(score.score)
    result = {}
    length = len(defectsList)
    for i in range(0, length):
        result[defectsList[i]] = scoresList[i]
    return result

def build_format_defects_list(defectsResult):
    """ Build and format the OrientDB query results for defects
    """
    defects = []
    for defect in defectsResult:
        detail = {}
        detail['ID'] = defect.ID
        detail['Severity'] = defect.Severity
        detail['Description'] = defect.Description
        defects.append(detail)
    return defects

def build_format_testcases_list(testcasesResult):
    """ Build and format the OrientDB query results for testcases
    """
    testcases = []
    for testcase in testcasesResult:
        detail = {}
        detail['ID'] = testcase.ID
        detail['Category'] = testcase.Category
        detail['Description'] = testcase.Description
        testcases.append(detail)
    return testcases  

def build_format_requirements_list(requirementsResult):
    """ Build and format the OrientDB query results for requirements
    """
    requirements = []
    for requirement in requirementsResult:
        detail = {}
        detail['ID'] =requirement.ID
        detail['Description'] = requirement.Description
        detail['User'] = requirement.User
        requirements.append(detail)
    return requirements  

def get_requirements():
    """ Get all requirements
    """
    requirementsQuery = "select * from Requirements"
    requirementsResult =  execute_query(requirementsQuery)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements  

def build_format_requirements_list(requirementsResult):
    """ Build and format the OrientDB query results for requirements
    """
    requirements = []
    for requirement in requirementsResult:
        detail = {}
        detail['ID'] =requirement.ID
        detail['Description'] = requirement.Description
        detail['User'] = requirement.User
        requirements.append(detail)
    return requirements  

def get_requirements():
    """ Get all requirements
    """
    requirementsQuery = "select * from Requirements"
    requirementsResult =  execute_query(requirementsQuery)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements 

def get_related_domaincases(reqid):
    """ Get the related domaincases for a requirements
    """
    domaincasesQuery = "select * from ( select expand( out('linkeddomains')) from Requirements where ID = '" + reqid +"' )"
    domaincases = execute_query(domaincasesQuery)
    scoresQuery = "select expand(out_linkeddomains) from Requirements where ID = '"+reqid+"'"
    scores = execute_query(scoresQuery)
    domaincaseList =[]
    domaincaseAction =[]
    scoresList= []
    for domaincase in domaincases:
        domaincaseList.append(domaincase.ID)
        domaincaseAction.append(domaincase.Action)
    for score in scores:
        scoresList.append(score.score)
    result = {}
    length = len(domaincaseList)
    for i in range(0, length):
        result[domaincaseList[i]] = scoresList[i]
        #result[domaincaseList[i]] = domaincaseAction[i]
    print(result)
    return result

def get_related_dataelements(domaincaseid):
    """ Get the related requirements for a testcase
    """
    dataelementsQuery = "select * from ( select expand( out('linkeddataelements') ) from Domains where ID = '" + domaincaseid +"' )"
    dataelements = execute_query(dataelementsQuery)
    #print(dataelements)
    scoresQuery = "select expand(out_linkeddataelements) from Domains where ID = '"+domaincaseid+"'"
    scores = execute_query(scoresQuery)
    dataelementsList =[]
    scoresList= []
    for dataelement in dataelements:
        dataelementsList.append(dataelement.ID)
    for score in scores:
        scoresList.append(score.score)
    result = {}
    length = len(dataelementsList)
    #print requirementsList, scoresList
    for i in range(0, length):
        result[dataelementsList[i]] = scoresList[i]
    return result

def get_related_user(reqid):
    reqQuery = "select * from Requirements where ID = '"+reqid+"'"
    requirements = execute_query(reqQuery)
    result = ''
    for requirement in requirements:
        print(requirement.User)
        result = requirement.User
    return result

def get_related_action(key):
    domQuery = "select * from Domains where ID = '"+key+"'"
    domains = execute_query(domQuery)
    result = ''
    for domain in domains:
        print(domain.Action)
        result = domain.Action
    return result

def get_related_shorthand(key):
    shortQuery = "select * from DataElements where ID = '"+key+"'"
    shorts = execute_query(shortQuery)
    result = ''
    for short in shorts:
        print(short.Short)
        result = short.Short
    return result
def get_requirement_defects(numdefects):
    """ Get requirements that have more than a given number of defects
    """
    query = "select ID,Description,Priority from Requirement where out('linkeddefects').size() >= " + str(numdefects)
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    for requirement in requirements:
        num = len(get_related_defects(requirement['ID']))
        requirement['defectcount'] = num
    return requirements 
def merge_apply_filters_d3_bubble(mainList, filterList):
    """ Add a filter attribute to the list elements for processing on UI
    """
    mainListChildren = mainList['children']
    filterListChildren = filterList['children']
    for child in mainListChildren:
        child['filter'] = 0
        for child1 in filterListChildren:
            if ( child['ID'] == child1['ID']):
                child['filter'] = 1
    return mainList

def get_requirement_defects(numdefects):
    """ Get requirements that have more than a given number of defects
    """
    query = "select ID,Description,Priority from Requirement where out('linkeddefects').size() >= " + str(numdefects)
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    for requirement in requirements:
        num = len(get_related_defects(requirement['ID']))
        requirement['defectcount'] = num
    return requirements 

def get_requirements_banker():
    """ Get requirements that have no defects
    """
    query = "Select * from Requirements where User = 'Banker'"
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements
def get_requirements_customer():
    """ Get requirements that have no defects
    """
    query = "Select * from Requirements where User = 'Customer'"
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements
def get_requirement_domain(numde):
    """ Get requirements that have more than a given number of defects
    """
    query = "select ID,Description,User from Requirements where out('linkeddomains').size() <= " + str(numde)
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements
 


# # 6. Data Preparation

# ## 6.1 Global variables and functions

# In[1]:


# Name of the excel file with data in S3 Storage
BrdFileName = "Banking-BRD.xlsx"
# Choose or get as an input as to which Domain it belongs to i.e banking, healthcare etc
Domain = "Banking"

# Name of the config files in Object Storage. Rule_brd will be used specifically for parsing requirement document
configFileName = "sample_config.txt"
BRD_configFileName = "Rule_BRD.txt"
# Config contents
config = None;

Path = "D:/machine learning/watson-document-co-relation-master/watson-document-co-relation-master/"
# Output excell

# Requirements dataframe
requirements_file_name = "data/Requirements.xlsx"
requirements_sheet_name = "".join((Domain,"-Requirements"))
requirements_df = None

# Domain/UseCase dataframe
domain_file_name = "data/Domain.xlsx"
domain_sheet_name = "".join((Domain,"-Domain"))
domain_df = None

# DataElements dataframe
dataelements_file_name ="data/DataElements.xlsx"
dataelements_sheet_name ="".join((Domain,"-Dataelements"))
dataelements_df = None

def load_artifacts():
    global requirements_df 
    global domain_df 
    global dataelements_df 
    global config
    global BRD_config
    global Path
    
    
    Location = "".join((Path,requirements_file_name))
    get_file(requirements_file_name,Location)
    excel = pd.ExcelFile(Location)
    requirements_df = excel.parse(requirements_sheet_name)
    Location = "".join((Path,domain_file_name))
    get_file(domain_file_name,Location)
    excel = pd.ExcelFile(Location)
    domain_df = excel.parse(domain_sheet_name)
    Location = "".join((Path,dataelements_file_name))
    get_file(dataelements_file_name,Location)
    excel = pd.ExcelFile(Location)
    dataelements_df = excel.parse(dataelements_sheet_name)
    rule_text = open(configFileName)
    config = rule_text.read()
    BRD_rule_text = open(BRD_configFileName)
    BRD_config = BRD_rule_text.read()
    
    

def prepare_artifact_dataframes():
    """ Prepare artifact dataframes by creating necessary output columns
    """
    global requirements_df 
    global domain_df 
    global dataelements_df 
    req_cols_len = len(requirements_df.columns)
    dom_cols_len = len(domain_df.columns)
    dat_cols_len = len(dataelements_df.columns)
    requirements_df.insert(req_cols_len, "ClassifiedText","")
    requirements_df.insert(req_cols_len+1, "Keywords","")
    requirements_df.insert(req_cols_len+2, "DomainMatchScore","")
    
    domain_df.insert(dom_cols_len, "ClassifiedText","")
    domain_df.insert(dom_cols_len+1, "Keywords","")
    domain_df.insert(dom_cols_len+2, "DataElementsMatchScore","")

    dataelements_df.insert(dat_cols_len, "ClassifiedText","")
    dataelements_df.insert(dat_cols_len+1, "Keywords","")
    dataelements_df.insert(dat_cols_len+2, "RequirementsMatchScore","")


# ## 6.2 Utility functions for Engineering Insights

# In[18]:


def mod_req_text_classifier_output(artifact_df, BRD_config, output_column_name):
    """ Add text classifier output to the artifact dataframe based on rule defined in config
    """
    for index, row in artifact_df.iterrows():
        summary = row["I want to <perform some task>"]
        modID = row["ID"]
        
        print(modID)
        modID = modID.replace("R","UC")
        print(modID)
        row["ID"]= modID
        user = row["As a <type of user>"]
        user = "".join((user," want to "))
        summary = "".join((user,summary))
        #print("--------------")
        #print(summary)
        classifier_journey_output = classify_BRD_text(summary, BRD_config)
        #print(classifier_journey_output)
        artifact_df.set_value(index, output_column_name, classifier_journey_output)
    return artifact_df 


def add_text_classifier_output(artifact_df, config, output_column_name):
    """ Add text classifier output to the artifact dataframe based on rule defined in config
    """
    for index, row in artifact_df.iterrows():
        summary = row["Description"]
        #print("--------------")
        #print(summary)
        classifier_journey_output = classify_text(summary, config)
        #print(classifier_journey_output)
        artifact_df.set_value(index, output_column_name, classifier_journey_output)
    return artifact_df 
           
def add_keywords_entities(artifact_df, classify_text_column_name, output_column_name):
    """ Add keywords and entities to the artifact dataframe"""
    for index, artifact in artifact_df.iterrows():
        keywords_array = []
        for row in artifact[classify_text_column_name]['Keywords']:
            #print(row)
            if not row['User'] in keywords_array:
                keywords_array.append(row['User'])
                
        for entities in artifact[classify_text_column_name]['Entities']:
            if not entities['text'] in keywords_array:
                keywords_array.append(entities['text'])
            if not entities['type'] in keywords_array:
                keywords_array.append(entities['type'])
        artifact_df.set_value(index, output_column_name, keywords_array)
        #print(keywords_array)
    return artifact_df 

#requirements_df, domain_df, keywords_column_name, output_column_name)

def populate_text_similarity_score(artifact_df1, artifact_df2, keywords_column_name, output_column_name):
    """ Populate text similarity score to the artifact dataframes
    """
    heading1 = "Description"
    heading2 = "Description"
    
    try:
        artifact_df1[heading1]
    except: 
        heading1 = "I want to <perform some task>"
    
    try:
        artifact_df2[heading2]
    except: 
        heading2 = "I want to <perform some task>"    
    
    
    for index1, artifact1 in artifact_df1.iterrows():
        matches = []
        top_matches = []
        for index2, artifact2 in artifact_df2.iterrows():
            matches.append({'ID': artifact2['ID'], 
                            'cosine_score': 0, 
                            'SubjectID':artifact1['ID']})
            cosine_score = compute_text_similarity(
                #artifact1['Description'], 
                #artifact2['Description'], 
                artifact1[heading1],
                artifact2[heading2],
                artifact1['Keywords'], 
                artifact2['Keywords'])
            matches[index2]["cosine_score"] = cosine_score
       
        sorted_obj = sorted(matches, key=lambda x : x['cosine_score'], reverse=True)
    
    # This is where the lower cosine value to be truncated is set and needs to be adjusted based on output
    
        for obj in sorted_obj:
            if obj['cosine_score'] > 0.6:
                top_matches.append(obj)
               
        artifact_df1.set_value(index1, output_column_name, top_matches)
    return artifact_df1


# ## 6.3 Process flow

# ** Prepare data **
# * Load artifacts from object storage and create pandas dataframes
# * Prepare the pandas dataframes. Add additional columns required for further processing.

# In[19]:


load_artifacts()
prepare_artifact_dataframes()


# ** Run Text Classification on data **
# * Add the text classification output to the artifact dataframes

# In[20]:


output_column_name = "ClassifiedText"
requirements_df = mod_req_text_classifier_output(requirements_df, BRD_config, output_column_name)

domain_df = add_text_classifier_output(domain_df,config, output_column_name)
dataelements_df = add_text_classifier_output(dataelements_df,config, output_column_name)

requirements_df.head()


# ** Populate keywords and entities **
# * Add the keywords and entities extracted from the unstructured text to the artifact dataframes

# In[21]:


classify_text_column_name = "ClassifiedText"
output_column_name = "Keywords"
requirements_df = add_keywords_entities(requirements_df, classify_text_column_name, output_column_name)
domain_df = add_keywords_entities(domain_df, classify_text_column_name, output_column_name)
dataelements_df = add_keywords_entities(dataelements_df, classify_text_column_name, output_column_name)


# ** Correlate keywords between artifacts **
# * Add the text similarity score of associated artifacts to the dataframe

# In[22]:


keywords_column_name = "Keywords"
output_column_name = "DomainMatchScore"
requirements_df = populate_text_similarity_score(requirements_df, domain_df, keywords_column_name, output_column_name)

output_column_name = "DataElementsMatchScore"
domain_df = populate_text_similarity_score(domain_df, dataelements_df, keywords_column_name, output_column_name)

output_column_name = "RequirementsMatchScore"
dataelements_df = populate_text_similarity_score(dataelements_df, requirements_df, keywords_column_name, output_column_name)


# # This section will be used to create the Output in excell format

# In[59]:


def extract_action(summary):
    action_string = ""
    count = 1
    for entities in summary['Entities']:
        #print(entities['text'])
        if not entities['text'] in action_string:
                action_string = action_string + entities['text']
                count = count + 1
                if count == 2:
                    count = 1
                    action_string = action_string + ","
    
    #print(action_string)
    return action_string

def lookup_use_case(temp,artifact3_df,column_name):
    #print(artifact3_df.get_value(0,'ID'))
    val = ""
    rowNum = len(artifact3_df.index)
    #print(rowNum)
    for j in range(0,rowNum):
        if temp == artifact3_df.get_value(j,'ID'):
            val = artifact3_df.get_value(j,column_name)
            #print(val)
    
    return val       
        
    
def extract_match(summary,no_of_matches,artifact3_df,column_name):
    match_array_description = []
    match_array_id = []
    for index in range(0,no_of_matches):
        try:
            temp = summary[index]["ID"]
            
        except:
              break
    
            
        temp = summary[index]["ID"]
        #print(temp)
        use_case = lookup_use_case(temp,artifact3_df,column_name)
        
        match_array_id.append(temp)
        #match_array_description.append(use_case + "(" + str(round(summary[index]["cosine_score"], 2)) +")")
        match_array_description.append(use_case)
        #print(use_case)
            
    
    #print("************")
    #print(match_array_id)       
    #print("************")
    return (match_array_description,match_array_id)


        
        
def extract_action_requirements_df(artifact1_df, artifact2_df):
    """ Add text classifier output to the artifact dataframe based on rule defined in config
    """
    for index, row in artifact2_df.iterrows():
        summary = row["ClassifiedText"]
        classifier_journey_output = extract_action(summary)
        artifact1_df.set_value(index, 'Use Case', classifier_journey_output)
    return artifact1_df 

def extract_bestmatch(artifact1_df, artifact2_df, artifact3_df, artifact4_df):
    """ Extract best Match
    """
    No_of_matches_user_function = 2
    No_of_matches_data_elements = 8
    best_match_output_domain_function = []
    best_match_output_dataelement_function = []
    
    for index, row in artifact2_df.iterrows():
        summary = row["DomainMatchScore"]
        #print(summary)
        (best_match_output_domain_function,best_match_output_domain_id) = extract_match(summary, No_of_matches_user_function, artifact3_df,"User Function")
        #print(best_match_output_domain_id)
        artifact1_df.set_value(index, 'Functionality', best_match_output_domain_function)
        print("************")
        for index2 in best_match_output_domain_id:
            #print(index2)
        
            row_domain = len(artifact3_df.index)
            for p in range(0,row_domain):
                if index2 == artifact3_df.get_value(p,'ID'):
                    dataelement_summary = artifact3_df.get_value(p,'DataElementsMatchScore')
                    #print(dataelement_summary)
                    #print("------")
                    (best_match_output_dataelement_function,best_match_output_dataelement_id) = extract_match(dataelement_summary, No_of_matches_data_elements, artifact4_df, "Short")
            
            
            #best_match_output_dataelement_function.append(best_match_output_dataelement_function)
            best_match_output_dataelement_function.extend(best_match_output_dataelement_function)
            print(best_match_output_dataelement_function)
          
        #print(best_match_output_dataelement_function)
        #print("==============")
        #print(index)
        best_match_output_dataelement_function = list(set(best_match_output_dataelement_function))
        artifact1_df.set_value(index, 'Attributes', best_match_output_dataelement_function)
    return artifact1_df 


# In[60]:


import pandas as pd

no_of_rows_brd = len(requirements_df.index)

index = range(0,no_of_rows_brd)
columns = ['User','Use Case', 'Functionality', 'Attributes']


SimMean = pd.DataFrame(index=index, columns=columns)
SimMean.loc[0:no_of_rows_brd,'User'] = requirements_df.loc[0:no_of_rows_brd,'As a <type of user>'].values
SimMean = extract_action_requirements_df(SimMean,requirements_df)
SimMean = extract_bestmatch(SimMean,requirements_df,domain_df,dataelements_df)
#SimMean = extract_bestmatch_domaintodataelem(SimMean,domain_df)
SimMean.get_value(0,"Attributes")
#print(SimMean)

#writer = pd.ExcelWriter('final_output_banking.xlsx', engine='xlsxwriter')
#SimMean.to_excel(writer, sheet_name='Sheet1')
#writer.save()


# #  "*"**********************************************************"
# # Next steps :
# 
# * Populate the correct wording in 3 sheets to provide more accurate and insightful data
# * Use OrientdB to graph the result of cosine
# * Use Node Red to start a UI dashboard
# * Optmize code to reduce memory usage
# * Move the components like Notebook,OrientDB etc to EC2 AWS .

# ** Utility functions to store entities and relations in Orient DB **

# In[25]:


def store_requirements(requirements_df):
    """ Store requirements into the database
    """
    for index, row in requirements_df.iterrows():
        attrs = {}
        reqid = row["ID"]
        attrs["Description"] = row["I want to <perform some task>"].replace('\n', ' ').replace('\r', '')
        attrs["ID"] = reqid
        attrs["User"]= str(row["As a <type of user>"])
        create_record(requirement_classname, reqid, attrs)    
        
def store_domain(domain_df):  
    """ Store domain which has user functions into the database
    """
    for index, row in domain_df.iterrows():
        attrs = {}
        tcaseid = row["ID"]
        attrs["Description"] = row["Description"].replace('\n', ' ').replace('\r', '')
        attrs["ID"] = tcaseid
        attrs["Action"] = str(row["User Function"])
        create_record(domain_classname, tcaseid, attrs)
        
def store_dataelements(dataelements_df):
    """ Store data elements or attributes into the database
    """
    for index, row in dataelements_df.iterrows():
        attrs = {}
        defid = row["ID"]
        attrs["Description"] = row["Description"].replace('\n', ' ').replace('\r', '')
        attrs["ID"] = defid
        attrs["Short"] = str(row["Short"])
        create_record(dataelement_classname, defid, attrs)
        
def store_dataelements_requirement_mapping(dataelements_df):
    """ Store the related requirements for testcases into the database
    """
    for index, row in dataelements_df.iterrows():
        tcaseid = row["ID"]
        requirements = row["RequirementsMatchScore"]
        for requirement in requirements:
            reqid = requirement["ID"]
            attributes = {}
            attributes['score'] = requirement['cosine_score']
            create_dataelements_requirement_edge(tcaseid,reqid, attributes)
            
def store_domain_dataelement_mapping(domain_df):
    """ Store the related dataelement for the domain into the database
    """
    for index, row in domain_df.iterrows():
        domainid = row["ID"]
        dataelements = row["DataElementsMatchScore"]
        count = 0
        #print("---------")
        for dataelement in dataelements:
            
            if count < 4:
                dataelementid = dataelement["ID"]
                attributes = {}
                attributes['score'] = dataelement["cosine_score"]
                create_domain_dataelements_edge(domainid,dataelementid, attributes)
                count = count + 1
            
def store_requirement_domain_mapping(requirements_df):
    """ Store the related domains for the requirements in the database
    """
    for index, row in requirements_df.iterrows():
        count = 0
        reqid = row["ID"]
        functionalities = row["DomainMatchScore"]
        #print("----------") 
        for functionality in functionalities:
            
            if count < 2:  
                functionalityID = functionality["ID"]
                cosine_score =  functionality["cosine_score"]
                attributes = {}
                attributes['score'] = cosine_score
                create_requirement_domain_edge(reqid, functionalityID, attributes)
                count = count + 1
                
            


# ** Store artifacts data and relations into OrientDB **
# * Drop and create a database
# * Create classes for each category of artifact
# * Store artifact data
# * Store artifact relations data

# In[26]:


drop_database("SoftwareDesignAI")
create_database("SoftwareDesignAI", "admin", "admin")

requirement_classname = "Requirements"
domain_classname = "Domains"
dataelement_classname = "DataElements"

create_class(requirement_classname)
create_class(domain_classname)
create_class(dataelement_classname)



store_requirements(requirements_df)
store_dataelements(dataelements_df)
store_domain(domain_df)


store_requirement_domain_mapping(requirements_df)
store_domain_dataelement_mapping(domain_df)
store_dataelements_requirement_mapping(dataelements_df)


# # 7. Transform results for Visualization

# In[27]:


def get_artifacts_mapping_d3_tree(defectId):
    """ Create an artifacts mapping json for display by d3js tree widget
    """
    depTree = {}
    depTree['ID'] = defectId
    testcases = get_related_testcases(defectId)
    
    depTree['children'] = []
    i=1
    for key in testcases:
        #print key,testcases[key]
        testcaseChildren = {}
        testcaseChildren['ID'] = key
        testcaseChildren['Score'] = testcases[key]
        testcaseChildren['children'] = []
        depTree['children'].append(testcaseChildren)
        requirements = get_related_requirements(key)
        
        for key in requirements:
            requirementChildren = {}
            requirementChildren['ID']=key
            requirementChildren['Score']=requirements[key]
            testcaseChildren['children'].append(requirementChildren)
    return depTree 

def get_artifacts_mapping_d3_network(reqid):
    """ Create an artifacts mapping json for display by d3js network widget
    """
    nodes =[]
    links =[] 
    req = {}
    req['id'] = reqid
    req['group'] = 1
    nodes.append(req)
    
    domaincases = get_related_domaincases(reqid)
    
    for key in domaincases:
        domaincase ={}
        domaincaseid = key
        domaincase['id'] = domaincaseid
        domaincase['group'] = 2
        if domaincase not in nodes:
            nodes.append(domaincase)
        link = {}
        link['source'] = reqid
        link['target']=domaincaseid
        link['value']=domaincases[domaincaseid]
        links.append(link)
        dataelements = get_related_dataelements(key)
        for key in dataelements:
            dataelement ={}
            dataelement['id'] = key
            dataelement['group'] = 3
            if dataelement not in nodes:
                nodes.append(dataelement)
            link = {}
            link['source'] = domaincaseid
            link['target'] = key
            link['value'] = dataelements[key]
            links.append(link)
            
    result ={}
    result["nodes"] = nodes
    result["links"] = links
    return result

def get_tc_req_mapping_d3_network(testcaseid):
    """ Create a testcases to requirement mapping json for display by d3js network widget
    """
    nodes =[]
    links =[] 
    testcase = {}
    testcase['id'] = testcaseid
    testcase['group'] = 2
    nodes.append(testcase)
    requirements = get_related_requirements(testcaseid)
    for key in requirements:            
        requirement ={}
        requirement['id'] = key
        requirement['group'] = 3
        nodes.append(requirement)
            
        link = {}
        link['source'] = testcaseid
        link['target'] = key
        link['value'] = requirements[key]
        links.append(link)
    result ={}
    result["nodes"] = nodes
    result["links"] = links
    return result

def transform_defects_d3_bubble(defects):
    """ Transform the defects list output to a json for display by d3js bubble chart"""
    defectsList = {}
    defectsList['name'] = "defect"
    children = []
    for defect in defects:
        detail = {}
        sizeList = [400,230,130]
        detail["ID"] = defect['ID']
        severity = int(defect['Severity'])
        detail["group"] = str(severity)
        detail["size"] = sizeList[severity-1]
        children.append(detail)
    defectsList['children'] = children 
    return defectsList

def transform_testcases_d3_bubble(testcases):
    """ Transform the testcases list output to a json for display by d3js bubble chart"""
    testcasesList = {}
    testcasesList['name'] = "test"
    sizeList = {}
    sizeList["FVT"]=200
    sizeList["TVT"]=110
    sizeList["SVT"]=400
    children = []
    for testcase in testcases:
        detail = {}
        detail["ID"] = testcase['ID']
        detail["group"] = testcase['Category']
        detail["size"]= sizeList[testcase['Category']]
        children.append(detail)
    testcasesList['children'] = children 
    return testcasesList

def transform_requirements_d3_bubble(requirements):
    """ Transform the requirements list output to a json for display by d3js bubble chart"""
    requirementsList = {}
    requirementsList['name'] = "requirement"
    sizeList = {}
    sizeList[1]=300
    sizeList[2]=100
    sizeList[3]=75
    children = []
    for requirement in requirements:
        detail = {}
        size = 0
        detail["ID"] = requirement['ID']
        detail["group"] = requirement['User']
        if requirement['User'] == 'Customer':
            size = 2
        elif requirement['User'] == 'Banker':
            size = 3
        else:
            size = 1
        detail["size"]= sizeList[size]
        if 'defectcount' in requirement:
            detail['defectcount'] = requirement['defectcount']
        children.append(detail)
    requirementsList['children'] = children 
    return requirementsList

def merge_apply_filters_d3_bubble(mainList, filterList):
    """ Add a filter attribute to the list elements for processing on UI
    """
    mainListChildren = mainList['children']
    filterListChildren = filterList['children']
    for child in mainListChildren:
        child['filter'] = 0
        for child1 in filterListChildren:
            if ( child['ID'] == child1['ID']):
                child['filter'] = 1
    return mainList  

def setup_download_excel():
    """ Transform the requirements list output to a json for display by d3js bubble chart"""
    
    requirementsList = {}
    requirementsList['name'] = "download"
    children = []
    detail = {}
    detail["filename"] = "final_output.xlsx"
    detail["path"] = "/Users/swaroopmishra/Desktop/pythonscript/NLP_Project/nWave_softwareDesign/notebook/"
    children.append(detail)
    detail = {}
    detail["filename"] = "excel_download.jpg"
    detail["path"] = "/Users/swaroopmishra/Desktop/pythonscript/NLP_Project/nWave_softwareDesign/notebook/"
    children.append(detail)
    requirementsList['children'] = children 
    return requirementsList


# # The following snippet is for temporary purpose and will be used to check server side programing

# In[28]:


def build_format_requirements_list(requirementsResult):
    """ Build and format the OrientDB query results for requirements
    """
    requirements = []
    for requirement in requirementsResult:
        detail = {}
        detail['ID'] =requirement.ID
        detail['Description'] = requirement.Description
        detail['User'] = requirement.User
        requirements.append(detail)
    return requirements  

def get_requirements():
    """ Get all requirements
    """
    requirementsQuery = "select * from Requirements"
    requirementsResult =  execute_query(requirementsQuery)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements  
def transform_requirements_d3_bubble(requirements):
    """ Transform the requirements list output to a json for display by d3js bubble chart"""
    requirementsList = {}
    requirementsList['name'] = "requirement"
    sizeList = {}
    sizeList[1]=300
    sizeList[2]=100
    sizeList[3]=75
    children = []
    for requirement in requirements:
        detail = {}
        size = 0
        detail["ID"] = requirement['ID']
        detail["group"] = requirement['User']
        if requirement['User'] == 'Customer':
            size = 2
        elif requirement['User'] == 'Banker':
            size = 3
        else:
            size = 1
        detail["size"]= sizeList[size]
        if 'defectcount' in requirement:
            detail['defectcount'] = requirement['defectcount']
        children.append(detail)
    requirementsList['children'] = children 
    return requirementsList


#wsresponse = {}
#wsresponse["forCmd"] = "ReqsList"
#requirements = get_requirements()
#wsresponse["response"] = transform_requirements_d3_bubble(requirements)
#print(json.dumps(wsresponse, indent=2))



def setup_download_excel():
    """ Transform the requirements list output to a json for display by d3js bubble chart"""
    
    requirementsList = {}
    requirementsList['name'] = "download"
    children = []
    detail = {}
    detail["filename"] = "final_output.xlsx"
    detail["path"] = "/Users/swaroopmishra/Desktop/pythonscript/NLP_Project/nWave_softwareDesign/notebook/"
    children.append(detail)
    detail = {}
    detail["filename"] = "excel_download.jpg"
    detail["path"] = "/Users/swaroopmishra/Desktop/pythonscript/NLP_Project/nWave_softwareDesign/notebook/"
    children.append(detail)
    requirementsList['children'] = children 
    return requirementsList

wsresponse = {}
wsresponse["forCmd"] = "GetExcel"
wsresponse["response"] = setup_download_excel()

print(json.dumps(wsresponse, indent=2))


# # 8. Expose integration point with a websocket client

# In[29]:


def get_related_domaincases(reqid):
    """ Get the related domaincases for a requirements
    """
    domaincasesQuery = "select * from ( select expand( out('linkeddomains')) from Requirements where ID = '" + reqid +"' )"
    domaincases = execute_query(domaincasesQuery)
    scoresQuery = "select expand(out_linkeddomains) from Requirements where ID = '"+reqid+"'"
    scores = execute_query(scoresQuery)
    domaincaseList =[]
    domaincaseAction =[]
    scoresList= []
    for domaincase in domaincases:
        domaincaseList.append(domaincase.ID)
        domaincaseAction.append(domaincase.Action)
    for score in scores:
        scoresList.append(score.score)
    result = {}
    length = len(domaincaseList)
    for i in range(0, length):
        result[domaincaseList[i]] = scoresList[i]
        #result[domaincaseList[i]] = domaincaseAction[i]
    print(result)
    return result

def get_related_dataelements(domaincaseid):
    """ Get the related requirements for a testcase
    """
    dataelementsQuery = "select * from ( select expand( out('linkeddataelements') ) from Domains where ID = '" + domaincaseid +"' )"
    dataelements = execute_query(dataelementsQuery)
    #print(dataelements)
    scoresQuery = "select expand(out_linkeddataelements) from Domains where ID = '"+domaincaseid+"'"
    scores = execute_query(scoresQuery)
    dataelementsList =[]
    scoresList= []
    for dataelement in dataelements:
        dataelementsList.append(dataelement.ID)
    for score in scores:
        scoresList.append(score.score)
    result = {}
    length = len(dataelementsList)
    #print requirementsList, scoresList
    for i in range(0, length):
        result[dataelementsList[i]] = scoresList[i]
    return result

def get_related_user(reqid):
    reqQuery = "select * from Requirements where ID = '"+reqid+"'"
    requirements = execute_query(reqQuery)
    result = ''
    for requirement in requirements:
        print(requirement.User)
        result = requirement.User
    return result

def get_related_action(key):
    domQuery = "select * from Domains where ID = '"+key+"'"
    domains = execute_query(domQuery)
    result = ''
    for domain in domains:
        print(domain.Action)
        result = domain.Action
    return result

def get_related_shorthand(key):
    shortQuery = "select * from DataElements where ID = '"+key+"'"
    shorts = execute_query(shortQuery)
    result = ''
    for short in shorts:
        print(short.Short)
        result = short.Short
    return result

def get_artifacts_mapping_d3_network(reqid):
    """ Create an artifacts mapping json for display by d3js network widget
    """
    nodes =[]
    links =[] 
    req = {}
    req['id'] = reqid
    req['group'] = 1
    req['desc'] = get_related_user(reqid)
    nodes.append(req)
    
    domaincases = get_related_domaincases(reqid)
    print("1 - nodes")
    print(nodes)
    print("2 - domaincases")
    print(domaincases)
    for key in domaincases:
        print("3  - For each domaincases")
        print(key)
        domaincase ={}
        domaincaseid = key
        domaincase['id'] = domaincaseid
        domaincase['group'] = 2
        domaincase['desc'] = get_related_action(key)
        if domaincase not in nodes:
            nodes.append(domaincase)
        print("4 - appended node")
        print(nodes)
        link = {}
        link['source'] = reqid
        link['target']=domaincaseid
        link['value']=domaincases[domaincaseid]
        links.append(link)
        print("5 - create links for individual domaincase")
        print(links)
        dataelements = get_related_dataelements(key)
        print("6 - for the domain case find dataelements")
        print(dataelements)
        for key in dataelements:
            print("7- individual dataelements")
            print(key)
            dataelement ={}
            dataelement['id'] = key
            dataelement['group'] = 3
            dataelement['desc'] = get_related_shorthand(key)
            if dataelement not in nodes:
                nodes.append(dataelement)
            print("8 - appended node with dataelement")
            print(nodes)
            link = {}
            link['source'] = domaincaseid
            link['target'] = key
            link['value'] = dataelements[key]
            links.append(link)
            print("9- create links for dataelements")
            print(links)
    result ={}
    result["nodes"] = nodes
    result["links"] = links
    return result


req_id = "R01"
wsresponse = {}
wsresponse["forCmd"] = "AllRelation" 
wsresponse["response"] = get_artifacts_mapping_d3_network(req_id)
print("***** final response ******")
print(json.dumps(wsresponse))


# In[30]:


def get_requirement_defects(numdefects):
    """ Get requirements that have more than a given number of defects
    """
    query = "select ID,Description,Priority from Requirement where out('linkeddefects').size() >= " + str(numdefects)
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    for requirement in requirements:
        num = len(get_related_defects(requirement['ID']))
        requirement['defectcount'] = num
    return requirements 
def merge_apply_filters_d3_bubble(mainList, filterList):
    """ Add a filter attribute to the list elements for processing on UI
    """
    mainListChildren = mainList['children']
    filterListChildren = filterList['children']
    for child in mainListChildren:
        child['filter'] = 0
        for child1 in filterListChildren:
            if ( child['ID'] == child1['ID']):
                child['filter'] = 1
    return mainList

def get_requirement_defects(numdefects):
    """ Get requirements that have more than a given number of defects
    """
    query = "select ID,Description,Priority from Requirement where out('linkeddefects').size() >= " + str(numdefects)
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    for requirement in requirements:
        num = len(get_related_defects(requirement['ID']))
        requirement['defectcount'] = num
    return requirements 

def get_requirements_banker():
    """ Get requirements that have no defects
    """
    query = "Select * from Requirements where User = 'Banker'"
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements
def get_requirements_customer():
    """ Get requirements that have no defects
    """
    query = "Select * from Requirements where User = 'Customer'"
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements
def get_requirement_domain(numde):
    """ Get requirements that have more than a given number of defects
    """
    query = "select ID,Description,User from Requirements where out('linkeddomains').size() <= " + str(numde)
    requirementsResult =  execute_query(query)
    requirements = build_format_requirements_list(requirementsResult)
    return requirements 

insight_id = 'Insight1 Get requirements for Banker'
requirements = get_requirements()
requirements = transform_requirements_d3_bubble(requirements)
print(requirements)
if (insight_id.find('Insight1') != -1):
    print('Insight1')
    #req = get_requirements_zero_defect()
    req = get_requirements_banker()
    req = transform_requirements_d3_bubble(req)
    print('******only banker********')
    print(req)
    response = merge_apply_filters_d3_bubble(requirements, req)
    print('******* Applying filter *****')
    print(response)
    print('********')
if (insight_id.find('Insight2') != -1):
    req = get_requirements_customer()
    req = transform_requirements_d3_bubble(req)
    response = merge_apply_filters_d3_bubble(requirements, req)
if (insight_id.find('Insight3') != -1):
    #req = get_requirement_defects(5)
    req = get_requirement_dataelements(5)
    req = transform_requirements_d3_bubble(req)
    response = merge_apply_filters_d3_bubble(requirements, req)
wsresponse = {}
wsresponse["forCmd"] = "Insight" 
wsresponse["response"] = response
print(json.dumps(wsresponse))


# In[31]:


def on_message(ws, message):
    print(message)
    msg = json.loads(message)
    print("message",msg)
    cmd = msg['cmd']
    
    print("Command :", cmd)

    if cmd == 'getExcel':
        wsresponse = {}
        wsresponse["forCmd"] = "GetExcel"
        wsresponse["response"] = setup_download_excel()
        ws.send(json.dumps(wsresponse))
    
    if cmd == 'ReqsList':
        wsresponse = {}
        wsresponse["forCmd"] = "ReqsList"
        requirements = get_requirements()
        wsresponse["response"] = transform_requirements_d3_bubble(requirements)
        ws.send(json.dumps(wsresponse))

    if cmd == 'AllRelation':
        req_id = msg['ID']
        wsresponse = {}
        wsresponse["forCmd"] = "AllRelation" 
        wsresponse["response"] = get_artifacts_mapping_d3_network(req_id)
        ws.send(json.dumps(wsresponse))

    if cmd == 'DataElementRelation':
        testcase_id = msg['ID']
        wsresponse = {}
        wsresponse["forCmd"] = "TestcaseRelation" 
        wsresponse["response"] = get_tc_req_mapping_d3_network(testcase_id)
        ws.send(json.dumps(wsresponse))

    #  the below protion is for getting insight into download section, this has not yet been done and will be done later if required
    
    if cmd == 'DownloadInsight':
        insight_id = msg['ID']
        testcases = get_testcases()
        testcases = transform_testcases_d3_bubble(testcases)
        if (insight_id.find('Insight1') != -1):
            fvtTests = get_testcases_category('FVT')
            fvtTests = transform_testcases_d3_bubble(fvtTests)
            response = merge_apply_filters_d3_bubble(testcases, fvtTests)
        if (insight_id.find('Insight2') != -1):
            svtTests = get_testcases_category('SVT')
            svtTests = transform_testcases_d3_bubble(svtTests)
            response = merge_apply_filters_d3_bubble(testcases, svtTests)
        if (insight_id.find('Insight3') != -1):
            tvtTests = get_testcases_category('TVT')
            tvtTests = transform_testcases_d3_bubble(tvtTests)
            response = merge_apply_filters_d3_bubble(testcases, tvtTests)
        if (insight_id.find('Insight4') != -1):
            testcase_zero_defect = get_testcases_zero_defects()
            testcase_zero_defect = transform_testcases_d3_bubble(testcase_zero_defect)
            response = merge_apply_filters_d3_bubble(testcases, testcase_zero_defect)
        wsresponse = {}
        wsresponse["forCmd"] = "Insight" 
        wsresponse["response"] = response
        ws.send(json.dumps(wsresponse))

    if cmd == 'ReqInsight':
        insight_id = msg['ID']
        requirements = get_requirements()
        requirements = transform_requirements_d3_bubble(requirements)
    #print(requirements)
        if (insight_id.find('Insight1') != -1):
            print('Insight1')
            req = get_requirements_banker()
            req = transform_requirements_d3_bubble(req)
        #print('******only banker********')
        #print(req)
            response = merge_apply_filters_d3_bubble(requirements, req)
        #print('******* Applying filter *****')
            #print(response)
            #print('********')
        if (insight_id.find('Insight2') != -1):
            req = get_requirements_customer()
            req = transform_requirements_d3_bubble(req)
            response = merge_apply_filters_d3_bubble(requirements, req)
        if (insight_id.find('Insight3') != -1):
            req = get_requirement_domain(1)
            req = transform_requirements_d3_bubble(req)
            response = merge_apply_filters_d3_bubble(requirements, req)

    wsresponse = {}
    wsresponse["forCmd"] = "Insight" 
    wsresponse["response"] = response
    ws.send(json.dumps(wsresponse)) 

def on_error(ws, error):
    print(error)

def on_close(ws):
    print ("DSX Listen End")
    ws.send("DSX Listen End")

def on_open(ws):
    def run(*args):
        for i in range(10000):
            hbeat = '{"cmd":"AI nWave HeartBeat"}'
            ws.send(hbeat)
            time.sleep(100)
            
    _thread.start_new_thread(run, ())


def start_websocket_listener():
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:1880/ws/software",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()


# ## 8.1 Start websocket client

# In[32]:


#start_websocket_listener()

