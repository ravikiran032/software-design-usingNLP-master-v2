
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
from nltk.tokenize import punkt
from nltk.tokenize.punkt import PunktSentenceTokenizer


# # 6. Data Preparation

# ## 6.1 Global variables and functions

# In[1]:


# Name of the excel file with data in S3 Storage
#BrdFileName = "Banking-BRD.xlsx"
# Choose or get as an input as to which Domain it belongs to i.e banking, healthcare etc
Domain = "Banking"

# Name of the config files in Object Storage. Rule_brd will be used specifically for parsing requirement document
configFileName = "config/sample_config.txt"
BRD_configFileName = "config/Rule_BRD.txt"
# Config contents
config = None;

Path = "D:/machine learning/software-design-usingNLP-master v2/"
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

#grammer = """Ravi:{<VB.?>+(<TO>|<DT>|<PRP.?>|<IN>|<JJ>)*<NN.?|NNPS>+}""" 




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
#function not used    
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
    # global grammer
    # parsed_chink = nltk.RegexpParser(grammer)
    # parsed_chink.parse(text).draw()
    
    parsed_cp = nltk.RegexpParser(chunk)
    pos_cp = parsed_cp.parse(text)
    #pos_cp.draw()
    
    #pos_cp = chunk_sentence(text) #*** use this for getting refined output after chinking but extra entities in output
    chunk_list=[]
    for root in pos_cp:
        if isinstance(root, nltk.tree.Tree):               
            if root.label() == tag:
                chunk_word = ''
                for child_root in root:
                    chunk_word = chunk_word +' '+ child_root[0]
                chunk_list.append(chunk_word)
    #print("txt: ",text)
    #print(chunk_list)
    return chunk_list

    
def augument_SpResponse(responsejson,updateType,text,tag): # update classified text and tag in entities of response json
    """ Update the output JSON with augumented classifications.
    """
   # print("augument_response  response: "+str(responsejson)+"updateType: "+updateType+"text or words: "+text+"tag in rule brd: "+tag )
    if(updateType == 'keyword'):
        if not any(d.get('text', None) == text for d in responsejson['Keywords']):
            responsejson['Keywords'].append({"User":text})
    else:
        if not any(d.get('text', None) == text for d in responsejson['Entities']) :
            
            responsejson['Entities'].append({"type":tag,"text":text}) 
            
    #print(responsejson)
    return responsejson

def classify_BRD_text(text, config, DOC_TYPE):
    """ Perform augumented classification of the text for BRD specifically for getting the output with action.
    """
    
    #will be used for storing initial value of response json, this is from nlu earlier
    with open(Path+'config/output_format_BRD.json') as f:
        responsejson = json.load(f)

        tokens = split_into_tokens(text)

        postags = POS_tagging(tokens)
        #print("POS tags for sentence "+str(tokens)+"  is "+ str(postags))
        configjson = json.loads(config)#Rule_BRD.txt
    
        no_of_items = 0
   
    
    for step in configjson['configuration']['classification']['stages']['steps']:
            #print('Stage - Performing ' + str(step))
                   
            if(step['type'] == 'chunking'):
                for chunk in step['chunk']:
                    if(chunk["tag"]==DOC_TYPE):
                        tag=chunk["tag"]
                        chunktags = BRD_chunk_tagging('ACTION',chunk['pattern'],postags) #overrite happens if there are two chunks of same tags
                        if(len(chunktags)>0):
                            for words in chunktags:
                                responsejson= augument_SpResponse(responsejson,'ACTION',words,tag)

                            
            else:
                print('UNKNOWN STEP')
    
    
    
    return responsejson



stopWords = get_stop_words('english')
# List of words to be ignored for text similarity
stopWords.extend(["The","This","That",".","!","?"])

def compute_text_similarity(text1, text2, text1tags, text2tags):
    """ Compute text similarity using cosine
    """
    #stemming is the process for reducing inflected (or sometimes derived) words to their stem, base or root form
    tokens_text1 = []
    tokens_text2 = []
    stemmer = nltk.stem.porter.PorterStemmer()#.WordNetLemmatizer()
    '''
    sentences_text1 = split_sentences(text1)
    sentences_text2 = split_sentences(text2)

    #print("sentence 1",sentences_text1)
    #print("sentence 2",sentences_text2)
    
    #for tags in text1tags:
        #pass

    
    for element in text1tags:
        tokens_text1.extend(split_into_tokens(element))
    for element in text2tags:
        tokens_text2.extend(split_into_tokens(element))
    
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
        '''
    for element in text1tags:
        tokens_text1.extend(split_into_tokens(element))
    for element in text2tags:
        tokens_text2.extend(split_into_tokens(element))
        
       

    
    

    tokens1Filtered = [stemmer.stem(x)  for x in tokens_text1 if x not in stopWords]
    
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
    '''
    with open(Path+"data/cosinesimilarity.txt","a") as fp:
        fp.write(str(tokens1Filtered))
        fp.write("\n")
        fp.write("                           -------vs---------                 ")
        fp.write("\n")
        fp.write(str(tokens2Filtered))
        fp.write("\n")
        fp.write(str(cosine_similarity))
        fp.write("\n")
        fp.write("\n")'''
        
    return cosine_similarity

def get_file(Path):
    dummpyfunction="123"
    


def load_artifacts():
    global requirements_df 
    global domain_df 
    global dataelements_df 
    global config
    global BRD_config
    global Path
    
    
    Location = "".join((Path,requirements_file_name))
    #get_file(requirements_file_name,Location)
    excel = pd.ExcelFile(Location)
    requirements_df = excel.parse(requirements_sheet_name)
    Location = "".join((Path,domain_file_name))
    #get_file(domain_file_name,Location)
    excel = pd.ExcelFile(Location)
    domain_df = excel.parse(domain_sheet_name)
    Location = "".join((Path,dataelements_file_name))
    #get_file(dataelements_file_name,Location)
    excel = pd.ExcelFile(Location)
    dataelements_df = excel.parse(dataelements_sheet_name)
    rule_text = open(Path+configFileName)
    config = rule_text.read()
    BRD_rule_text = open(Path+BRD_configFileName)
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
    

    
def mod_req_text_classifier_output(artifact_df, BRD_config, output_column_name,DOC_TYPE):
    """ Add text classifier output to the artifact dataframe based on rule defined in config
    """
    for index, row in artifact_df.iterrows():
        summary = row["I want to <perform some task>"]
        modID = row["ID"]
        
       # print(modID)
        modID = modID.replace("R","UC")
        user = row["As a <type of user>"]
        user = "".join((user," want to "))
        summary = "".join((user,summary))
        #print("--------------")
        #print(summary)
        classifier_journey_output = classify_BRD_text(summary, BRD_config,DOC_TYPE)
        #print("classifieer ourney out",classifier_journey_output)
        artifact_df.set_value(index, output_column_name, classifier_journey_output)
        
       
    return artifact_df 


def add_text_classifier_output(artifact_df, config, output_column_name, DOC_TYPE):
    """ Add text classifier output to the artifact dataframe based on rule defined in config
    """
    for index, row in artifact_df.iterrows():
        summary = row["Description"]
        #print("--------------")
        #print(summary)
        classifier_journey_output = classify_BRD_text(summary, BRD_config,DOC_TYPE)
        #print(classifier_journey_output)
        artifact_df.set_value(index, output_column_name, classifier_journey_output)
    return artifact_df 
           
def add_keywords_entities(artifact_df, classify_text_column_name, output_column_name):
    """ Add keywords and entities to the artifact dataframe"""
    for index, artifact in artifact_df.iterrows():
        keywords_array = []
        for row in artifact[classify_text_column_name]['Keywords']:
            #print("add key word entities: classifiedtext[keywords] ",row)
            if not row['User'] in keywords_array and row['User']!="":
                #print("add key word entities: classifiedtext[entities] ",keywords_array)
                keywords_array.append(row['User'])
                
        for entities in artifact[classify_text_column_name]['Entities']:
            
            if not entities['text'] in keywords_array  and entities['text']!="":
                
                keywords_array.append(entities['text'])

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
                #artifact1[\'Description\'], 
                #artifact2[\'Description\'], 
                artifact1[heading1],
                artifact2[heading2],
                artifact1['Keywords'], 
                artifact2['Keywords'])
            matches[index2]["cosine_score"] = cosine_score
       
        sorted_obj = sorted(matches, key=lambda x : x['cosine_score'], reverse=True)
    
    # This is where the lower cosine value to be truncated is set and needs to be adjusted based on output
    
        for obj in sorted_obj:
            if obj['cosine_score'] > 0.55:
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

DOC_TYPE = ['REQ_ACTION','DOMAIN_ACTION','DE_ACTION']
output_column_name = "ClassifiedText"
requirements_df = mod_req_text_classifier_output(requirements_df, BRD_config, output_column_name,DOC_TYPE[0])

domain_df = add_text_classifier_output(domain_df,BRD_config, output_column_name,DOC_TYPE[1])
dataelements_df = add_text_classifier_output(dataelements_df,BRD_config, output_column_name,DOC_TYPE[2])


'''temp_df=pd.DataFrame(data=None,columns=requirements_df.columns,index=requirements_df.index)
    #print("lenght ",len(action_keywords))
#print(temp_df)    
for index,row in requirements_df.iterrows():
    #print("row in requirents after keywords: ",row)
    #action_keywords =   [entity for entity in row.loc['ClassifiedText']['Entities'] if entity['text']!=""]
    
    if row.loc['ClassifiedText']['Entities'][0]['text']==""  :
        del row.loc['ClassifiedText']['Entities'][0]
    
    number_of_actions=len(row.loc['ClassifiedText']['Entities'])
    
    if number_of_actions>1:
        print(row.loc['ClassifiedText']['Entities'])
        for i,entity in enumerate(row.loc['ClassifiedText']['Entities']):
            record = requirements_df.loc[index,:]
            print("i values ",i)
            print("row recor",row['ClassifiedText']['Entities'])
            for j,rec in enumerate(record.loc['ClassifiedText']['Entities']):
                print("j values : ",j)
                if j!=i:
                    print("j:",j)
                    print("i:",i)
                    print("record entieis ",record.loc['ClassifiedText']['Entities'])
                    #del record.loc['ClassifiedText']['Entities'][j]
            print("modified records",record['ClassifiedText']['Entities'])

            
            
            #temp_df.append(row, ignore_index=True)
            #print("if cond",temp_df)
            #print("row",row)
    else:
        temp_df.append(row, ignore_index=True)
        #print("temp ",temp_df)'''


        
        
    
#requirements_df= temp_df

            


    
    #print("entities in ",action_keywords)
    
    
    
    



# ** Populate keywords and entities **
# * Add the keywords and entities extracted from the unstructured text to the artifact dataframes

# In[21]:


classify_text_column_name = "ClassifiedText"
output_column_name = "Keywords"



requirements_df = add_keywords_entities(requirements_df, classify_text_column_name, output_column_name)
domain_df = add_keywords_entities(domain_df, classify_text_column_name, output_column_name)
dataelements_df = add_keywords_entities(dataelements_df, classify_text_column_name, output_column_name)





for index,row in requirements_df.iterrows():
    #print("row in requirents after keywords: ",row.loc['Keywords'])
    action_keywords = row.loc['Keywords']
    #classified_text = row.loc['ClassifiedText']
    
    if len(action_keywords)>1:
        # print(action_keywords)
        # print("index ",index)
        #print("upto index ",requirements_df.loc[0:index-1,'Keywords'])
        #print(" index ",requirements_df.loc[index,'Keywords'])
        #print("after index ",requirements_df.loc[index+1:,'Keywords'])
        pre_index_df=requirements_df.loc[0:index-1,:]
        post_index_df=requirements_df.loc[index+1:,:]
        for action in action_keywords:
            temp=[]
            line = requirements_df.loc[index,:]
            line['Keywords']=eval('["'+action+'"]')
           
            pre_index_df=pre_index_df.append(line,ignore_index=False)
            #print("modified df ", pre_index_df.append(line))
            #i=i+1
        requirements_df = pre_index_df.append(post_index_df,ignore_index=True)


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

writer = pd.ExcelWriter(Path+'data/keywords_Output.xlsx')
requirements_df.to_excel(writer, sheet_name='Sheet1')
domain_df.to_excel(writer, sheet_name='Sheet2')
dataelements_df.to_excel(writer, sheet_name='Sheet3')
writer.save()

# # This section will be used to create the Output in excell format

# In[59]:


def extract_action(summary):
    for entities in summary:
        return entities

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
        summary = row.loc["Keywords"]
        #print("summary ",summary)
        classifier_journey_output = extract_action(summary)
        #print("classifer out",classifier_journey_output)
        artifact1_df.set_value(index, 'Use Case', classifier_journey_output)
        
    return artifact1_df 

def extract_bestmatch(artifact1_df, artifact2_df, artifact3_df, artifact4_df):
    """ Extract best Match
    """
    No_of_matches_user_function = 2
    No_of_matches_data_elements = 5
    best_match_output_domain_function = []
    best_match_output_dataelement_function = []
    print("-------------------------")
     
    for index, row in artifact2_df.iterrows():
        temp1=""
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
            temp1 = temp1 +','+ str(best_match_output_dataelement_function)
            #print("best elemenets ",temp1)
        
        

        artifact1_df.set_value(index, 'Attributes', temp1)
        
    return artifact1_df 


# In[60]:


import pandas as pd

no_of_rows_brd = len(requirements_df.index)

index = range(0,no_of_rows_brd)
columns = ['ID','User','Use Case', 'Functionality', 'Attributes']


SimMean = pd.DataFrame(index=requirements_df.index, columns=columns)# change to requirements index for getting index from intermediate output
SimMean.loc[0:no_of_rows_brd,'ID'] = requirements_df.loc[0:no_of_rows_brd,'ID'].values
SimMean.loc[0:no_of_rows_brd,'User'] = requirements_df.loc[0:no_of_rows_brd,'As a <type of user>'].values

SimMean = extract_action_requirements_df(SimMean,requirements_df)
SimMean = extract_bestmatch(SimMean,requirements_df,domain_df,dataelements_df)

temp_df=SimMean[SimMean["ID"].duplicated(keep=False)]
merged_df=temp_df.astype(str).groupby(temp_df.ID, as_index=False).agg(','.join)
temp_df=SimMean.drop(temp_df.index,axis=0).append(merged_df,ignore_index=True).sort_values("ID").drop(columns="ID")
SimMean = temp_df
SimMean['User']=temp_df['User'].str.split(',').apply(set).str.join("")     

writer = pd.ExcelWriter(Path+'data/final_output_banking_2.xlsx')
SimMean.to_excel(writer, sheet_name='Sheet1',index=False)
writer.save()