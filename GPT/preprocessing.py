## Loading and preprocessing data using our Master's class Discord chat. 
## The data is in a csv file with the following columns: Author, AuthorID, Date, Content, Attachments, Reactions
## The data is preprocessed by removing the AuthorID, Date, Attachments and Reactions columns, replacing the new line character with a space and removing the rows with nan values in the Content column.

## To run, add folder GPT\inputs to your project and add the csv file with the data to the folder. 
## The csv file should be named input.csv and should be in the same folder as preprocessing.py [THIS WILL BE FIXED]
## The preprocessed data will be saved in the same folder as preprocessing.py as input_preprocessed.txt

## ToDo: 
## - Figure a way to randomize reactions and attachments
## - Understand best practice for preprocessing text data for LLMs
## - Possibly implement this when users chat with Discord bot

import pandas as pd
import re
##import nltk
##nltk.download('punkt')

corpus_only = True
data_path = 'GPT\inputs\input.csv'

def save_to_txt(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(data)

class Preprocessing:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        data = pd.read_csv(self.data_path)
        print(f'Data loaded with {len(data.index)}!')
        return data

## Should I remove Author to create a single bot first, or work on a multi-agent system?
    def preprocess_data(self, data):
        data[['Author','Content','Attachments']] = data[['Author','Content','Attachments']].astype(str)
        data = data.drop(['AuthorID', 'Date', 'Attachments','Reactions'], axis=1) ## Figure a way to randomize reactions and attachments
        data['Content'] = data['Content'].apply(lambda x: x.replace('\n', ''))
        data = data[~data.Content.str.contains("nan")]
        data['Content'] = data['Content'].apply(lambda x: x if isinstance(x,str) else '')
        print(f'Data preprocessed with {len(data.index)}!')
        print(data.head(5))
        return data

## To remove all authors, use this function instead of preprocess_data
    def corpus_only_data(self, data):
        data = data.drop(['AuthorID', 'Author', 'Date', 'Attachments','Reactions','Attachments'], axis=1) ## Figure a way to randomize reactions and attachments
        data[['Content']] = data[['Content']].astype(str)
        data['Content'] = data['Content'].apply(lambda x: x.replace('\n', ''))
        data = data[~data.Content.str.contains("nan")]
        data['Content'] = data['Content'].apply(lambda x: x if isinstance(x,str) else '')
        print(f'Data preprocessed with {len(data.index)}!')
        print(data.head(5))
        return data
    
    def remove_urls(self, data):
        if not isinstance(data, str):
            return ''
        return re.sub(r'http[s]?://\S+', '', data)
        
    def remove_emojis(self, data):
        if not isinstance(data, str):
            return ''
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', data)
    
    def remove_special_characters(self, data):
        if not isinstance(data, str):
            return ''
        return re.sub(r'[^A-Za-z0-9\s.,?!]', '', data)

    ##Don't need but good practice ##
    def tokenize_sentences(self, data):
        return nltk.sent_tokenize(data)
    
    def tokenize_words(self, data):
        return nltk.word_tokenize(data)
    ## END OF DON'T NEED ##

    def drop_blank_rows(self, data):
        data = data[data['Content'].str.strip() != '']
        return data
    
    def to_lowercase(self, data):
        return data.lower()
    
    def save_data(self, data):
        if corpus_only:
            data.to_csv('GPT\inputs\corpus_only.txt', index=None, header=None, sep='\t')
        else:
            data.to_csv('GPT\inputs\input_preprocessed.txt', index=None, header=None, sep='\t')
    
    ## For saving to txt file. Maybe use this instead for saving and have option to pick which one##
    def save_to_txt(data, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(data)

if __name__ == '__main__':
    preprocessing = Preprocessing(data_path)
    data = preprocessing.load_data()
    # if corpus_only:
    #     data = preprocessing.corpus_only_data(data)
    # else:
    #     data = preprocessing.preprocess_data(data)
    data['Content'] = data['Content'].apply(preprocessing.remove_urls)
    data['Content'] = data['Content'].apply(preprocessing.remove_emojis)
    data['Content'] = data['Content'].apply(preprocessing.remove_special_characters)
    data = data[data['Content'].str.strip() != '']
    data = preprocessing.drop_blank_rows(data)
   ## When keeping authors, use data['ProcessedText'] = data['Author'] + ': ' + data['Content'] ##
    data['ProcessedText'] = data['Content']
    text_data = ' '.join(data['ProcessedText'].tolist()).replace('\n', ' ')
    preprocessing.save_data(data)
    #Add way to save both sets of data to save_data
    save_to_txt(text_data, 'GPT\inputs\corpus_combined.txt')


### REMOVED TOKENIZATION CODE ###
    #data['Tokenized'] = data['Content'].apply(lambda sentences: [preprocessing.tokenize_words(sentence) for sentence in sentences])
    #data['Content'] = data['Content'].apply(preprocessing.tokenize_sentences())
    #data = preprocessing.tokenize_words(data)

