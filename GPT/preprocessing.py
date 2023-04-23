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

data_path = 'GPT\inputs\input.csv'

class Preprocessing:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        data = pd.read_csv(self.data_path)
        print(f'Data loaded with {len(data.index)}!')
        return data

    def preprocess_data(self, data):
        data[['Author','Content','Attachments']] = data[['Author','Content','Attachments']].astype(str)
        data = data.drop(['AuthorID', 'Date', 'Attachments','Reactions'], axis=1) ## Figure a way to randomize reactions and attachments
        data['Content'] = data['Content'].apply(lambda x: x.replace('\n', ''))
        data = data[~data.Content.str.contains("nan")]
        print(f'Data preprocessed with {len(data.index)}!')
        print(data.head(5))
        return data

    def save_data(self, data):
        data.to_csv('GPT\inputs\input_preprocessed.txt', index=None, header=None, sep='\t')

if __name__ == '__main__':
    preprocessing = Preprocessing(data_path)
    data = preprocessing.load_data()
    data = preprocessing.preprocess_data(data)
    preprocessing.save_data(data)

