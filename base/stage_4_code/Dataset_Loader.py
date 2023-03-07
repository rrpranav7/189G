'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from transformers import BertTokenizer, BertModel
import os
import re
from base.base_class.dataset import dataset
from nltk.corpus import stopwords
import torch


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def clean_text(text):
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', text)
        # Remove non-alphabetic characters
        text = re.sub('[^A-Za-z]+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        # Join words back into sentence
        text = ' '.join(words)
        return text

    def convert_to_bert_embedding(documents):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        encoded_inputs = tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = bert_model(**encoded_inputs)
        embeddings = model_output.last_hidden_state[:, 0, :]
        return embeddings

    def load(self):
        print('loading data...')

        file_path = self.dataset_source_folder_path+self.dataset_source_file_name

        document = []
        y = []

        for label in ['pos', 'neg']:
            for file in os.listdir(os.path.join(file_path, label)):
                with open(os.path.join(file), 'r') as f:
                    lines = f.read()
                    document.append(self.clean_text(lines))
                    y += [1 if label == 'pos' else 0]

        X = self.convert_to_bert_embedding(document)

        return {'X':X, 'y':y}