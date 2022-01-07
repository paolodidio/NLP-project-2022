# -*- coding: utf-8 -*-
"""William.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sfsX7q_p7CWjJNDSr4DNZCnFeFtbisHq

https://gist.github.com/jamescalam/55daf50c8da9eb3a7c18de058bc139a3
"""

# !pip install transformers[torch]

import io
import os
import json
import pandas as pd
import random
import torch
import sys
import tensorflow as tf

from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, AdamW, get_scheduler

from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

TRAIN_RATIO = 0.8


def add_splits(dataset):
    """
    Adds a split tag based on the titles
    """
    ratio = int(len(dataset) * TRAIN_RATIO)
    for article in dataset[:ratio]:
        article['split'] = 'train'
    for article in dataset[ratio:]:
        article['split'] = 'val'

    return dataset


def create_dataframe(dataset):
    title = []
    id = []
    context = []
    question = []
    answer_text = []
    answer_start = []
    answer_end = []
    split = []
    for i in range(len(dataset)):
        t = dataset[i]['title']
        s = dataset[i]['split']
        for p in range(len(dataset[i]['paragraphs'])):
            ctx = dataset[i]['paragraphs'][p]['context']
            for q in range(len(dataset[i]['paragraphs'][p]['qas'])):
                qst = dataset[i]['paragraphs'][p]['qas'][q]['question']
                ID = dataset[i]['paragraphs'][p]['qas'][q]['id']
                for a in range(len(dataset[i]['paragraphs'][p]['qas'][q]['answers'])):
                    ans_start = dataset[i]['paragraphs'][p]['qas'][q]['answers'][a]['answer_start']
                    text = dataset[i]['paragraphs'][p]['qas'][q]['answers'][a]['text']
                    ans_end = get_end_index(text, ans_start, ctx)

                    title.append(t)
                    split.append(s)
                    context.append(ctx)
                    question.append(qst)
                    answer_start.append(ans_start)
                    answer_end.append(ans_end)
                    answer_text.append(text)
                    id.append(ID)

    df = pd.DataFrame(
        columns=['title', 'split', 'context', 'question', 'answer_start', 'answer_end', 'answer_text', 'id'])
    df.id = id
    df.title = title
    df.split = split
    df.context = context
    df.question = question
    df.answer_start = answer_start
    df.answer_end = answer_end
    df.answer_text = answer_text
    no_duplicates_df = df.drop_duplicates(keep='first')

    return no_duplicates_df


def get_end_index(answer_text, answer_start, context):
    """
    returns the end index of the answer in the context
     - There might be some mistakes in this meaning that the
       answer index has been pushed with 1 or 2
     - In that case this is something that has to be handled here

     # ...however, sometimes squad answers are off by a character or two
          if context[start_idx:end_idx] == gold_text:
              # if the answer is not off :)
              answer['answer_end'] = end_idx
          else:
              for n in [1, 2]:
                  if context[start_idx-n:end_idx-n] == gold_text:
                      # this means the answer is off by 'n' tokens
                      answer['answer_start'] = start_idx - n
                      answer['answer_end'] = end_idx - n
    """
    end_index = answer_start + len(answer_text)
    assert context[answer_start:end_index] == answer_text
    return end_index


def print_squad_sample(train_data, line_length=100, separator_length=120):
    sample = train_data.sample(frac=1).head(1)
    context = sample.context.values
    print('=' * separator_length)
    print('CONTEXT: ')
    print('=' * separator_length)
    lines = [''.join(context[0][idx:idx + line_length]) for idx in range(0, len(context[0]), line_length)]
    for l in lines:
        print(l)
    print('=' * separator_length)
    questions = train_data[train_data.context.values == context]
    print('QUESTION:', ' ' * (3 * separator_length // 4), 'ANSWER:')
    for idx, row in questions.iterrows():
        question = row.question
        answer = row.answer_text
        print(question, ' ' * (3 * separator_length // 4 - len(question) + 9),
              (answer if answer else 'No awnser found'))


def create_from_splits(dataframe: pd.DataFrame, split: str):
    """
    returns the lists from the dataframe with the
    purpose of creating training and validation datasets
    """
    data = dataframe[dataframe['split'] == split]
    context = data['context'].values
    questions = data['question'].values
    answer_text = data['answer_text'].values
    answer_start = data['answer_start'].values
    answer_end = data['answer_end'].values
    return context, questions, answer_start, answer_end, answer_text


def add_answer_positions(encoding, start_pos, end_pos, tokenizer):
    start_token_positions = []
    end_token_positions = []

    for index, (start, end) in enumerate(zip(start_pos, end_pos)):
        start_token_positions.append(encoding.char_to_token(index, start))
        end_token_positions.append(encoding.char_to_token(index, end))

        # Encoding of positions can lead to encodings being encoded as None if
        # the passage has been truncated. I found a solution which sets the
        # start postion to the max lenght of the tokenizer and changes the end postion
        # if this is also None.

        """
        Gathered from https://gist.github.com/jamescalam/55daf50c8da9eb3a7c18de058bc139a3
        """
        if start_token_positions[-1] is None:
            start_token_positions[-1] = tokenizer.model_max_length
            # end position cannot be found, char_to_token found space, so shift one token forward
        go_back = 1
        while end_token_positions[-1] is None:
            end_token_positions[-1] = encoding.char_to_token(index, end - go_back)
            go_back += 1

        # Lastly we update our encodings with these positions
        encoding.update({'start_positions': start_token_positions, 'end_positions': end_token_positions})


def train_bert_model(train_loader, tokenizer, device):
    """ ---
     # Model Training

    Here we load our pretrained bert model and train it on the dataset we have just created using Dataloader from PyTorch
    This code block is also gathered from https://gist.github.com/jamescalam/55daf50c8da9eb3a7c18de058bc139a3
    """

    torch.cuda.empty_cache()

    # Initialize Model
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.to(device)
    # Setup GPU/CPU
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Move model over to detected device
    model.to(device)
    # Activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # initialize data loader for training data
    # train_loader = DataLoader(small_train_dataset, batch_size=16, shuffle=True)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Saving the model
    model_path = 'models/distilbert-custom'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model


def evaluate_bert(model, eval_loader, val_questions, tokenizer, device):
    """
    # Evaluation
    """

    # switch model out of training mode
    model.eval()

    answers = []
    answers_true = []
    acc = []
    start_pos_prediction = []
    end_pos_prediction = []
    # initialize loop for progress bar
    loop = tqdm(eval_loader)

    # loop through batches
    for batch in loop:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull preds out

            start_pred = outputs['start_logits']
            # start_pos_prediction.append(start_pred.tolist())
            end_pred = outputs['end_logits']
            # end_pos_prediction.append(end_pred.tolist())

            # calculate accuracy for both and append to accuracy list
            acc.append(
                ((torch.argmax(start_pred, dim=1) == start_true).sum() / len(torch.argmax(start_pred, dim=1))).item())
            acc.append(((torch.argmax(end_pred, dim=1) == end_true).sum() / len(torch.argmax(end_pred, dim=1))).item())

            for inputs, start_true, end_true, s, e in zip(input_ids, start_true, end_true, start_pred, end_pred):
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(inputs[torch.argmax(s):torch.argmax(e) + 1]))
                answer_true = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(inputs[start_true:end_true + 1]))
                answers.append(answer)
                answers_true.append(answer_true)

    # calculate average accuracy in total
    accuracy = sum(acc) / len(acc)
    print(acc)

    for question, answer, true in zip(val_questions[:len(answers)], answers, answers_true):
        if answer != '' or answer == '[PAD]':
            print("QUESTION:")
            print(question)
            print("")

            print("TRUE ANSWER:")
            print(true)

            print("")

            print('\nPREDICTED ANSWER')
            print(answer)
            print("")
            print("")

    print("T/F\tstart\tend\n")
    for i in range(len(start_true.tolist())):
        print(f"true\t{start_true[i].item()}\t{end_true[i].item()}\n"
              f"pred\t{start_pred[i]}\t{end_pred[i]}\n")


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


"""
if __name__ == '__main__':
    # verify GPU availability

    # Device Name
    print("Device Name: ",tf.test.gpu_device_name())
    # Version-check
    print("Version:" ,tf.__version__)
    # CUDA Support
    print("CUDA Support: ",str(tf.test.is_built_with_cuda()))

    device_name = tf.test.gpu_device_name()
    print(tf.config.list_physical_devices('GPU'))
    if device_name != '/device:GPU:0':
      pass#raise SystemError('GPU device not found')
    #print('Found GPU at: {}'.format(device_name))

    # specify GPU device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_gpu = torch.cuda.device_count()
    #torch.cuda.get_device_name(0)

    input_file = './training_set.json'

    try:
      with open(input_file, 'r') as f:
          squad_dict = json.load(f)
          flag = True
    except IOError:
        print("Error with the file")

    data = add_splits(squad_dict['data'])
    df = create_dataframe(data)

    print('Training sample')
    print_squad_sample(df[df['split']=='train'])
    print('Validation sample')
    print_squad_sample(df[df['split']=='val'])

    # The actual answer text is probably not necessary but just here for testing now
    train_context, train_questions, train_start_pos, train_end_pos, train_answer = create_from_splits(df, 'train')
    val_context, val_questions, val_start_pos, val_end_pos, val_answer = create_from_splits(df, 'val')

    assert len(train_start_pos) == len(train_answer)
    assert len(val_start_pos) == len(val_end_pos)

    ## Encodings

    print(df['answer_start'])

    print(type(train_context.tolist()))
    print(type(train_questions.tolist()))
    print(type(train_start_pos.tolist()))


    # Using the distiledbert tokenizer we can easily encode both the question and context.
    # As we want the token start and position we need to use the char_to_token method from the tokenizer
    # for the answer positions and add this to the encodings

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_context.tolist(), train_questions.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_context.tolist(), val_questions.tolist(), truncation=True, padding=True)

    add_answer_positions(train_encodings, train_start_pos.tolist(), train_end_pos.tolist(), tokenizer)
    add_answer_positions(val_encodings, val_start_pos.tolist(), val_end_pos.tolist(), tokenizer)

    # PyTorch Dataclass / Loader

    # Dataset class from Huggingface / Pytorch



    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    small_train_dataset = Subset(train_dataset, range(0,20000))
    small_val_dataset = Subset(val_dataset, range(0,1000))
    train_loader = DataLoader(small_train_dataset, batch_size=16,
                                                shuffle=True, num_workers=2)
    eval_loader = DataLoader(small_val_dataset, batch_size=16,
                                                shuffle=False, num_workers=2)

    
     # Model Training
    
    #Here we load our pretrained bert model and train it on the dataset we have just created using Dataloader from PyTorch
    #This code block is also gathered from https://gist.github.com/jamescalam/55daf50c8da9eb3a7c18de058bc139a3
    

    torch.cuda.empty_cache()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Initialize Model
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.to(device)
    # Setup GPU/CPU
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Move model over to detected device
    model.to(device)
    # Activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # initialize data loader for training data
    #train_loader = DataLoader(small_train_dataset, batch_size=16, shuffle=True)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Saving the model
    model_path = 'models/distilbert-custom'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    
    # Evaluation


    #switch model out of training mode
    model.eval()

    answers = []
    answers_true = []
    acc = []
    start_pos_prediction = []
    end_pos_prediction = []
    # initialize loop for progress bar
    loop = tqdm(eval_loader)

    # loop through batches
    for batch in loop:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull preds out

            start_pred = outputs['start_logits']
            #start_pos_prediction.append(start_pred.tolist())
            end_pred = outputs['end_logits']
            #end_pos_prediction.append(end_pred.tolist())


            # calculate accuracy for both and append to accuracy list
            acc.append(((torch.argmax(start_pred, dim=1) == start_true).sum()/len(torch.argmax(start_pred, dim=1))).item())
            acc.append(((torch.argmax(end_pred, dim=1) == end_true).sum()/len(torch.argmax(end_pred, dim=1))).item())


            for inputs, start_true, end_true, s, e in zip(input_ids, start_true, end_true, start_pred, end_pred):
              answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs[torch.argmax(s):torch.argmax(e)+ 1]))
              answer_true = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs[start_true:end_true+1]))
              answers.append(answer)
              answers_true.append(answer_true)



    # calculate average accuracy in total
    accuracy = sum(acc)/len(acc)
    print(acc)

    accuracy

    for question, answer, true in zip(val_questions[:len(answers)], answers, answers_true):
      if answer != '' or answer == '[PAD]':
        print("QUESTION:")
        print(question)
        print("")

        print("TRUE ANSWER:")
        print(true)

        print("")

        print('\nPREDICTED ANSWER')
        print(answer)
        print("")
        print("")

    print("T/F\tstart\tend\n")
    for i in range(len(start_true.tolist())):
        print(f"true\t{start_true[i].item()}\t{end_true[i].item()}\n"
              f"pred\t{start_pred[i]}\t{end_pred[i]}\n")
        

"""
