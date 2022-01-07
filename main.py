from bertQA import *


if __name__ == '__main__':
    # verify GPU availability



    # Device Name
    print("Device Name: ", tf.test.gpu_device_name())
    # Version-check
    print("Version:", tf.__version__)
    # CUDA Support
    print("CUDA Support: ", str(tf.test.is_built_with_cuda()))

    device_name = tf.test.gpu_device_name()
    print(tf.config.list_physical_devices('GPU'))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if device_name != '/device:GPU:0':
        pass    #raise SystemError('GPU device not found')

    else:
        print('Found GPU at: {}'.format(device_name))

        #specify GPU device
        n_gpu = torch.cuda.device_count()
        torch.cuda.get_device_name(0)

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
    print_squad_sample(df[df['split'] == 'train'])
    print('Validation sample')
    print_squad_sample(df[df['split'] == 'val'])

    # The actual answer text is probably not necessary but just here for testing now
    train_context, train_questions, train_start_pos, train_end_pos, train_answer = create_from_splits(df, 'train')
    val_context, val_questions, val_start_pos, val_end_pos, val_answer = create_from_splits(df, 'val')

    assert len(train_start_pos) == len(train_answer)
    assert len(val_start_pos) == len(val_end_pos)

    """# Encodings"""

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

    """# PyTorch Dataclass / Loader"""

    # Dataset class from Huggingface / Pytorch

    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    small_train_dataset = Subset(train_dataset, range(0, 20000))
    small_val_dataset = Subset(val_dataset, range(0, 1000))
    train_loader = DataLoader(small_train_dataset, batch_size=16,
                              shuffle=True, num_workers=2)
    eval_loader = DataLoader(small_val_dataset, batch_size=16,
                             shuffle=False, num_workers=2)

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = train_bert_model(train_loader, tokenizer, device)

    evaluate_bert(model,eval_loader, val_questions,tokenizer, device)
