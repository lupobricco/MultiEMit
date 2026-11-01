import argparse
import sys
import re
import torch
import torch.nn as nn
import torch.cuda
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from transformers import BitsAndBytesConfig
import prompts_short_medium_long
import os
import requests
from io import BytesIO
from tqdm import tqdm
#from peft import PeftModel

def parse_args(args):
    """
    Parses command-line arguments.

    Args:
        args (list): A list of arguments to parse (e.g., sys.argv[1:]).

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Script for Hugging Face model generation experiments.'
    )

    # Required Arguments
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='A unique name for the current experiment.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the dataset file.'
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        choices=['emotions','relations', 'irony'],
        help='Labels to classify.'
    )
    # Optional Arguments
    parser.add_argument(
        '--icl_strategy',
        type=str,
        default='zero_shot',
        choices=['zero_shot', 'three_shots'],
        help='ICL strategy to use. Either zero_shot or three_shots'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature for text generation. Higher values (e.g., 1.0) make output more random, '
             'lower values (e.g., 0.5) make it more deterministic.'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='mistral-community/pixtral-12b',
        help='Model name from huggingface or path to adapter.'
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        default=None,
        help='Adapter path.'
    )
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Run on GPU (if available).'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='generated_outputs',
        help='Directory to save generated text and experiment logs.'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='generated_outputs',
        help='File name to save generated text and experiment logs).'
    )
    return parser.parse_args(args)


# Function to create the 'Type' column
def create_type_column(row):
    if row['Additiva'] == 1:
        return 1
    elif row['Parallela'] == 1:
        return 2
    elif row['Divergente'] == 1:
        return 3
    else:
        return 0
    

def preprocess_data(examples, target_names):
    # take a batch of texts
    text = examples["text"]

    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in target_names}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(target_names)))
    # fill numpy array
    for idx, label in enumerate(target_names):
        labels_matrix[:, idx] = labels_batch[label]

    labels_matrix = labels_matrix.astype(int)
    examples["labels"] = labels_matrix.tolist()

    return examples


def proprocess(row, processor, args, prompt, prompt_args):
    if args.icl_strategy == 'zero_shot':
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "content": prompt+row}, {"type": "image"}]
            }
        ]    
    elif args.icl_strategy == "three_shots":
        messages = [
            # ADDITIVE EXAMPLE
            { 
                "role": "user",
                "content": [{"type": "text", "content": prompt+prompt_args['add_text']}, {"type": "image"}],
            },
            {
                "role": "assistant",
                "content": prompt_args['add_res'], 
            },
            # PARALLEL EXAMPLE
            { 
                "role": "user",
                "content": [{"type": "text", "content": prompt+prompt_args['par_text']}, {"type": "image"}],
            },
            {
                "role": "assistant",
                "content": prompt_args['par_res'], 
            },
            # DIVERGENT EXAMPLE
            {
                "role": "user", 
                "content": [{"type": "text", "content": prompt+prompt_args['div_text']}, {"type": "image"}],
            },
            {
                "role": "assistant",
                "content": prompt_args['div_res'], 
            },
            {
                "role": "user",
                "content": [{"type": "text", "content": prompt+row}, {"type": "image"}]
            }
        ]
    prompt = processor.apply_chat_template(messages)
    return prompt


def load_model_processor(args, device):
    model_id = args.model_id
    
    # Configure 4-bit quantization for efficient memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,        
        quantization_config=quantization_config
    )

    if args.adapter_path != None:
        # Load the PEFT adapter on top of the base model
        print(f">> Loading adapters: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        # Ensure the model is in evaluation mode
        model.eval()
        print(f">> Adapters loaded \n\n")
    
    processor = AutoProcessor.from_pretrained(model_id)
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    return model, processor


def load_image_from_url_or_path(path_or_url):
    if os.path.isfile(path_or_url):
        # local path
        try:
            image = Image.open(path_or_url).convert('RGB')
            return image
        except Exception as e:
            print(f"Unable to load image from path: {e}")
            return None
    else:
        # url path
        try:
            response = requests.get(path_or_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Unable to load image from path: {e}")
            return None
        
    

def main(args):
    # printing args
    print("\n\n=== EXP ARGS ===\n")
    for k, v in args.__dict__.items():print(f"{k}: {v}")
    print("\n==/ EXP ARGS /==\n\n")

    nltk.download('punkt_tab')
    
    # Activate GPU
    if args.cuda:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"DEVICE IS: {device}")

    # LOAD DATASET
    print("\nLOADING DATASET... \n\n")
    df = pd.read_csv(args.dataset_path)
    if args.labels == "emotions":
        to_keep=["text","Additiva","Amore","Anticipazione","Disgusto","Divergente","Fiducia","Gioia","Neutro","Parallela","Paura","Rabbia","Sorpresa","Tristezza","img_path"]
        target_names = ['Amore','Anticipazione','Disgusto','Fiducia','Gioia','Neutro','Paura','Rabbia','Sorpresa','Tristezza']
    elif args.labels == "relations":
        to_keep=["text",'Additiva','Divergente','Parallela',"img_path"]
        target_names = ['Additiva','Divergente','Parallela']
    elif args.labels == "irony":
        to_keep=["text",'Ironia','Sarcasmo','None',"img_path", 'Additiva', 'Divergente', 'Parallela']
        target_names = ['Ironia','Sarcasmo','None']

    df = df[to_keep]
    df['type'] = df.apply(create_type_column, axis=1)

    if args.labels == "emotions" or args.labels == "irony":
        df.drop(['Additiva', 'Parallela', 'Divergente'], axis=1, inplace=True)

    df1, train_df = train_test_split(df, test_size=0.2, random_state=42) 

    train_df.reset_index(drop=True, inplace=True)
    if args.labels == 'emotions':
        labels = ['Love', 'Anticipation', 'Disgust', 'Trust', 'Joy', 'Neutral', 'Fear', 'Anger', 'Surprise', 'Sadness']
    elif args.labels == 'relations':
        labels = ['Additive', 'Divergent', 'Parallel']
    elif args.labels == 'irony':
        labels = ['Irony', 'Sarcasm', 'None']

    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    train_df.reset_index(drop=True, inplace=True)
    train_dict = Dataset.from_pandas(train_df)
    dataset = DatasetDict({"train": train_dict})
    if args.labels == "emotions":
        encoded_dataset = dataset.map(preprocess_data, batched=True, fn_kwargs={"target_names": target_names}, remove_columns=["text","Rabbia","Anticipazione","Disgusto","Paura","Gioia","Amore","Neutro","Tristezza","Sorpresa","Fiducia"])
    elif args.labels == "relations":
        encoded_dataset = dataset.map(preprocess_data, batched=True, fn_kwargs={"target_names": target_names}, remove_columns=["text",'Additiva','Divergente','Parallela'])
    elif args.labels == "irony":
        encoded_dataset = dataset.map(preprocess_data, batched=True, fn_kwargs={"target_names": target_names}, remove_columns=["text",'Ironia','Sarcasmo','None'])

    add_df = pd.DataFrame(columns=to_keep)
    par_df = pd.DataFrame(columns=to_keep)
    div_df = pd.DataFrame(columns=to_keep)

    for index, row in train_df.iterrows():
        rel = row['type']
        if rel == 1:
            add_df.loc[len(add_df)] = row
        elif rel== 2:
            par_df.loc[len(par_df)] = row
        elif rel == 3 or rel == 0:
            div_df.loc[len(div_df)] = row
            
    add_df.reset_index(drop=True, inplace=True)
    par_df.reset_index(drop=True, inplace=True)
    div_df.reset_index(drop=True, inplace=True)

    add_dict = Dataset.from_pandas(add_df)
    par_dict = Dataset.from_pandas(par_df)
    div_dict = Dataset.from_pandas(div_df)
    eval_dataset = DatasetDict({"add": add_dict, "par": par_dict, "div": div_dict})
    if args.labels == "emotions":
        encoded_eval = eval_dataset.map(preprocess_data, batched=True, fn_kwargs={"target_names": target_names}, remove_columns=["text","Rabbia","Anticipazione","Disgusto","Paura","Gioia","Amore","Neutro","Tristezza","Sorpresa","Fiducia"])

    # LOAD MODEL
    print("\nLOADING MODEL... \n\n")
    model, processor = load_model_processor(args, device)

    # LOAD PROMPT
    print("\nLOADING PROMPT... \n\n")
    try:
        if args.icl_strategy == 'three_shots':
            df1.reset_index(drop=True, inplace=True)
            # three-shots examples
            prompt_kwards = {
                "add_text": str(df1.iloc[426]['text']),
                "par_text": str(df1.iloc[177]['text']),
                "div_text": str(df1.iloc[207]['text']),
                "add_res": (
                    "Relationship detected: Additive"
                ),
                "par_res": (
                    "Relationship detected: Parallel"
                ),
                "div_res": (
                    "Relationship detected: Divergent"
                )
            }
        else: 
            prompt_kwards = {}
        current_prompt = prompts_short_medium_long.get_prompt(args.experiment_name)
        print(f"\n--- Generated Prompt for '{args.experiment_name}' ---")
        print(current_prompt.strip())
        print("--------------------------------------------------")
    except ValueError as e:
        print(f"Error loading prompt: {e}")
        sys.exit(1) # Exit if the experiment name is invalid

    # apply chat template
    #Adding templated prompt to df
    train_df["text_proce"] = train_df['text'].apply(proprocess, args=(processor,args,current_prompt,prompt_kwards,))
    
    # loading images
    image_folder = "/source/andrea/pixtral_code/images"
    train_df["img"] = train_df["img_path"].apply(lambda x: load_image_from_url_or_path(os.path.join(image_folder, x)))

    # img examples for three-shots
    if args.icl_strategy == 'three_shots':
        add_path = df1["img_path"][426]
        par_path = df1["img_path"][177]
        div_path = df1["img_path"][207]

        add_example = load_image_from_url_or_path(os.path.join(image_folder, add_path))
        par_example = load_image_from_url_or_path(os.path.join(image_folder, par_path))
        div_example = load_image_from_url_or_path(os.path.join(image_folder, div_path))  

    # GENERATION
    print("\nGENERATION... \n\n")
    pred_y=[]
    pred_add=[]
    pred_div=[]
    pred_par=[]

    true_y = np.array(encoded_dataset["train"]["labels"])


    # Open the file in write mode
    f = open(f"{args.output_dir}{args.output_name}.txt", "w", encoding="utf-8")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), leave=True):
        input_text = row["text_proce"]
        input_img = row["img"]

        if args.icl_strategy == 'zero_shot':
            inputs = processor(images=[input_img], 
                                text=input_text, 
                                return_tensors="pt").to(device, dtype=torch.half)
        elif args.icl_strategy == 'three_shots':
            inputs = processor(images=[add_example, par_example, div_example, input_img], 
                                text=input_text, 
                                return_tensors="pt").to(device, dtype=torch.half)
        
        with torch.no_grad():
            generate_ids = model.generate(**inputs, 
                                            max_new_tokens=500, 
                                            do_sample=True, 
                                            temperature=args.temperature)
        
        cut_output = [ out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids) ]
        output = processor.batch_decode(cut_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        f.write("\n")
        f.write(f" {idx} out of 129\n")
        f.write("------------------------------INPUT------------------------------\n")
        f.write(f"{row['text']}\n")
        f.write("------------------------------OUTPUT------------------------------\n")
        f.write(f"{output}\n")


        text_after_emotions = output.strip()

        if text_after_emotions != None:
            encode = np.zeros(len(labels))
            for label in labels:
                if label.lower() in text_after_emotions.lower():
                    k = label2id[label]
                    encode[k] = 1
            
            if 'Neutral' in label2id and encode[label2id['Neutral']] == 1:
                encode = np.zeros(len(labels))
                encode[label2id['Neutral']] = 1
        else:
            if args.labels == "emotions":
                 encode = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif args.labels == "relations":
                encode = [0, 0, 0]
            elif args.labels == "irony":
                encode = [0, 0, 1]


        pred_y.append(encode)

        pred_lab = [id2label[i] for i, val in enumerate(encode) if val == 1]
        f.write(f'Predicted labels: {",".join(pred_lab)}\n')
        true_lab = [id2label[i] for i, val in enumerate(true_y[idx]) if val == 1]
        f.write(f'True labels: {",".join(true_lab)}\n')
        sys.exit("Stopping here to check data before loading model.") 

        rel = row['type']
        if rel == 1:
            pred_add.append(encode)
        elif rel == 2:
            pred_par.append(encode)
        elif rel == 3 or rel == 0:
            pred_div.append(encode)
    f.close()

    # EVALUATION
    print("\nEVALUATION... \n\n")
    true_y = np.array(encoded_dataset["train"]["labels"])
    report_normal = classification_report(true_y,pred_y,output_dict=True,target_names=target_names)
    print(report_normal)

    if args.labels == "emotions":
        true_y = np.array(encoded_eval["add"]["labels"])
        report_add = classification_report(true_y,pred_add,output_dict=True,target_names=target_names)
        print(report_add)

        true_y = np.array(encoded_eval["div"]["labels"])
        report_div = classification_report(true_y,pred_div,output_dict=True,target_names=target_names)
        print(report_div)

        true_y = np.array(encoded_eval["par"]["labels"])
        report_par = classification_report(true_y,pred_par,output_dict=True,target_names=target_names)
        print(report_par)

    # Convert dictionaries to DataFrames
    df1 = pd.DataFrame(report_normal).T.round(3)
    if args.labels == "emotions" or args.labels == "irony":
        df2 = pd.DataFrame(report_add).T.round(3)
        df3 = pd.DataFrame(report_div).T.round(3)
        df4 = pd.DataFrame(report_par).T.round(3)

    # Create a Pandas Excel writer
    with pd.ExcelWriter(f'{args.output_dir}{args.output_name}_reports.xlsx') as writer:
        df1.to_excel(writer, sheet_name='normal', index=True)
        if args.labels == "emotions" or args.labels == "irony":
            df2.to_excel(writer, sheet_name='add', index=True)
            df3.to_excel(writer, sheet_name='div', index=True)
            df4.to_excel(writer, sheet_name='par', index=True)


if __name__ == '__main__':
    # When the script is run directly, parse arguments from sys.argv[1:]
    # sys.argv[0] is the script name itself, so we slice from index 1.
    print(sys.argv[1:])
    parsed_args = parse_args(sys.argv[1:])
    main(parsed_args)