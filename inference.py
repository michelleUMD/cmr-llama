import os
import torch
from datasets.arrow_dataset import Dataset
from transformers import (
    HfArgumentParser,
    BitsAndBytesConfig,
)
import pandas as pd
from torch.utils.data import DataLoader
import gc

curr_dir  = "/data/aiiih/projects/fangm/llama/Llama_text_medtator/"


# Import from utils.py
from utils import (
    ScriptArguments,
    template,
    parse_xml,
    prep_xml,
    gen_batches_cmr,
    create_and_prepare_model,
    filter_txt,
    load_model
)
from torch.utils.data import DataLoader
torch.manual_seed(42)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

    
def gen_batches_cmr_test_csv():
    # df = pd.read_csv('/data/aiiih/data/train_test_csv/img_text/text2npy.csv')
    # df_cmu = pd.read_json('/data/aiiih/projects/huangp2/Video_text_retrieval/data/CMR_cine_lge/annotation/CMR.json')
    # df_cmu['AccessionNumber'] = df_cmu['cine_lax'].str.split('/').str[-1].str.split('.npy').str[0].astype(int)
    # df = df[df['AccessionNumber'].isin(df_cmu['AccessionNumber'].unique())]
    # for idx in range(len(df)):
    #     Accession_Number = df.iloc[idx]['AccessionNumber']
    #     input_text = df.iloc[idx]['impressions']
    #     formatted_prompt = (
    #         f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    #         f"{template}\n\n"
    #         f"### Impression:\n{input_text}\n\n### dict Output:\n"
    #         f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # Fixed typo in 'assistant'
    #     )
    #     formatted_prompt = "".join(formatted_prompt)
    #     yield {'text': formatted_prompt, 'impression':input_text, 'filename':Accession_Number, 'actual_report':''}
    df = pd.read_csv('/data/aiiih/data/train_test_csv/img_text/text2npy.csv')
    df_cmu = pd.read_json('/data/aiiih/projects/huangp2/Video_text_retrieval/data/CMR_cine_lge/annotation/CMR.json')
    df_cmu['AccessionNumber'] = df_cmu['cine_lax'].str.split('/').str[-1].str.split('.npy').str[0].astype(int)
    df = df[df['AccessionNumber'].isin(df_cmu['AccessionNumber'].unique())]
    for idx in range(len(df)):
        Accession_Number = df.iloc[idx]['AccessionNumber']
        input_text = df.iloc[idx]['impressions']
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{template}\n\n"
            f"### Impression:\n{input_text}\n\n### dict Output:\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # Fixed typo in 'assistant'
        )
        formatted_prompt = "".join(formatted_prompt)
        yield {'text': formatted_prompt, 'impression':input_text, 'filename':Accession_Number, 'actual_report':''}


def gen_single_test():
    df = pd.read_csv('/data/aiiih/data/train_test_csv/img_text/text2npy.csv')
    df = df[df['AccessionNumber']==106271248]
    df = df.drop_duplicates(subset='AccessionNumber')
    df = df.loc[df.index.repeat(5)]

    for idx in range(len(df)):
        Accession_Number = df.iloc[idx]['AccessionNumber']
        input_text = df.iloc[idx]['impressions']
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{template}\n\n"
            f"### Impression:\n{input_text}\n\n### dict Output:\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # Fixed typo in 'assistant'
        )
        formatted_prompt = "".join(formatted_prompt)
        yield {'text': formatted_prompt, 'impression':input_text, 'filename':Accession_Number, 'actual_report':''}

# steps = 600
steps = 1600
checkpoint_path = f'{curr_dir}results_{script_args.model_type}/checkpoint-{steps}'
# checkpoint_path = f'results_{script_args.model_type}/checkpoint-{steps}'
print(f"Model name: {script_args.model_name}")
print(f"Model type: {script_args.model_type}")
print(f"Using checkpoint: {checkpoint_path}")
print(f"Using 4-bit quantization: {script_args.use_4bit}")

model, tokenizer = load_model(script_args, checkpoint_path)
tokenizer.padding_side = 'left'

# Add memory optimization for generation
torch.cuda.empty_cache()
gc.collect()

# Set environment variable to control memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

dataset = 'test'
if dataset=='test':
    # data_path = '/data/aiiih/projects/fangm/nlp_data/Test/*/*.xml'
    data_path = "/data/aiiih/projects/fangm/llama/Llama_text_medtator/Test_Reconciled/test_reconciled.csv"
    # data_path = f'/data/aiiih/projects/nakashm2/multimodal/Llama_text_medtator/fl_data/*.xml'
    test_gen = Dataset.from_generator(lambda: gen_batches_cmr(data_path, phase='test'))
    output_filename = f'oh_{script_args.model_type}_{steps}_nbeam{script_args.num_beams}.csv'
elif dataset=='checking':
    test_gen = Dataset.from_generator(gen_single_test)
    output_filename = f'checking_{script_args.model_type}_{steps}_nbeam{script_args.num_beams}.csv'   
elif dataset=='cmu':
    test_gen = Dataset.from_generator(gen_batches_cmr_test_csv)
    output_filename = f'cmu_{script_args.model_type}_{steps}_nbeam{script_args.num_beams}.csv'

# Create DataLoader for efficient batching
dataloader = DataLoader(test_gen, batch_size=script_args.per_device_eval_batch_size, shuffle=False)

result_lst = []
for batch in dataloader:
    input_txts = batch['text']
    impressions = batch['impression']
    actual_txts = batch['actual_report']
    filenames = batch['filename']
    
    # Tokenize the entire batch at once
    inputs = tokenizer(input_txts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    # Generate for the entire batch
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=script_args.num_beams,
            use_cache=True,
            early_stopping=True 
        )
    
    # Process each item in the batch
    for i in range(len(input_txts)):
        # Find where this example's input ends
        # input_length = (inputs.input_ids[i] != tokenizer.pad_token_id).sum()
        input_length = len(input_ids[i])
        generated_ids = output_ids[i, input_length:]
        
        generated_txt = tokenizer.decode(generated_ids)
        generated_filter_txt = filter_txt(generated_txt)
        
        print(generated_filter_txt, flush=True)
        dic = {
            'filename': filenames[i],
            'impression': impressions[i],
            'actual_txt': actual_txts[i],
            'generated_txt': generated_txt,
            'generated_filter_txt': generated_filter_txt
        }
        result_lst.append(dic)
    
    # Clear memory after each batch
    del input_ids, attention_mask, output_ids
    torch.cuda.empty_cache()


df = pd.DataFrame(result_lst)
output_filename = f'{curr_dir}table/{output_filename}'
df.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")


