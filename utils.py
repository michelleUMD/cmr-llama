import os
import glob
import torch
import xml.etree.ElementTree as ET
import xmltodict
import json
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoProcessor,
)
from transformers import (
    BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM

from peft import LoraConfig


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=3e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.01)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.0)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=4096)
    num_beams: Optional[int] = field(default=1)
    report_to:Optional[str] = field(default='tensorboard')
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3.1-8B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    model_type: Optional[str] = field(
        default="llama3.1",
        metadata={
            "help": "The type of model architecture (llama3.1, llama3.3, etc.)"
        }
    )
    dataset_name: Optional[str] = field(
        default="tatsu-lab/alpaca",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=2000, metadata={"help": "How many optimizer update steps to take"})
    warmup_steps: int = field(default=100, metadata={"help": "# of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=200, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=5, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./results_packing",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

def remove_tags(obj, tags_to_remove):
    """
    Recursively remove specified tags from a dictionary or list.
    
    Args:
        obj: Dictionary or list to process
        tags_to_remove: List of tag names to remove
    """
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in tags_to_remove:
                del obj[key]
            else:
                remove_tags(obj[key], tags_to_remove)
    elif isinstance(obj, list):
        for item in obj:
            remove_tags(item, tags_to_remove)


def prep_xml(filename):
    """
    Prepare XML file by removing specified tags and returning the TAGS section.
    
    Args:
        filename: Path to the XML file
        
    Returns:
        JSON string of the processed dictionary with unwanted tags removed
    """
    # @pattern
    tags_to_remove = ["thrombus", "@spans", "@id", '@oth_type', '@type', '', ]
    with open(filename, 'r') as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
    remove_tags(data_dict, tags_to_remove)
    return json.dumps(data_dict['cmr_interp']['TAGS'], separators=(',', ':'))


def parse_xml(xml_file):
    """
    Parse an XML file and extract text and tags.
    
    Args:
        xml_file: Path to the XML file
        
    Returns:
        Tuple of (raw_data, text, tags)
    """
    with open(xml_file, 'r') as f:
        data = f.read() 
    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    text = root.find('TEXT').text.strip()
    tags = {tag.tag.split('}')[-1]: ([], []) for tag in root.find('TAGS')}

    list_of_tag_elements = []

    for tag_name, tag_list in tags.items():
        for tag in root.findall(f".//{tag_name}"):
            tag_text = tag.attrib.get('text', '')
            tag_list[0].append(tag_text)
            tag_list[1].append(tag)

    return data, text, tags


def gen_batches_cmr(data_path, phase='train'):
    """
    Generate batches from CMR data for training or inference.
    
    Args:
        data_path: Path pattern to XML files
        empty_files: List of files to exclude
        
    Yields:
        Dictionary with text, actual_report, and filename
    """

    if phase == 'train':
        file_lst = [i for i in glob.glob(data_path) if i not in empty_files]
        print(f"Number of files in {data_path}: {len(file_lst)}")

        for filename in file_lst:
            data, text, tags = parse_xml(filename)
            
            # Extract instruction and input from the sample
            input_text = text
            out_text = str(prep_xml(filename))

            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{template}\n\n"
                f"### Impression:\n{input_text}\n\n### dict Output:\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{str(out_text)}"
                f"<|eot_id|><|end_of_text|>"
            )
            formatted_prompt = "".join(formatted_prompt)
            yield {'text': formatted_prompt}
        
    elif phase == 'test':

        df = pd.read_csv(data_path)

        for _, row in df.iterrows():
            input_text = row["input_text"]
            out_text = row["out_text"]
            filename = row["file_id"]

            if pd.isna(out_text):
                out_text = "None"

            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{template}\n\n"
                f"### Impression:\n{input_text}\n\n### dict Output:\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            formatted_prompt = "".join(formatted_prompt)
            yield {'text': formatted_prompt, 'impression':input_text, 'actual_report': out_text, 'filename': filename}


def create_and_prepare_model(args):
    """
    Create and prepare the model with the specified configuration.
    
    Args:
        args: ScriptArguments object with model configuration
        
    Returns:
        For training: Tuple of (model, peft_config, tokenizer)
        For inference: Tuple of (model, tokenizer)
    """
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    # Enable quantization for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = "auto" if args.model_type.lower() in ["qwq32b"] else {"": 0}
    
    # Set torch_dtype based on precision flags
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    

    # For other models, use the existing approach
    # For Llama 3.3 and Llama 4, use the sdpa attention implementation to avoid BFloat16 issues
    attn_implementation = "sdpa" if args.model_type.lower() in ["llama3.3"] and args.bf16 else "eager"
    print(f"Using attention implementation: {attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config if args.use_4bit else None, 
        device_map="auto" if args.model_type.lower() in ["qwq32b"] else {"": 0}, 
        token=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    
    # Set target modules based on model type for LoRA
    target_modules = []
    if args.model_type.lower() in ["llama3.1", "llama3"]:
        target_modules = ['q_proj', 'v_proj']
    elif args.model_type.lower() in ["llama3.3"]:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    elif args.model_type.lower() == "qwq32b":
        # Target modules for QwQ-32B model
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    else:
        # Default target modules for other models
        target_modules = ['q_proj', 'v_proj']
    
    print(f"Using target modules for LoRA: {target_modules}")
    
    # Create peft_config for training
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=target_modules,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

def load_model(script_args, checkpoint_path):
    """
    Load the model with the specified configuration.
    
    Args:
        script_args: ScriptArguments object with model configuration
        checkpoint_path: Path to the checkpoint to load
    """
    # For other models, use the existing approach
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Set torch_dtype based on precision flags
    if script_args.bf16:
        torch_dtype = torch.bfloat16
    elif script_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # For Llama 3.3 , use the sdpa attention implementation to avoid BFloat16 issues
    attn_implementation = "sdpa" if script_args.model_type.lower() in ["llama3.3"] and script_args.bf16 else "eager"
    print(f"Using attention implementation: {attn_implementation}")

    # Load the base model based on model_type
    compute_dtype = getattr(torch, script_args.bnb_4bit_compute_dtype)
    # Enable quantization for efficient inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=script_args.use_4bit,
        bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=script_args.use_nested_quant,
    )

    # Load the model with quantization to save memory
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=bnb_config if script_args.use_4bit else None,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        use_auth_token=False,
    )
    return model, tokenizer


def filter_txt(generated_txt):
    """
    Filter generated text to remove special tokens.
    
    Args:
        generated_txt: Text to filter
        
    Returns:
        Filtered text
    """
    generated_txt = generated_txt.split('<|reserved_special_token_')[0].split('<|eot_id|>')[0].split('<|end_of_text|>')[0]
    return generated_txt 


# Common template for both training and inference

# @location (optional): One of the following la, la_append, ra, rv, lv, root, asc, arch, dsc.

# la, la_append, ra, rv, lv, 

template = """You are a medical language model. Analyze the following impression text and extract the specified medical terms (tags). For each tag found in the text, provide the following information:

@text: The relevant excerpt from the impression.
@mention: One of the following—negated, possible, or positive.
@severity (optional): One of the following ectatic, mild, moderate, aneurysmal, severe.
@location (optional): One of the following root, asc, arch, dsc.
@pattern (optional): One of the following subendocardial, transmural, mid-myocardial, epicardial.

Use the following format:
Copy code
{"tag_name":{"@text": "relevant excerpt from the impression","@mention": "negated/possible/positive","@severity": "ectatic/mild/moderate/aneurysmal/severe","@location": "la/la_append/ra/rv/lv/root/asc/arch/dsc"}}
Instructions:

Only include tags that are mentioned in the impression text. Omit any tags that are not present.
If @severity or @location or @pattern is not applicable, you can omit these fields.
List of tags to extract:
- no_valve_abnorm
- no_aortic_abnorm
- no_ventricular_abnorm
- av_stenosis
- av_regurg
- av_bicuspid
- mv_stenosis
- mv_regurg
- mv_prolapse
- mv_ann_calc
- pv_stenosis
- pv_regurg
- tv_stenosis
- tv_regurg
- perivalvular_abscess
- endocarditis
- aortic_dilation
- aortic_atherosclerosis
- aortic_dissection
- aortic_hematoma
- aortic_artheritis
- thrombus
- ra_dilation
- la_dilation
- la_thrombus
- rv_dysfunc
- rv_dilation
- lv_dysfunc
- lv_dilation
- lv_hypertrophy
- lv_lge_pattern
- lv_edema
- lv_fibrosis
- lv_aneurysm
- lv_noncompact
- papillary_thicken
- pericardial_effus
- myo_pericarditis
- pleura_effus
- genetic_cm
- hcm
- arrhy_cm
- infiltrative_cm
- amyloid
- sarcoid
- ischemic_cm
- nonischemic_cm
- myocardial_infarct
- acute_coronary_synd
- coronary_aneurysm
- hypertensive_heart_dis
- hypertensive_pul_dis
- cardiac_masses
- arvc
- intracardiac_congenital_dis
- vascular_congenital_dis
- oth_card_dis"""

empty_files = ['/data/aiiih/projects/fangm/nlp_data/Train/Train_5_KD_annotated/copy_of_-999999999951236066.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_6_EY_annotated/copy_of_-999999999956339568.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_6_EY_annotated/copy_of_-999999999957535726.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_7_EY_annotated/copy_of_-999999999961070370.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_7_EY_annotated/copy_of_-999999999962324390.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_8_EY_annotated/copy_of_-999999999962931847.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_8_EY_annotated/copy_of_-999999999964866596.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_8_EY_annotated/copy_of_-999999999965110753.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_8_EY_annotated/copy_of_-999999999965132237.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_8_EY_annotated/copy_of_-999999999965168287.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_8_EY_annotated/copy_of_-999999999965273071.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_8_EY_annotated/copy_of_-999999999965518035.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_8_EY_annotated/copy_of_-999999999966702569.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_9_EY_annotated/copy_of_-999999999967712074.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_9_EY_annotated/copy_of_-999999999968314203.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_9_EY_annotated/copy_of_-999999999969709674.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Train/Train_9_EY_annotated/copy_of_-999999999970958013.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 4-KD-annotated/copy_of_-999999999970730699.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 4-KD-annotated/copy_of_-999999999965104647.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 4-KD-annotated/copy_of_-999999999964507345.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 3-MF-annotated/-999999999964507345.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 3-MF-annotated/-999999999965104647.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 3-MF-annotated/-999999999970730699.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 1-EY-annotated/copy_of_-999999999964507345.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 1-EY-annotated/copy_of_-999999999965104647.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 1-EY-annotated/copy_of_-999999999970730699.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 2-SS-annotated/copy_of_-999999999964507345.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 2-SS-annotated/copy_of_-999999999965104647.txt.xml',
                '/data/aiiih/projects/fangm/nlp_data/Test/TEST 2-SS-annotated/copy_of_-999999999970730699.txt.xml']
