import json
import pandas as pd
import csv
import json_repair
import os


tag_lst = ["no_valve_abnorm", "no_aortic_abnorm", "no_ventricular_abnorm",
            "av_stenosis", "av_regurg", "av_bicuspid",
            "mv_stenosis","mv_regurg", "mv_prolapse",
            "mv_ann_calc", "pv_stenosis", "pv_regurg",
            "tv_stenosis","tv_regurg","perivalvular_abscess",
            "endocarditis","aortic_dilation","aortic_atherosclerosis",
            "aortic_dissection","aortic_hematoma","aortic_artheritis",
            "thrombus","ra_dilation","la_dilation","la_thrombus",
            "rv_dysfunc","rv_dilation","lv_dysfunc","lv_dilation",  
            "lv_hypertrophy","lv_lge_pattern","lv_edema",
            "lv_fibrosis","lv_aneurysm","papillary_thicken",
            "pericardial_effus","myo_pericarditis","pleura_effus",
            "lv_noncompact","genetic_cm","hcm",
            "arrhy_cm", "infiltrative_cm", "amyloid", "sarcoid",
            "ischemic_cm", "nonischemic_cm", "myocardial_infarct",
            "acute_coronary_synd", "coronary_aneurysm",
            "hypertensive_heart_dis", "hypertensive_pul_dis",
            "cardiac_masses", "arvc", "intracardiac_congenital_dis",
            "vascular_congenital_dis", "oth_card_dis"]

def get_data(data, label):
    """Extract condition information with mentions and severities from data."""
    condition_info = {}
    for condition_name, value in data.items():
        condition_name = condition_name.replace(' ', '')
        if not condition_name in tag_lst:
            continue
        
        # Skip keys that start with '@' as they are not conditions
        if condition_name.startswith('@'):
            continue

        # Initialize empty defaults
        combined_mention = ''
        combined_severity = ''
        combined_location = ''
        
        if isinstance(value, list):
            # Process list of dicts
            mentions = []
            severities = []
            locations = []
            for item in value:
                if isinstance(item, dict):
                    mention = item.get('@mention', 'positive').lower()
                    severity = item.get('@severity', '*').lower()
                    location = item.get('@location', '*').lower()
                else:
                    # If item is not a dict, use defaults
                    mention = 'positive'
                    severity = '*'
                    location = '*'
    
                if severity == '':
                    severity = '*'
                if location == '':
                    location = '*'
    
                mentions.append(mention)
                severities.append(severity)
                locations.append(location)
            # Combine found mentions and severities, filtering out empty strings
            combined_mention = ';'.join(m for m in mentions if m)
            combined_severity = ';'.join(s for s in severities if s)
            combined_location = ';'.join(l for l in locations if l)
            
        elif isinstance(value, dict):
            # Single dictionary
            combined_mention = value.get('@mention', 'positive').lower()
            combined_severity = value.get('@severity', '*').lower()
            combined_location = value.get('@location', '*').lower()
        else:
            # Unexpected type, use defaults
            combined_mention = 'positive'
            combined_severity = '*'
            combined_location = '*'
        condition_info[f'{label}_{condition_name}'] = (combined_mention, combined_severity, combined_location)
    return condition_info


def create_csv(all_data, all_conditions, output_path='table/combined_conditions.csv'):
    """Create CSV file with mention and severity columns for each condition."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Create header
        header = ['filename']
        
       
        for condition in sorted(all_conditions):
            header.append(f"{condition}_mention")
            header.append(f"{condition}_severity")
            header.append(f"{condition}_location")
        header.append(f"txt")
        writer.writerow(header)

        
        # Write rows
        for filename, condition_info in all_data.items():
            row = [filename]
            unique_entries = set()
            for condition in sorted(all_conditions):
                mention, severity, location = condition_info.get(condition, ("", "", ""))
                row.append(mention)
                row.append(severity)
                row.append(location)
                
                if 'pred' in condition and mention != '':
                    # Create a set to avoid duplicates directly
                    

                    if ';' in mention:
                        mentions = mention.split(';')
                        # Handle cases where severity or location might not have the same number of elements
                        severities = severity.split(';') if ';' in severity else [severity] * len(mentions)
                        locations = location.split(';') if ';' in location else [location] * len(mentions)
                            
                        # Use zip to safely iterate through all lists together
                        for m, s, l in zip(mentions, severities, locations):
                            condition_name = condition.split('pred_')[1]
                            entry = f"{condition_name} {m} {s} {l}\n"
                            unique_entries.add(entry)
                    else:       
                        # txt += f"{condition.split('pred_')[1]} {mention} {severity} {location}\n"
                        entry = f"{condition.split('pred_')[1]} {mention} {severity} {location}\n"
                        unique_entries.add(entry)
                    
            # Add all unique entries to txt
            txt = ''.join(unique_entries)
            row.append(txt)
            writer.writerow(row)
            print(txt)

def convert_to_int(str):
    return int(str.replace("tensor(", "").replace(")", ""))


def main():
    """Main function to process data and create CSV."""
    root_dir = "/data/aiiih/projects/fangm/llama/Llama_text_medtator/"
    
    output_csv = f'{root_dir}/table/combined_conditions_llama3.3_1000_nbeam1.csv'
    # Load and preprocess data
    df = pd.read_csv(f'{root_dir}/table/cmu_llama3.3_1000_nbeam1.csv')

    df['filename'] = df['filename'].str.split('copy_of_').str[-1].str.split('.json').str[0]
    df['names'] = df['filename'].str.split('/').str[-1]
    df = df.drop_duplicates('names')
    
    df_text2npy = pd.read_csv('/data/aiiih/data/train_test_csv/img_text/text2npy.csv')
    df['AccessionNumber'] = df['filename'].apply(convert_to_int).abs() # File IDs no longer negative
    df = df.merge(df_text2npy[['AccessionNumber', 'impressions']], on='AccessionNumber', how='left')
    
    all_data = {}
    all_conditions = set()
    
    # Process each row in the dataframe
    for i in range(len(df)):
        filename = df['filename'].iloc[i].split('/')[-1].replace('.txt.xml', '.json')
        generated_filter_txt = df.iloc[i]['generated_filter_txt']
        
        # Skip if generated_filter_txt is missing
        if isinstance(generated_filter_txt, float):
            continue
            
        # Parse JSON data
        try:
            generated_filter_txt = generated_filter_txt.replace('}]},"', '}]},"{')
            pred_txts = json_repair.loads(generated_filter_txt)
            
            # Handle different data structures
            pred_data = {}
            if isinstance(pred_txts, list) and len(pred_txts) > 0:
                for pred_txt in pred_txts:
                    if (not pred_txt) or (not isinstance(pred_txt, dict)):
                        continue
                    pred_data2 = get_data(pred_txt, 'pred')
                    pred_data.update(pred_data2)
            elif isinstance(pred_txts, dict):
                pred_data = get_data(pred_txts, 'pred')
                
            # Update collections
            if pred_data:
                all_conditions.update(list(pred_data.keys()))
                all_data[filename] = pred_data
        except Exception as e:
            breakpoint()
            print(f"Error processing {filename}: {e}")
    
    # Create the CSV file
    create_csv(all_data, all_conditions, output_csv)
    print(f"CSV created at {output_csv}")


if __name__ == "__main__":
    main()