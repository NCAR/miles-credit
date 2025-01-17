import os
import shutil
import yaml
# to run: 
# python Stage_Files_Climate.py 00181 /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy model_multi_WxFormer.yml

# python Stage_Files_Climate.py 00220 /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb /glade/derecho/scratch/wchapman/CREDIT_runs/wxformer_1dg_cesm_data_nopost_bigbig_SSTforced_DryWaterEnergy/model_00191_nb model_multi_example-v2025.2.0.yml

def copy_and_replace_files(number: str, source_dir: str, target_dir: str, yaml_file: str):
    # Validate input
    if len(number) != 5 or not number.isdigit():
        raise ValueError("Number must be exactly 5 digits.")
    
    # Construct the new directory path
    target_model_dir = os.path.join(target_dir, f'model_{number}')
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_model_dir, exist_ok=True)
    
    # Hardcoded list of files to copy
    files_to_copy = [
         yaml_file,
        'checkpoint.pt'
    ]
    
    # First, copy the files over
    for file in files_to_copy:
        
        
        source_file = os.path.join(source_dir, file)
        # Special case for checkpoint file
        if 'checkpoint.pt' in file:
            source_file = source_file.replace('checkpoint.pt', f'checkpoint.pt{number}.pt')
        
        target_file = os.path.join(target_model_dir, file)
        
        # Ensure the target subdirectory exists
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        # Copy the file directly
        shutil.copy2(source_file, target_file)
        print(f"Copied: {source_file} -> {target_file}")


    # Alter the YAML file specifically
    target_yaml_file = os.path.join(target_model_dir, yaml_file)
    with open(target_yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Modify the predict.save_forecast field
    save_forecast_path = yaml_data.get('predict', {}).get('save_forecast', '')
    if save_forecast_path:
        base_name = os.path.basename(save_forecast_path.rstrip('/'))
        new_save_forecast_path = os.path.join(os.path.dirname(save_forecast_path), f"{base_name}_{number}")
        yaml_data['predict']['save_forecast'] = new_save_forecast_path
        print(f"Updated 'predict.save_forecast' to: {new_save_forecast_path}")
    
    # Write the modified YAML back to the file with improved formatting
    class IndentedDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentedDumper, self).increase_indent(flow, False)
    
    with open(target_yaml_file, 'w') as f:
        yaml.dump(
            yaml_data,
            f,
            Dumper=IndentedDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=4  # Control indentation for better readability
        )
        print(f"Updated YAML file saved: {target_yaml_file}")

    run_fil = './run_climate_out.sh'
    cp_run_fil = f'./run_climate_out_{number}.sh'
    shutil.copy2(run_fil,cp_run_fil)

    with open(cp_run_fil, 'r') as f:
        content = f.read()
    content = content.replace('XXXXX', number)
    
    with open(cp_run_fil, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copy files and replace placeholder text.")
    parser.add_argument("number", type=str, help="Five-digit number to replace placeholder.")
    parser.add_argument("source_dir", type=str, help="Source directory to copy files from.")
    parser.add_argument("target_dir", type=str, help="Target base directory.")
    parser.add_argument("yaml_file", type=str, help="yaml file to drive model")

    args = parser.parse_args()

    try:
        copy_and_replace_files(args.number, args.source_dir, args.target_dir, args.yaml_file)
        print("All files processed successfully.")
    except Exception as e:
        print(f"Error: {e}")
