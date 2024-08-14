import os
import json

# Define the folder containing the JSON files
json_folder = 'outputs/jsons'
output_file = 'merged_json.txt'

# Initialize variables to store the sum of values and count of JSON files
total_sum = [0, 0, 0, 0]
num_files = 0

# Open the output file for writing
with open(output_file, 'w') as f_out:
    # Loop through each JSON file in the folder
    for json_filename in os.listdir(json_folder):
        if json_filename.endswith('.json'):
            json_path = os.path.join(json_folder, json_filename)
            
            # Read the JSON file
            with open(json_path, 'r') as f_in:
                data = json.load(f_in)
                
                # Extract the four values and round to two decimal places
                values = [
                    round(data.get("dukemtmcreid", 0), 2),
                    round(data.get("market1501", 0), 2),
                    round(data.get("occluded_duke", 0), 2),
                    round(data.get("p_dukemtmc", 0), 2)
                ]
                
                # Write the JSON filename and values to the output file
                f_out.write(f"{json_filename.split('.')[0]}:\n")
                f_out.write(f'\tdukemtmcreid: {values[0]}\n')
                f_out.write(f'\tmarket1501: {values[1]}\n')
                f_out.write(f'\toccluded_duke: {values[2]}\n')
                f_out.write(f'\tp_dukemtmc: {values[3]}\n\n')
                
                # Update the total sum and count of files
                total_sum = [total_sum[i] + values[i] for i in range(4)]
                num_files += 1
    
    # Calculate and write the average values at the end of the file, rounded to two decimal places
    if num_files > 0:
        averages = [round(total_sum[i] / num_files, 2) for i in range(4)]
        f_out.write(f'\tAverage Values:\n')
        f_out.write(f'\tdukemtmcreid: {averages[0]}\n')
        f_out.write(f'\tmarket1501: {averages[1]}\n')
        f_out.write(f'\toccluded_duke: {averages[2]}\n')
        f_out.write(f'\tp_dukemtmc: {averages[3]}\n')

print("Merging complete. Output written to", output_file)
