

import json
import os
import random
import pandas as pd

input_file = '/home/lybarger/data/sdoh_id_map.json'
output_dir = '/home/lybarger/sdoh_challenge/analyses/step002_get_mimic_ids/'



output_file = os.path.join(output_dir, 'MIMIC_file_alignment.json')

with open(input_file, "r") as f:
    input_ids = json.load(f)

print("input_id_count", len(input_ids))

output_ids = {}

for orig_id, new_id in input_ids.items():


    round, annotator, mimic_type_id = orig_id.split('/')

    mimic_type, mimic_id = mimic_type_id.split("-")


    assert new_id not in output_ids

    if ('mimic' in orig_id) and ('mimic' in new_id):

        if ('test' in new_id) or ('dev' in new_id) or ('train' in new_id):


            print(new_id, mimic_type, mimic_id)

            output_ids[new_id] = mimic_id

print("output_id_count", len(output_ids))


with open(output_file, "w") as f:
    json.dump(output_ids, f, indent=4)

df = pd.DataFrame(output_ids.items(), columns=['SHAC FILENAME', 'MIMIC ROW_ID'])
f = os.path.join(output_dir, 'MIMIC_file_alignment.csv')
df.to_csv(f, index=False)


df_spot = df.sample(n=20, random_state=10, axis=0)
f = os.path.join(output_dir, 'MIMIC_file_alignment_SAMPLE.csv')
df_spot.to_csv(f, index=False)
print(df_spot)
