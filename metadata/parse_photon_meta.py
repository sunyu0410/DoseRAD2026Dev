import pandas as pd
import json

info = json.load(open("stacked-photon-beam-level-metadata.txt"))

df = pd.json_normalize(
    info, 
    record_path=['beams', 'control_points'], 
    meta=[
        'image_file_idx', 
        'anatomical_region', 
        ['beams', 'SAD'], 
        ['beams', 'iso_center'], 
        ['beams', 'num_mlc_leaf_pairs'],
    ]
)

df.to_csv('stacked-photon-beam-level-metadata.csv', index=False)
groups = df.groupby('output_info.output_file_idx')
print(len(groups))
print(groups.get_group(0))
