import pandas as pd
import json

info = json.load(open("stacked-proton-beam-level-metadata.txt"))

df = pd.json_normalize(
    info,
    record_path=['beams', 'rays', 'beamlets'],
    meta=[
        'iso_center',
        'image_file_idx',
        'anatomical_region',
        ['beams', 'gantry_angle'],
        ['beams', 'rays', 'ray_source'],
        ['beams', 'rays', 'ray_target']
    ],
)

df.to_csv('stacked-proton-beam-level-metadata.csv', index=False)
groups = df.groupby('output_info.output_file_idx')
print(len(groups))
print(groups.get_group(0))
