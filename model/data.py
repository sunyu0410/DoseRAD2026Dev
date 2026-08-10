from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('bev')

from bev.plan import Plan


# img_rot, dose_rot, mask_rot, bev = plan.get(beam_id=0, cp_idx=0)

class PlanData(Dataset):
    def __init__(self, img_file_path, info_json_path, dose_dir):
        self.plan = Plan(
            img_file_path,
            info_json_path,
            dose_dir
        )

        self.ids = [(i, j) for i in range(self.plan.n_beams) for j in range(self.plan.beam_info[i]['n_cp'])]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        beam_id, cp_idx = self.ids[idx]
        return self.plan.get(beam_id, cp_idx)

if __name__ == "__main__":
    dataset = PlanData(
        img_file_path=r"data/ct.mha",
        info_json_path=r"data/1ABB006.json",
        dose_dir=r"data",
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    for epoch in range(1):
        for batch in loader:
            break
            