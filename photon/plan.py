import SimpleITK as sitk
from pathlib import Path
import json
from collections import OrderedDict as odict

def get_body_mask(img_sitk, thres=-1024):
    return img_sitk > thres

class Plan:
    def __init__(self, img_file_path, info_json_path, dose_dir):
        self.img_file_path = img_file_path
        self.info_json_path = info_json_path
        self.dose_dir = Path(dose_dir)

        self.img = sitk.ReadImage(img_file_path)
        self.img_arr = sitk.GetArrayFromImage(self.img)
        self.body_mask = get_body_mask(self.img, thres=-1024)
        self.info = json.load(open(info_json_path))

        self.and_filter = lambda x, y: sitk.AndImageFilter().Execute(x, y)

        self.parse_json()

        self.set_state(beam_id=0, cp_idx=0)

    @property
    def isocentre(self):
        return self.beam_info[self.beam_id]["isocentre"]

    @property
    def isocentre_ijk(self):
        return self.img.TransformPhysicalPointToIndex(self.isocentre) 

    @property
    def gantry_angle(self):
        return self.cp[self.beam_id][self.cp_idx]["ga"]

    def parse_json(self):
        # In [113]: beam_info[0]
        # Out[113]:
        # {'SAD': 1000,
        # 'isocentre': [-46.8471844842125, 27.777663262437926, -28.13538836315937],
        # 'n_mlc_leaf': 80}
        self.beam_info = odict(
            (
                b["beam_idx"],
                dict(
                    sad=b["SAD"],
                    isocentre=b["iso_center"],
                    n_mlc_leaf=b["num_mlc_leaf_pairs"],
                    n_cp=len(b["control_points"]),
                ),
            )
            for b in self.info["beams"]
        )
        self.cp = odict(
            (
                beam_id,
                odict(
                    (
                        cp["cp_idx"],
                        dict(
                            ga=cp["gantry_angle"],
                            l=cp["mlc_left_int_mm"],
                            r=cp["mlc_right_int_mm"],
                        ),
                    )
                    for cp in self.info["beams"][beam_id]["control_points"]
                ),
            )
            for beam_id in range(len(self.beam_info))
        )
        self.n_beams = len(self.beam_info)

    def set_state(self, beam_id, cp_idx):
        assert cp_idx in self.cp[beam_id]
        self.beam_id = beam_id
        self.cp_idx = cp_idx

        print('State set to {self.beam_id}, {self.cp_idx}')


