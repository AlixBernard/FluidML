#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-06-06 21:36:08
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-09 02:47:33

""" Description. """

# Built-in packages
from pathlib import Path

# Third party packages

# Local packages

__all__ = []


def main() -> None:
    path = Path(__file__).parent
    tbrf_num = 0
    cases = [
        (
            "template_case",
            [
                "PeriodicHills/Re10595_kOmega_140",
            ],
        ),
        (
            "PH-Re10595_CD-Re12600",
            [
                "PeriodicHills/Re10595_kOmega_140",
                "ConvDivChannel/Re12600_kOmega_100",
            ],
        ),
        (
            "PH-Re10595_SD-Re2900",
            [
                "PeriodicHills/Re10595_kOmega_140",
                "SquareDuct/Re2900_kOmega_50",
            ],
        ),
        (
            "PH-Re10595_CD-Re12600_SD-Re2900",
            [
                "PeriodicHills/Re10595_kOmega_140",
                "ConvDivChannel/Re12600_kOmega_100",
                "SquareDuct/Re2900_kOmega_50",
            ],
        ),
        (
            "SD-Re2900_SD-Re3500",
            [
                "SquareDuct/Re2900_kOmega_50",
                "SquareDuct/Re3500_kOmega_50",
            ],
        ),
    ]
    for case in cases:
        case_name, train_paths = case
        str_train_paths = r""",
    data_path / """.join([f'"{p}"' for p in train_paths])
        p = path / f"{case_name}/config.py"
        with open(p, "w") as file:
            print(f"Making: '{p}'")
            file.write(template_header)
            file.write(
                template_config.format(
                    str_train_paths=str_train_paths,
                    case_name=case_name,
                    tbrf_num=tbrf_num,
                )
            )
            file.write(template_log)


template_header = '''#!/usr/bin/env python
# coding: utf-8
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-06-04 00:00:00
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-06 00:00:00

""" Description. """

# Built-in packages
import logging
from pathlib import Path

# Third party packages
import numpy as np

# Local packages


'''
template_config = '''# Configuration
seed = 42
scale_SR = True
scale_Ak = True
scale_TB = True
select_feats = [0, 1, 4, 6, 7, 8, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

# Paths to data
data_path = Path.home() / "FluidML/data/train_data"
trees_path = Path.home() / "FluidML/trees"
pictures_path = Path.home() / "FluidML/examples/pictures"

train_data_paths = [
    data_path / {str_train_paths},
]
test_data_paths = [
    data_path / "ConvDivChannel/Re12600_kOmega_100",
    data_path / "PeriodicHills/Re5600_kOmega_140",
    data_path / "PeriodicHills/Re10595_kOmega_140",
    data_path / "SquareDuct/Re2600_kOmega_50",
    data_path / "SquareDuct/Re2900_kOmega_50",
    # data_path / "Bump/h31",
]

# Fields required and field names
required_fields = set([
    'Cx', 'Cy',  # For filter
    'gradU', 'k', 'epsilon',  # For Symmetry Rotation tensors
    'gradk', 'k', 'epsilon',  # For Tke features? Or tensor?
    'U', 'k', 'epsilon', 'tau', 'nu', 'd', # For Wang invariants
    'gradp', 'gradk', 'gradU', 'gradU2', # For Wang invariants
])
fields_filenames = {{
    'Cx': "Cx", 'Cy': "Cy",
    'U': "U", 'p': "p", 'k': "k", 'omega': "omega", 'epsilon': "epsilon",
    'nu': "nu", 'nut': "nut", 'phi': "phi", 'tau': "R", 'd': "d",
    'gradU': "gradU", 'gradp': "gradp", 'gradk': "gradk",
    'gradU2': "gradU2"
}}

# Tensor Basis keyword arguments
tbdt_kwargs = {{
    'max_depth': 400,
    'min_samples_leaf': 1,
    'max_features': "sqrt",
    'gamma': 1e-6,  # Set to 0 for no regularization
    'optim_threshold': -1,  # Set to -1 to disable optimization altogether
}}
tbrf_kwargs = {{
    'name': "{case_name}_TBRF{tbrf_num}",
    'n_estimators': 20,
    'bootstrap': True,
    'max_samples': 0.4,
    'n_jobs': 10,
    'random_state': seed,
    'tbdt_kwargs': tbdt_kwargs,
}}

# Filter keyword arguments
gaussianFilter_kwargs = {{
    "sigma": 1,
}}
medianFilter_kwargs = {{
    "size": 7,   
}}

'''
template_log = r'''# Logging template
ret = '\n'
tab = '    '
config_log_text = f"""Configuration logs
General:
    {seed = }
    {scale_SR = }
    {scale_Ak = }
    {scale_TB = }
    {select_feats = }

Paths:
    data_path = '{data_path}'
    trees_path = '{trees_path}'
    pictures_path = '{pictures_path}'
    train_data_paths = [
        {f",{ret}{2*tab}".join([f"'{s}'" for s in train_data_paths])},
    ]
    test_data_paths = [
        {f",{ret}{2*tab}".join([f"'{s}'" for s in test_data_paths])},
    ]

Fields required and field names:
    required_fields = [{", ".join(required_fields)}]
    fields_filenames = {{{f", ".join(
        [f"{k}: {v}" for k, v in fields_filenames.items()]
    )}}}

Tensor Basis keyword arguments:
    tbdt_kwargs = {{{ret}{2*tab}{f",{ret}{2*tab}".join(
        [
            f"{k}: {v!r}" if not isinstance(v, dict) else f"{k}: {{...}}"
            for k, v in tbdt_kwargs.items()
        ]
    )}
    }}
    tbrf_kwargs = {{{ret}{2*tab}{f",{ret}{2*tab}".join(
        [
            f"{k}: {v!r}" if not isinstance(v, dict) else f"{k}: {{...}}"
            for k, v in tbrf_kwargs.items()
        ]
    )}
    }}

Filter keyword arguments:
    gaussianFilter_kwargs = {{{ret}{2*tab}{f",{ret}{2*tab}".join(
        [
            f"{k}: {v!r}" if not isinstance(v, dict) else f"{k}: {{...}}"
            for k, v in gaussianFilter_kwargs.items()
        ]
    )}
    }}
    medianFilter_kwargs = {{{ret}{2*tab}{f",{ret}{2*tab}".join(
        [
            f"{k}: {v!r}" if not isinstance(v, dict) else f"{k}: {{...}}"
            for k, v in medianFilter_kwargs.items()
        ]
    )}
    }}
"""
'''


if __name__ == "__main__":
    main()
