"""¹H shifts (ppm) of common laboratory impurities by NMR solvent.

After Fulmer et al., Organometallics 29, 2176 (2010) (an extension of Gottlieb,
Kotlyar & Nudelman, J. Org. Chem. 62, 7512, 1997), for the solvents this group uses
most. Exchangeable OH/NH signals vary with concentration and are omitted, except water.

NOTE: entered by hand, not machine-checked against the paper's tables: verify an entry
before relying on it for a publication. Shifts move slightly with concentration and
temperature, so matching uses a tolerance, and labels are suggestions ("impurity: X").
"""

from __future__ import annotations

#: solvent -> impurity -> list of ¹H shifts (ppm)
IMPURITIES_1H: dict[str, dict[str, list[float]]] = {
    "CDCl3": {
        "water": [1.56],
        "acetic acid": [2.10],
        "acetone": [2.17],
        "acetonitrile": [2.10],
        "benzene": [7.36],
        "tert-butanol": [1.28],
        "dichloromethane": [5.30],
        "diethyl ether": [3.48, 1.21],
        "DMF": [8.02, 2.96, 2.88],
        "DMSO": [2.62],
        "ethanol": [3.72, 1.25],
        "ethyl acetate": [4.12, 2.05, 1.26],
        "hexane": [1.26, 0.88],
        "methanol": [3.49],
        "pentane": [1.27, 0.88],
        "2-propanol": [4.04, 1.22],
        "pyridine": [8.62, 7.68, 7.29],
        "silicone grease": [0.07],
        "tetrahydrofuran": [3.76, 1.85],
        "toluene": [7.25, 7.17, 2.36],
        "triethylamine": [2.53, 1.03],
    },
    "DMSO-d6": {
        "water": [3.33],
        "acetic acid": [1.91],
        "acetone": [2.09],
        "acetonitrile": [2.07],
        "benzene": [7.37],
        "chloroform": [8.32],
        "dichloromethane": [5.76],
        "diethyl ether": [3.38, 1.09],
        "DMF": [7.95, 2.89, 2.73],
        "ethanol": [3.44, 1.06],
        "ethyl acetate": [4.03, 1.99, 1.17],
        "hexane": [1.25, 0.86],
        "methanol": [3.16],
        "pyridine": [8.58, 7.79, 7.39],
        "silicone grease": [-0.06],
        "tetrahydrofuran": [3.60, 1.76],
        "toluene": [7.18, 2.30],
        "triethylamine": [2.43, 0.93],
    },
    "CD3OD": {
        "water": [4.87],
        "acetone": [2.15],
        "acetonitrile": [2.03],
        "chloroform": [7.90],
        "dichloromethane": [5.49],
        "diethyl ether": [3.49, 1.15],
        "DMF": [7.97, 2.99, 2.86],
        "DMSO": [2.65],
        "ethanol": [3.60, 1.17],
        "ethyl acetate": [4.09, 2.01, 1.24],
        "hexane": [1.29, 0.90],
        "tetrahydrofuran": [3.71, 1.87],
        "toluene": [2.32],
        "triethylamine": [2.58, 1.05],
    },
}

#: Matching tolerance for impurity shifts (ppm).
IMPURITY_TOLERANCE_PPM = 0.02
