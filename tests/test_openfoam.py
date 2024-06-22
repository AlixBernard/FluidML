"""Tests of functions from the module `fluidml.features`."""

# Built-in packages

# Third party packages
import numpy as np

# Local packages
from fluidml.openfoam import write_array_to_openfoam


OPENFOAM_FILE_TEMPLATE_STR = r"""
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2306                                  |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";
    class       volVectorField;
    location    "5000";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector> 
7
(
(0.60827 9.82541e-17 -5.56498e-17)
(0.61195 1.24322e-16 -2.9996e-17)
(0.615425 8.70808e-17 1.41947e-17)
(0.618672 7.44653e-17 1.94089e-17)
(0.621657 2.55887e-17 1.02646e-17)
(0.624333 -4.45153e-18 1.15427e-17)
(0.626644 -4.24091e-17 -7.21914e-17)
)
;

boundaryField
{
    inlet
    {
        type            cyclic;
    }
    outlet
    {
        type            cyclic;
    }
    symmetries
    {
        type            symmetry;
    }
    walls
    {
        type            noSlip;
    }
}


// ************************************************************************* //
"""


def test_write_array_to_openfaom(tmp_path):
    arr = np.array(
        [
            [0.60827, 9.82541e-17, -5.56498e-17],
            [0.61195, 1.24322e-16, -2.9996e-17],
            [0.615425, 8.70808e-17, 1.41947e-17],
            [0.618672, 7.44653e-17, 1.94089e-17],
            [0.621657, 2.55887e-17, 1.02646e-17],
            [0.624333, -4.45153e-18, 1.15427e-17],
            [0.626644, -4.24091e-17, -7.21914e-17],
        ]
    )
    template_path = tmp_path / "U_template"
    target_path = tmp_path / "U_target"
    template_path.write_text(OPENFOAM_FILE_TEMPLATE_STR)

    write_array_to_openfoam(target_path, template_path, arr)
    assert target_path.read_text() == OPENFOAM_FILE_TEMPLATE_STR
    assert target_path.read_text() == template_path.read_text()
