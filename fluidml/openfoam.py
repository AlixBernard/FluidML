"""Functions to interact with the OpenFOAM software."""

# Built-in packages
from pathlib import Path

# Third party packages
import numpy as np

# Local packages


__all__ = [
    "write_array_to_openfoam",
]


def write_array_to_openfoam(
    target_path: Path | str,
    template_path: Path | str,
    arr: np.ndarray,
    fmt: str = ".6g",
    check_length: bool = True,
    class_: str | None = None,
    location_: str | None = None,
    object_: str | None = None,
) -> None:
    """Write the data from an array into an OpenFOAM file based on a
    template file.

    Parameters
    ----------
    target_path : Path | str
    template_path : Path | str
    arr : np.ndarray[shape=(n,) | shape=(n, m)]
    fmt : str, default='.6g'
        Format to parse the values of `arr` into str. Default is '.6g'
        as it seems to be the closest to OpenFOAM.
    check_length : bool, default=True
        Whether to check if `arr` and the 'internalField' from the
        template have the same length or not, in which case it raises a
        ValueError.
    class_, location_, object_ : str | None, default=None
        FoamFile data to replace, if None keep the same data from the
        template.

    Raises
    ------
    ValueError
        - If `arr` and 'internalField' from the template file have
        different length and `check_length` is True.
        - If the dimension of `arr` is different from 1 or 2.

    """
    target_path = Path(target_path)
    template_path = Path(template_path)
    lines = template_path.read_text().splitlines()

    i0 = 0
    num_entries = 0
    for i, line in enumerate(lines):
        # Modify FoamFile data if required
        if class_ is not None and "class" in line:
            lines[i] = f"    class       {class_};"
        elif location_ is not None and "location_" in line:
            lines[i] = f"    location_    {location_};"
        elif object_ is not None and "object" in line:
            lines[i] = f"    object      {object_};"

        # Find the start index `i0` of the internal field
        if line.startswith("internalField"):
            num_entries = int(lines[i + 1].strip("\n").strip())
            i0 = i + 3
            break

    if check_length and len(arr) != num_entries:
        raise ValueError("`arr` and 'internalField' have different length")

    # Compute the end index `i1` of the internal field
    i1 = i0 + num_entries

    # Replace the internal field by `arr`
    if arr.ndim == 1:
        new_lines_i0_i1 = [f"{x:{fmt}}" for x in arr]
    elif arr.ndim == 2:
        new_lines_i0_i1 = [
            f"({' '.join(f'{x:{fmt}}' for x in vec)})" for vec in arr
        ]
    else:
        raise ValueError(f"{arr.ndim=} must be 1 or 2")
    lines[i0:i1] = new_lines_i0_i1
    lines.append("")  # Add trailing line

    target_path.write_text("\n".join(lines))
