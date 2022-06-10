# Examples

Here are a few example on how to use FluidML.

The name of the folder indicates the cases used for training, e.g. the folder `PH-Re10595_CD-Re12600` uses the cases Periodic Hills with Reynolds number 10595 and Converging Diverging channel with Reynolds number 12600 for training.

To run multiple cases copied from `template_case` use the following command:
```bash
bash make_folders.sh  # Copy the content from 'template_case' to other folders
python make_config.py  # Replace the 'config.py' files in folders
bash run_cases.sh  # Successively run 'train.py' and 'plot_b.py' in folders
```
The file `make_config.py` can be edited to modify the `config.py` files.

The folder `pictures` contains subfolders for each test cases with their plots, e.g. the folder `pictures/SD-Re2600` contains the plots for the case Square Duct with Reynolds number 2600. The plot names follow the pattern `<b_name>_<training_cases>_<algorithm><number>.png`
