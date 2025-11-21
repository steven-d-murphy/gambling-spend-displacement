# Spending displacement from gambling

This project looks at within person displacement of spend between gambling and other categories of spend as well as savings and investments.

This software was developed using python version 3.10.11

Please use the following to ensure that you are using the correct versions of all libraries:
`pip install -r requirements.txt`

Before executing code, edit `src/locations.py` to use the correct file paths for:
+ `data_folder_uk`
+ `data_folder_us`

To execute all computations run the following files in order:
+ `src/A_get_data.py`
+ `src/B_regression_analysis.py`
+ `src/C_gva_analysis.py`
+ `src/D_get_figures_and_tables.py`

The folder `src/classify` contains software to classify individual's financial transactions via OpenAI's public APIs. Use `prompt_tuning.ipynb` for prompt tuning and `run_gen_ai_classifier.py` to obtain classifications for a dataset of brands extracted from transaction descriptions.