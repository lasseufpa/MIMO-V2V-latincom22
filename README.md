# MIMO-V2V-latincom22

This repo guide you trhought the process of replicating the results from the **Ray-Tracing MIMO Channel Dataset for Machine Learning Applied to V2V Communication**. First, you'll need to download or generate the dataset to work with, you can either download the baseline dataset from the [Raymobtime site](https://www.lasse.ufpa.br/raymobtime/) or download the raw data and preprocess it. If you wish to use the baseline data, just go to **Step 2**, but if you wish to preprocess your data, and change it, just keep reading.

## Step 1 - Preprocessing

First, download the raw data from [Raymobtime site](https://www.lasse.ufpa.br/raymobtime/), that will be used to preprocess your data. Second, you'll need to generate your input data, for that, you'll need the `CoordVehiclesRxPerScene.csv` file and run the following code:

```
python3 preprocessing/process_coord_matrix_input.py
```

Feel free to change variables in the code to reach different outcomes, such as: `analysis_area` or `analysis_area_resolution`. By the end of the code you'll have an `coord_matrix_input.npz` that it'll be your input dataset. Then, you'll need to generate the output dataset, for that, download the `ray_tracing_data_v002_carrier60GHz.zip` and run the following code:

```
python3 preprocessing/process_beam_output.py -e 2500 -p path/to/ray_tracing_data_v002_carrier60GHz.zip -c cfg/training_pipeline.json
```

With the flag `-e` being the number of episodes, and `-p` being the path to the `ray_tracing_data_v002_carrier60GHz.zip` folder (ex: `/home/user/data/ray_tracing_data_v002_carrier60GHz`). Feel free to change parameters if you want acheive different results, such as `nTx`, `nRx` (but remember to change in both json files). By the end of it, you'll have your output dataset `beam_output.npz`.

## Step 2

With the dataset, you can run the Deep Neural Network, but first, you'll need onw first step to preprocess the data for the DLL, so first, run the following code:

```
python3 src/datastructure.py -c cfg/data_structure.json
```

Remember to change `input_data` and `output_data` to indicate the path of your dataset. Feel free to play around and change parameters from the json, such as: `nTx`, `nRx` that change the number of the antennas in each veicle, `splits` that are the percentages of split for train, validation and test datasets.This will generate the structure of folders for the dataset. Then run the following code:

```
python3 src/datagenerator.py -c cfg/training_pipeline.json
```

With that, the preprocessing is finished and you can run the DNN. To train simply run:

```
python3 training_pipeline.py -c cfg/training_pipeline.json
```

Feel free to change the parameters of the Optmizers from the `training_pipeline.json`. Then you can test your DNN, run:

```
python3 testing_pipeline.py -c cfg/training_pipeline.json
```
