# A CRITICAL ANALYSIS OF CLASSIFIER SELECTION IN LEARNED BLOOM FILTERS
Dario Malchiodi, Davide Raimondi, Giacomo Fumagalli, Raffaele Giancarlo and Marco Frasca

The repository includes python files which provide implementations of the Bloom filter (Bloom_filter.py), learned Bloom filter (learned_bloom_filter.py), sandwiched learned Bloom filter (sandwiched_learned_bloom_filter.py) and Ada Bloom filter (Ada-BF.py). Script to train both classifiers and filters, or classifiers alone are also provided. In particular, a list that summarizes the purpose of each of these script is provided below, more details about specific command line arguments are given later.
- `classifier.py` trains classifiers on a given dataset and saves both trained models and their scores.
- `Bloom_filter.py`  initializes a Bloom filter on a given set of keys, tests the initialized structure on a set of negative queries and saves the obtained result.
- `main.py` trains both classifiers (as in `classifier.py`) and learned filters on a given dataset and saves the obtained results.

## Datasets

Datasets used in our experiments are located in the folder `Datasets/...`. Other datasets can be used by specifying their paths through the respective command line argument.
Datasets are loaded as Pandas Dataframes; if needed, one can specify the data-type of each column by creating a `.json` file named `"Dataset_name"_dtypes.json` with the following form:
`{col1_name: dtype, ... , colN_name: dtype}`. This .json file must be in the same folder of the dataset.

For example, suppose to have a dataset named `data.csv` with three columns: `value, feature, label`, the `data_dtypes.json` could be something like
`{value: int32, feature: int8, label: int8}`.

## Classifiers hyperparameters

Values for some classifiers hyperparameters are chosen with a model selection performed with a nested cross validation. The grid of hyperparameters for each classifier can be modified through the `params_grid_search.json` file in the `models` folder. Furthermore, the value of hyperparameters not included in the model selection can be modified through the `classifier_conf.json` file in the same folder. The general form for both these .json file is provided below.

- `\models\classifier_conf.json`:

 `{name_classifier_1 : {name_param_1 : value, name_param2 : value...}, name_classifier_2 : {...} ... }`.
 
- `\models\params_grid_search.json`:

`{name_classifier_1 : {name_param1 :Grid, name_param2 : Grid_2 ...}...}`.

In the `params_grid_search.json` file, the following notation is used:
 - `[start, end, "range"]` -> try all the integer from start to end (e.g [1,4,"range"] -> 1,2,3,4),
 - `[start, end, n_el]` -> try n_el elements in the log space form 10^(start) to 10^(end) (e.g [-3,1,5] -> 10e-3, 10e-2, 10e-1, 10e0. 10e1),
 - `[el1, el2,..., eln, "list"]` -> try all the elements in the list (e.g [1,2,3,4,"list"] -> 1,2,3,4).
 
An example of `classifier_conf.json` and `params_grid_search.json` is given in `\models\`.

## Common command line arguments

Some of the command line arguments are common to every script, these are listed below, together with their behavior. 
 
- `--data_path`: specifies the path of the .csv file to be used as dataset.
- `--pos_ratio`, `--neg_ratio`: given the initial dataset, they specify the ratio of positive and negative samples to be used in the filter training phase. Classic Bloom filter is "trained" simply by passing all keys to hash functions of the filter. LBF and SLBF are trained by trying different values for the classifier threshold (chosen according to `thresholds_q`), and then choosing the value that results in the filter having the lowest fpr value. Default is 1 for both arguments.
-  `--negTest_ratio`: specifies the ratio of negative samples used for testing the trained filter. The samples are extracted from the unsued part of the initial dataset (if `--test_path` is None), or from the given testing dataset (if `--test_path` is not None). Default is 1. In `classifier.py`, this argument is only used for memory usage efficiency issues.
- `--test_path` : optional argument that specifies the path of a testing dataset for the trained filter.

## Bloom_filter.py command line arguments 

A list of command line arguments of `Bloom_filter.py` is given below, together with their behavior.

-- `size_of_BF`: specifies the size of the Bloom filter, in bytes.

## classifier.py command line arguments

A list of command line arguments of `classifier.py` is given below, together with their behavior.

- `--nfoldsCV` : specifies the number of fold used in CV. Default is 5.
- `--classifier_list`: list of classifiers that will be used in the learned filter. In particular, in this case there are only three valid values, corresponding to three classifiers:
    - `RF`: random forest, 
    - `SVM`: linear support vector machine,
    - `FFNN`: multi-layer perceptron.
- `--pos_ratio_clc`, `--neg_ratio_clc`: given the dataset obtained according to `pos_ratio` and `neg_ratio`, specify the ratio of positive and negative samples to be used in the training phase of the classifiers. In particular, the resulting dataset will be used to perform a nested cross validation, with which the best configuration of hyperparameters is chosen. Default is 1 for both arguments.
- `--trees` : specifies the number of trees of the RF.
- `--layers` : specifies the number of neurons in the hidden layers of the FFNN (e.g: `--layers 50 50 50`  specifies 3 hidden layers with 50 neurons each).


## main.py command line arguments

A list of command line arguments of `main.py` is given below, together with their behavior.

- `--nfoldsCV` : specifies the number of fold used in CV. Default is 5.
- `--classifier_list`: list of classifiers that will be used in the learned filter. Here there are only three valid values, corresponding to three different classifiers:
    - `RF`: random forest, 
    - `SVM`: linear support vector machine,
    - `FFNN`: multi-layer perceptron.
- `--pos_ratio_clc`, `--neg_ratio_clc`: given the dataset obtained according to `pos_ratio` and `neg_ratio`, specify the ratio of positive and negative samples to be used in the training phase of the classifiers. In particular, the resulting dataset will be used to perform a nested cross validation, with which the best configuration of hyperparameters is chosen. Default is 1 for both arguments.
- `--trees` : specifies the number of trees of the RF.
- `--layers` : specifies the number of neurons in the hidden layers of the FFNN (e.g: `--layers 50 50 50`  specifies 3 hidden layers with 50 neurons each).
- `--size_of_filter`: specifies the size of the entire structure in bytes (classifier + filter).
- `--force_train`: if True, force training of all classifiers specified in classifier list. If the argument isn't provided, only classifier without a saved model and score are trained.
- `--type_filter` : specifies the type of filter. Here the possible options are:
    - `learned_bloom_filter`, 
    - `sandwiched_learned_bloom_filter`,
    - `Ada-BF`.
- (LBF and SLBF only) `--thresholds_q`: for these types of filters, the thresholds to be tested correspond to the q-order quantiles of the dataset used for training the filter, the one that generates the structure with the lowest number of false positives is chosen. This argument specifies the order q of the quantiles. For example, if thresholds_q is set to 10, all quantiles of order 10 will be tested as thresholds.
- (Ada-BF and disjoint Ada-BF only) `--num_group_min` and `--num_group_max` give the range of number of groups to divide (range of *g*
)
- (Ada-BF and disjoint Ada-BF only)`--c_min` and `--c_max` provide the range of *c* where *c=m_j/m_{j+1}*
- `--save_path` : specifies a path where the results of filter are saved. If there is already a file then the results are append at it.

## Results

Results are saved in `\results\params_path`, where params_path is obtained by concatenation of the name of used Dataset, value of seed and params relative to dataset Ratio. 

In particular, for the classifiers three files are generated, namely: 
- `results\params_path\name_classifer_result.csv`, which contains metrics for all classifier trained.
- `\results\params_path\total_result.csv` which contains the results of all classifier summarized in a single file. 
- `\results\params_path\info_dataset.csv` which contains information about the dimension of the used dataset.

For the filters, if --save_path is non used, the results are saved in `results\params_path\name_filter`. Furthermore, at the end of the execution of `main.py`, is printed the path of the results of the filter tested in that run.

## Commands
- Run Bloom filter: 

`python Bloom_filter.py --data_path ./Datasets/URL_data_features_all.csv --size_of_BF 200000 --pos_ratio 0.7 --neg_ratio 0.7 --negTest_ratio 1.0`

- Run learned Bloom filter on all classifier: 

`python main.py --data_path ./Datasets/URL_data_features_all.csv --size_of_filter 500000 --thresholds_q 10 --type_filter learned_Bloom_filter --classifier_list RF SVM FFNN --pos_ratio 0.7 --neg_ratio 0.7 --pos_ratio_clc 0.7 --neg_ratio_clc 0.7 --negTest_ratio 1.0`  
- Run sandwiched learned Bloom filter on all classifier: 

`python main.py --data_path ./Datasets/URL_data_features_all.csv --size_of_filter 500000 --thresholds_q 50 --type_filter sandwiched_learned_Bloom_filter --classifier_list RF SVM FFNN  --pos_ratio 0.7 --neg_ratio 0.7 --pos_ratio_clc 0.7 --neg_ratio_clc 0.7 --negTest_ratio 1.0` 
- Run Ada-BF on all classifier: 

`python main.py --data_path ./Datasets/URL_data_features_all.csv --size_of_filter 500000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5 --type_filter Ada-BF --classifier_list RF SVM FFNN  --pos_ratio 0.7 --neg_ratio 0.7 --pos_ratio_clc 0.7 --neg_ratio_clc 0.7 --negTest_ratio 1.0`

- Run disjoint Ada-BF (not tested): 

`python disjoint_Ada-BF.py --data_path ./Datasets/URL_data.csv --size_of_Ada_BF 200000  --num_group_min 8  --num_group_max 12  --c_min 1.6  --c_max 2.5`

- Run only analysis on classifiers:

 `python classifier.py --data_path ./Datasets/URL_data_features_all.csv --classifier_list RF SVM FFNN --nfoldCV 5  --pos_ratio 0.7 --neg_ratio 0.7 --pos_ratio_clc 0.7 --neg_ratio_clc 0.7`


