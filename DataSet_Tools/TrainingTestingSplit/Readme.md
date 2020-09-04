# Script to generate Training and Test Split

```
python generate_train_test_split.py \
--dataset "MY_DATASET_PATH"
```

### Parameters

##### Required
`--dataset`  
Path to the DataSet which contain **.dat** files.

##### Optional
```
--train_ratio,
--min_cat_train_size,
--output_path
```


### Improvements
Use pandas to speed up calculation of training and testing split   
Bad implementation
