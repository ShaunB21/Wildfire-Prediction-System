import re
import tensorflow as tf
import pandas as pd
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from matplotlib import colors
import matplotlib.pyplot as plt
import glob
import os

#Extracts the float list from each record for a specified key
# num_records is the number of records to extract data from
def extract_lists(raw_dataset, key, num_records):
    float_lists = []
    for raw_record in raw_dataset.take(num_records):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        float_list = example.features.feature[key].float_list.value
        float_lists.append(float_list)
    return float_lists

def convert_tfrecord_to_dataframe(tfrecord_file, num_records):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Defines the keys of the tfrecords
    keys = ["tmmx", "tmmn", "elevation", "pr", "NDVI", "sph", "th", "vs", "erc", "population", "pdsi", "PrevFireMask", "FireMask"]

    data_dict = {key: [] for key in keys}

    # Reads the data for each key in each record
    for key in keys:
        # Extracts the data for each key
        float_lists = extract_lists(raw_dataset, key, num_records)
        # Concatinates all the float lists for each key into one float list
        concat_float_list = [item for sublist in float_lists for item in sublist]
        # Assigns the concatinated float list to the respective key in the dictionary
        data_dict[key] = concat_float_list

    # Converts the dictionary into a dataframe
    df = pd.DataFrame(data_dict)
    return df

# Reads all the files that match the pattern
def process_tfrecords(pattern, num_records):
    file_list = glob.glob(pattern)
    dataframes = []

    for file in file_list:
        print(f"Processing {file}...")
        df = convert_tfrecord_to_dataframe(file, num_records)
        dataframes.append(df)

    return dataframes


# Loads the testing and training data
pattern="next_day_wildfire_spread_train_*.tfrecord"
train_dataframes = process_tfrecords(pattern, 1)

pattern="next_day_wildfire_spread_test_*.tfrecord"
test_dataframes = process_tfrecords(pattern, 100)

# Visualises the data as 64*64 pixel images
def visualise_predictions(y_true, y_pred):    
    output_dir = "predictions/modelv2"
    CMAP = colors.ListedColormap(['black', 'gray', 'red'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    num_tests = 200

    # Reshapes the arrays to be 64*64
    y_true = np.array(y_true).reshape(num_tests, 64, 64)
    y_pred = np.array(y_pred).reshape(num_tests, 64, 64)
    
    for i in range(num_tests):
        # Creates the plots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(y_true[i], cmap=CMAP, norm=NORM)
        axes[0].set_title("True Fire Mask")
        axes[0].set_axis_off()
        
        axes[1].imshow(y_pred[i], cmap=CMAP, norm=NORM)
        axes[1].set_title("Predicted Fire Mask")
        axes[1].set_axis_off()
        
        # Saves the figure
        image_path = os.path.join(output_dir, f"prediction_{i}.png")
        plt.savefig(image_path, bbox_inches='tight')
        # Closes the figure to free memory
        plt.close(fig)
        
    print(f"Predictions have been saved to /{output_dir}")


# Compares the predictions of each model
# Predicts 1 if at least one model predicts 1
def find_best(y_true, y_preds):
    y_pred_list = []

    i = 0

    for val in y_true:
        result = 0
        for x in range(len(y_preds)):
            result = result + y_preds[x][i]
        if result >= 1:
            y_pred_list.append(1)
        else:
            y_pred_list.append(0)
        i = i + 1
    y_pred = np.array(y_pred_list)
    return y_pred

# Concatinates the data from all the dataframes
train_data = pd.concat(train_dataframes, ignore_index=True)

scaler = MinMaxScaler()

# Normalises the first eleven columns of data
X_train = train_data.iloc[:, :11]
X_train = scaler.fit_transform(X_train)
# Doesn't normalise the twelfth column as the data is already normalised
X_train_remaining = train_data.iloc[:, 11:12].values
# Select first twelve columns as features
X_train = np.hstack((X_train, X_train_remaining))
# Select thirteenth column as the target feature
y_train = train_data.iloc[:, 12]

# Trains the specified models
model1 = LogisticRegression(max_iter=5000)
model2 = DecisionTreeClassifier(max_depth=5)
model3 = SVC(kernel='rbf', gamma='scale', C=1)

models = [model1, model2, model3]

for model in models:
    model.fit(X_train, y_train)

# Concatinates the data from all the dataframes
test_data = pd.concat(test_dataframes, ignore_index=True)

# Normalises the first eleven columns of data
X_test = test_data.iloc[:, :11]
X_test = scaler.transform(X_test)
# Doesn't normalise the twelfth column as the data is already normalised
X_test_remaining = test_data.iloc[:, 11:12].values
# Select first twelve columns as features
X_test = np.hstack((X_test, X_test_remaining))
# Select thirteenth column as the target feature
y_test = test_data.iloc[:, 12]

# Makes predictions using each model
y_preds = []
for model in models:
    y_pred = model.predict(X_test)
    y_preds.append(y_pred)

# Finds best predictions by comparing the predictions of the models
y_pred = find_best(y_test, y_preds)

# Scores the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Visualises the predictions and saves the figures to a folder
visualise_predictions(y_test, y_pred)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=(0, 1))
visualise_predictions(y_test, y_pred)
