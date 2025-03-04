import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from matplotlib import colors
import glob

column_labels = ["tmmx", "tmmn", "elevation", "pr", "NDVI", "sph", "th", "vs", "erc", "population", "pdsi", "PrevFireMask", "FireMask"]
#Max Temps, Min Temps, Elevation, Precipitation, Vegetation, Humidity, Wind Direction, Wind Speed, Energy Release Component, Population, Drought, Previous Fire Mask, Fire Mask

train_files = glob.glob("wildfire_data_train_*.csv")
test_files = glob.glob("new_wildfire_data_test_*.csv")

# Visualises the data as 64*64 pixel images
def visualise_predictions(y_true, y_pred):
    # Sets the colours to use for the images
    CMAP = colors.ListedColormap(['black', 'gray', 'red'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    num_tests = 6 #Max 200
    sample_index = 24
    fig, axes = plt.subplots(num_tests, 2, figsize=(10, 5 * num_tests))

    if num_tests == 1:
        axes = np.array([[axes[0], axes[1]]])  # Ensure axes is 2D for single test case
    
    for i in range(num_tests):
        start_index = (i + sample_index) * 4096 #7, 20, 24, 95, 37 (ok), 48 (ok), 49 (ok), 60 (ok)(test)
        end_index = start_index + 4096
        
        new_y_true= np.array(y_true[start_index:end_index]).reshape(64, 64)
        new_y_pred = np.array(y_pred[start_index:end_index]).reshape(64, 64)

        axes[0, 0].set_title(f"True Fire Mask")       
        axes[0, 1].set_title(f"Predicted Fire mask")

        axes[i, 0].imshow(new_y_true, cmap=CMAP, norm=NORM)
        axes[i, 0].set_axis_off()

        axes[i, 1].imshow(new_y_pred, cmap=CMAP, norm=NORM)
        axes[i, 1].set_axis_off()
        #accuracy = accuracy_score(new_y_true, new_y_pred, normalize=False)
        print(f"Test: {i + sample_index} Accuracy: {accuracy}")
    
    plt.show()

# Loads and prepares training data
dataframes = [pd.read_csv(f) for f in train_files]


# Concatinates the data from all the files
train_data = pd.concat(dataframes, ignore_index=True)

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

# Trains logistic regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Loads and prepares training data
dataframes = [pd.read_csv(f) for f in test_files]

# Concatinates the data from all the files
test_data = pd.concat(dataframes, ignore_index=True)

# Normalises the first eleven columns of data
X_test = test_data.iloc[:, :11] 
X_test = scaler.transform(X_test)
# Doesn't normalise the twelfth column as the data is already normalised
X_test_remaining = test_data.iloc[:, 11:12].values
# Select first twelve columns as features
X_test = np.hstack((X_test, X_test_remaining))
# Select thirteenth column as the target feature
y_test = test_data.iloc[:, 12]

# Makes predictions
y_pred = model.predict(X_test)
# Scores the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=(0, 1))
visualise_predictions(y_test, y_pred)
