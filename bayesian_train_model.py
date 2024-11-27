import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Creates a CSV file with external labels based on the highest emotion percentage for each video
def create_external_labels(file_csv, external_labels_file):
    df = pd.read_csv(file_csv)
    emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
    video_data = {}

    for _, row in df.iterrows():
        video_name_full, percentage = row['Video Name'], float(row['Percentage'])
        name, emotion_type = video_name_full.split('-')
        if name not in video_data:
            video_data[name] = {}
        video_data[name][emotion_type] = percentage

    with open(external_labels_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video Name', 'Label'])
        for video, emotions_dict in video_data.items():
            if emotions_dict:
                label = max(emotions_dict, key=emotions_dict.get)
                writer.writerow([video, label])
            else:
                print(f"No emotion found for video {video}")
    print("")
    print(f"File {external_labels_file} created successfully.")

# Evaluates the model using cross-validation and prints metrics
def evaluate_with_cross_validation(X, y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianNB())
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average cross-validation: {np.mean(cv_scores):.2f}")
    print(f"Cross-validation standard deviation: {np.std(cv_scores):.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Calculates the mean of each column in an Excel file and normalizes the results
def calculate_column_means(file_excel):
    df = pd.read_excel(file_excel, header=None)
    results = []
    video_names = df.iloc[0, 1:].tolist()
    emotions = df.iloc[1, 1:].tolist()
    values = df.iloc[2:, 1:].astype(float)
    column_means = values.mean(axis=0)
    for i in range(0, len(column_means), 4):
        sub_means = column_means[i:i + 4]
        total = sub_means.sum()
        normalized_means = sub_means / total if total > 0 else [0] * 4
        for j, normalized_value in enumerate(normalized_means):
            results.append((f"{video_names[i + j]}-{emotions[i + j]}", round(normalized_value, 5)))
    return results

# Loads data for Bayesian network training, processing labels and features
# Loads data for Bayesian network training, processing labels and features
def load_data_for_bayesian_network(file_csv, external_labels_file):
    emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
    video_data = {}
    labels = {}

    if os.path.exists(external_labels_file):
        labels_df = pd.read_csv(external_labels_file)
        labels = dict(zip(labels_df['Video Name'], labels_df['Label']))
    else:
        print(f"File {external_labels_file} not found. Ensure it is created correctly.")
        return None, None, emotions

    df = pd.read_csv(file_csv)
    for _, row in df.iterrows():
        video_name_full, percentage = row['Video Name'], float(row['Percentage'])
        try:
            video_name, emotion_type = video_name_full.split('-')
        except ValueError:
            print(f"Invalid video name format: {video_name_full}. Skipping.")
            continue

        if emotion_type not in emotions:
            print(f"Unknown emotion '{emotion_type}' for video '{video_name}'. Skipping.")
            continue

        if video_name not in video_data:
            video_data[video_name] = {em: [] for em in emotions}
        video_data[video_name][emotion_type].append(percentage)

    X_temp = []
    for video, emotion_lists in video_data.items():
        averages = [
            np.mean(emotion_lists[em]) 
            for em in emotions 
            if len(emotion_lists[em]) > 0 and np.mean(emotion_lists[em]) != 0.0
        ]

        if not averages:
            print(f"No valid values for video: {video}. Skipping.")
            continue

        label = labels.get(video)
        if label is None:
            print(f"Missing label for video: {video}. Skipping.")
            continue

        X_temp.extend(averages)

    X = [X_temp[i:i+4] for i in range(0, len(X_temp), 4)]

    # Feature Matrix X
    print("")
    print("########### Feature Matrix (X) ###########")
    print("Each sample in X contains normalized percentages for emotions: Happy, Sad, Angry, Neutral - For each video in the dataset.")
    print("")
    print("-----------------------------------------------------------------------------")
    print(X)
    print("-----------------------------------------------------------------------------")
    print(f"Total number of samples in X: {len(X)}")

    X_final = X      
    if len(X_final) != 84:
        print(f"Error: number of sets is not 84 but {len(X)}. Check the data.")
        return None, None, emotions

    create_external_labels_2('./dativideo.csv', 'external_labels2.csv')

    if os.path.exists('external_labels2.csv'):
        labels_df = pd.read_csv('external_labels2.csv')
        if 'Video Name' not in labels_df.columns or 'Label' not in labels_df.columns:
            print(f"The file 'external_labels2.csv' must contain 'Video Name' and 'Label' columns.")
            return None, None, emotions
        labels = dict(zip(labels_df['Video Name'], labels_df['Label']))
    else:
        print(f"File 'external_labels2.csv' not found. Ensure it is created correctly.")
        return None, None, emotions

    df = pd.read_csv('./dativideo.csv')
    if 'Video Name' not in df.columns or 'Percentage' not in df.columns:
        print(f"The file ./dativideo.csv must contain 'Video Name' and 'Percentage' columns.")
        return None, None, emotions

    for _, row in df.iterrows():
        video_name_full = row['Video Name']
        try:
            percentage = float(row['Percentage'])
        except ValueError:
            print(f"Invalid Percentage value: {row['Percentage']} for video {video_name_full}. Skipping.")
            continue

        try:
            name, emotion_type = video_name_full.split('-')
        except ValueError:
            print(f"Invalid video name: {video_name_full}. Must be in 'Name-Emotion' format. Skipping.")
            continue

        if emotion_type not in emotions:
            print(f"Unknown emotion '{emotion_type}' for video '{name}'. Skipping.")
            continue

        if name not in video_data:
            video_data[name] = {em: 0.0 for em in emotions}
        
        video_data[name][emotion_type] = percentage

    X, y = [], []
 
    for video, percentages in video_data.items():
        label = labels.get(video)
        if label is None:
            continue

        emotion_values = [percentages.get(em, 0.0) for em in emotions]
        X.append(emotion_values)
        y.append(label)
    y.append('Neutral')
    y = np.array(y)

    # Custom Label Mapping
    label_mapping = {
        'Happy': 0,
        'Sad': 1,
        'Angry': 2,
        'Neutral': 3
    }
    y = np.array([label_mapping[label] for label in y])

    # Label Vector y
    print("")
    print("########### Label Vector (y) ###########")
    print("Each entry in y corresponds to the label assigned by a human to the respective video in the dataset to prevent the model from overgeneralizing.")
    print("Label Mapping: 0 = Happy, 1 = Sad, 2 = Angry, 3 = Neutral")
    print("")
    print("-----------------------------------------------------------------------------")
    print(y)
    print("-----------------------------------------------------------------------------")
    print(f"Total number of labels in y: {len(y)}")

    return X_final, y


# Creates a second external labels file with averaged emotions
def create_external_labels_2(file_csv, external_labels_file):
    df = pd.read_csv(file_csv)
    emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
    video_data = {}

    for _, row in df.iterrows():
        video_name, percentage = row['Video Name'], float(row['Percentage'])
        name, emotion_type = video_name.split('-')
        if name not in video_data:
            video_data[name] = {em: [] for em in emotions}
        video_data[name][emotion_type].append(percentage)

    with open(external_labels_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video Name', 'Label'])
        
        video_index = 1
        emotion_group = []
        
        for video, emotions_dict in video_data.items():
            emotion_means = {}
            for emotion, values in emotions_dict.items():
                emotion_means[emotion] = np.mean(values) if values else 0
            
            emotion_group.append(emotion_means)
            
            if len(emotion_group) == 4:
                combined_emotions = {emotion: sum(d[emotion] for d in emotion_group) for emotion in emotion_group[0]}
                
                if combined_emotions:
                    label = max(combined_emotions, key=combined_emotions.get)
                else:
                    label = "Unknown"
                
                video_name = f"VID_RGB_{str(video_index).zfill(3)}"
                writer.writerow([video_name, label])
                
                video_index += 1
                emotion_group = []

        if emotion_group:
            print(f"Incomplete group at the end: {emotion_group}")
            combined_emotions = {emotion: sum(d[emotion] for d in emotion_group) for emotion in emotion_group[0]}
            label = max(combined_emotions, key=combined_emotions.get)
            video_name = f"VID_RGB_{str(video_index).zfill(3)}"
            writer.writerow([video_name, label])
    print("")
    print(f"File {external_labels_file} created successfully.")

# Saves the calculated means to a CSV file
def save_results_to_csv(file_excel, output_csv):
    results = calculate_column_means(file_excel)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video Name', 'Percentage'])
        for video_name, percentage in results:
            writer.writerow([video_name.strip(), percentage])
    print(f"Data successfully saved to CSV: {output_csv}")

# Trains a model using cross-validation and prints the results
def train_with_cross_validation(X, y, model, n_splits=5):
    print("\n" + "#" * 50)
    print(f"Training {model.__class__.__name__} with Cross-Validation ({n_splits} folds)")
    print("#" * 50 + "\n")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")

# Trains Gaussian Naive Bayes with optional SMOTE and various split ratios
def train_gaussian_nb(X, y, use_smote=False, split_ratios=[0.6, 0.7, 0.8]):
    print("\n" + "#" * 50)
    if use_smote:
        print("Training Gaussian Naive Bayes with SMOTE")
    else:
        print("Training Gaussian Naive Bayes without SMOTE")
    print("#" * 50 + "\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_smote:
        smote = SMOTE(k_neighbors=1, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        print(f"Class distribution after SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    else:
        X_resampled, y_resampled = X_scaled, y

    for split_ratio in split_ratios:
        if use_smote:
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=1-split_ratio, random_state=42, stratify=y_resampled
            )
            print("\n" + "*" * 30)
            print(f"Train/Test Split {int(split_ratio*100)}:{round((1-split_ratio)*100)}")
            print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")
            print("*" * 30 + "\n")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-split_ratio, random_state=42, stratify=None
            )
            print("\n" + "*" * 30)
            print(f"Train/Test Split {int(split_ratio*100)}:{round((1-split_ratio)*100)}")
            print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")
            print("*" * 30 + "\n")

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 50)

# Trains Random Forest with optional SMOTE and various split ratios
def train_random_forest(X, y, use_smote=False, split_ratios=[0.6, 0.7, 0.8]):
    print("\n" + "#" * 50)
    if use_smote:
        print("Training RandomForestClassifier with SMOTE")
    else:
        print("Training RandomForestClassifier without SMOTE")
    print("#" * 50 + "\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_smote:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        print(f"Class distribution after SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    else:
        X_resampled, y_resampled = X_scaled, y

    for split_ratio in split_ratios:
        if use_smote:
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=1-split_ratio, random_state=42, stratify=y_resampled
            )
            print("\n" + "*" * 30)
            print(f"Train/Test Split {int(split_ratio*100)}:{round((1-split_ratio)*100)}")
            print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")
            print("*" * 30 + "\n")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-split_ratio, random_state=42, stratify=None
            )
            print("\n" + "*" * 30)
            print(f"Train/Test Split {int(split_ratio*100)}:{round((1-split_ratio)*100)}")
            print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")
            print("*" * 30 + "\n")

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 50)

# Main function to execute the workflow
def main():
    file_excel = "datasets_xlsx/fullData_with_totals.xlsx"
    file_csv = './dativideo.csv'
    external_labels_file = 'external_labels.csv'

    save_results_to_csv(file_excel, file_csv)
    create_external_labels(file_csv, external_labels_file)

    X, y = load_data_for_bayesian_network(file_csv, external_labels_file)
    if X is None or y is None:
        print("Error loading data. Ensure external labels are available.")
        return

    print(f"\nNumber of samples in X: {len(X)}")
    print(f"Number of samples in y: {len(y)}\n")

    train_gaussian_nb(X, y, use_smote=False)
    train_gaussian_nb(X, y, use_smote=True)
    train_random_forest(X, y, use_smote=False)
    train_random_forest(X, y, use_smote=True)

    gaussian_nb = GaussianNB()
    random_forest = RandomForestClassifier(random_state=42)

    train_with_cross_validation(X, y, gaussian_nb)
    train_with_cross_validation(X, y, random_forest)

if __name__ == '__main__':
    main()
