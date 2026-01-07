import os                                   # Provides operating system utilities (paths, directories)
import glob                                 # Used to find all files matching a pattern
import librosa                              # Library for audio processing and feature extraction
import numpy as np                          # Numerical computing library
from sklearn.model_selection import train_test_split  # Splits dataset into train and test sets
from sklearn.preprocessing import StandardScaler      # Standardizes features (mean=0, std=1)
from sklearn.svm import SVC                 # Support Vector Classifier (SVM model)
from sklearn.metrics import accuracy_score, confusion_matrix  # Evaluation metrics
import joblib                               # Used to save and load trained models and scalers

def extract_mfcc_features(audio_path, n_mfcc=20, n_fft=2048, hop_length=512, target_sr=16000):
    """Extract comprehensive audio features for voice authenticity detection."""  # Function documentation
    
    try:
        audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        # Loads audio file, resamples to target sampling rate, converts to mono
        
        if audio_data is None or len(audio_data) == 0:
            print(f"Empty or invalid audio: {audio_path}")  # Handles empty audio files
            return None
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")  # Handles loading errors
        return None

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # Extracts MFCC features from audio

    d1 = librosa.feature.delta(mfcc)         # First-order delta (velocity of MFCCs)
    d2 = librosa.feature.delta(mfcc, order=2)  # Second-order delta (acceleration of MFCCs)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Measures brightness of sound
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Measures frequency below which most energy lies
    
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Measures contrast between spectral peaks and valleys
    
    zcr = librosa.feature.zero_crossing_rate(y=audio_data, hop_length=hop_length)
    # Measures how often signal crosses zero (noisiness)
    
    rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)
    # Measures energy (loudness)
    
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Measures pitch class energy distribution

    feats = np.concatenate([mfcc, d1, d2], axis=0)  # Combines MFCCs and their deltas

    feature_list = []                              # List to store aggregated features
    
    feature_list.append(np.mean(feats, axis=1))   # Mean of MFCC-based features
    feature_list.append(np.std(feats, axis=1))    # Standard deviation of MFCC-based features
    
    feature_list.append([np.mean(spectral_centroid), np.std(spectral_centroid)])
    # Mean and std of spectral centroid
    
    feature_list.append([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
    # Mean and std of spectral rolloff
    
    feature_list.append(np.concatenate([np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1)]))
    # Mean and std of spectral contrast
    
    feature_list.append([np.mean(zcr), np.std(zcr)])
    # Mean and std of zero crossing rate
    
    feature_list.append([np.mean(rms), np.std(rms)])
    # Mean and std of RMS energy
    
    feature_list.append(np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)]))
    # Mean and std of chroma features
    
    feature_vec = np.concatenate([f.flatten() if hasattr(f, 'flatten') else np.array(f) for f in feature_list])
    # Flattens and concatenates all features into a single vector
    
    return feature_vec                            # Returns final feature vector

def create_dataset(directory, label):
    X, y = [], []                                # Lists for features and labels
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    # Finds all .wav files in the directory
    
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)  # Extracts features
        if mfcc_features is not None:
            X.append(mfcc_features)              # Adds features to dataset
            y.append(label)                      # Adds corresponding label
        else:
            print(f"Skipping audio file {audio_path}")  # Skips invalid audio

    print("Number of samples in", directory, ":", len(X))  # Prints dataset size
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    # Prints audio filenames
    
    return X, y                                  # Returns features and labels

def train_model(X, y):
    unique_classes = np.unique(y)                # Finds unique labels
    print("Unique classes in y_train:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError("Atleast 2 set is required to train")  # Ensures at least two classes

    print("Size of X:", X.shape)                 # Prints feature matrix shape
    print("Size of y:", y.shape)                 # Prints label vector shape

    class_counts = np.bincount(y)                # Counts samples per class
    if np.min(class_counts) < 2:
        print("Combining both classes into one for training")  # Handles low sample case
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Splits dataset into training and testing sets

        print("Size of X_train:", X_train.shape)
        print("Size of X_test:", X_test.shape)
        print("Size of y_train:", y_train.shape)
        print("Size of y_test:", y_test.shape)

    scaler = StandardScaler()                    # Initializes feature scaler
    X_train_scaled = scaler.fit_transform(X_train)  # Fits and scales training data

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)  # Scales test data

        svm_classifier = SVC(
            kernel='rbf', gamma='scale', C=1.0,
            probability=True, class_weight='balanced', random_state=42
        )
        # Initializes SVM classifier
        
        svm_classifier.fit(X_train_scaled, y_train)  # Trains the SVM model

        y_pred = svm_classifier.predict(X_test_scaled)  # Predicts on test data

        accuracy = accuracy_score(y_test, y_pred)  # Computes accuracy
        confusion_mtx = confusion_matrix(y_test, y_pred)  # Computes confusion matrix

        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(confusion_mtx)
    else:
        print("Insufficient samples for stratified splitting. Combine both classes into one for training.")
        print("Train on all available data.")

        svm_classifier = SVC(
            kernel='rbf', gamma='scale', C=1.0,
            probability=True, class_weight='balanced', random_state=42
        )
        # Initializes SVM classifier
        
        svm_classifier.fit(X_train_scaled, y_train)  # Trains model on all data

    model_filename = "model.pkl"                 # Output model filename
    scaler_filename = "scaler.pkl"               # Output scaler filename
    joblib.dump(svm_classifier, model_filename)  # Saves trained model
    joblib.dump(scaler, scaler_filename)         # Saves scaler

def main():
    real_dir = "dataset/real"                    # Directory containing real voice samples
    fake_dir = "dataset/fake"                    # Directory containing fake voice samples

    X_real, y_real = create_dataset(real_dir, label=0)  # Loads real dataset
    X_fake, y_fake = create_dataset(fake_dir, label=1)  # Loads fake dataset

    if len(X_real) < 2 or len(X_fake) < 2:
        print("Each class should have at least two samples for stratified splitting.")
        print("Combining both classes into one for training.")
        X = np.vstack((X_real, X_fake))           # Combines feature matrices
        y = np.hstack((y_real, y_fake))           # Combines labels
    else:
        X = np.vstack((X_real, X_fake))           # Combines feature matrices
        y = np.hstack((y_real, y_fake))           # Combines labels

    train_model(X, y)                             # Trains the model

if __name__ == "__main__":                        # Entry point of the script
    main()                                        # Runs training process
