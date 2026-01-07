import os                              # Provides functions for file and path handling
from train import extract_mfcc_features  # Imports MFCC feature extraction function from train.py
import joblib                          # Used to load saved machine learning models and scalers

def analyze_audio(input_audio_path):   # Defines a function to analyze a given audio file
    model_filename = "model.pkl"       # Filename of the trained SVM model
    scaler_filename = "scaler.pkl"     # Filename of the scaler used during training
    svm_classifier = joblib.load(model_filename)  # Loads the trained SVM classifier from disk
    scaler = joblib.load(scaler_filename)          # Loads the feature scaler from disk

    if not os.path.exists(input_audio_path):       # Checks whether the audio file exists
        print("Error: The specified file does not exist.")  # Prints error if file path is invalid
    elif not input_audio_path.lower().endswith(".wav"):    # Checks if file is a .wav audio file
        print("Error: The specified file is not a .wav file.")  # Prints error if file format is wrong

    mfcc_features = extract_mfcc_features(input_audio_path)  # Extracts MFCC features from the audio file

    if mfcc_features is not None:                    # Checks if MFCC feature extraction was successful
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))  
        # Reshapes MFCC features and scales them using the trained scaler

        prediction = svm_classifier.predict(mfcc_features_scaled)  
        # Uses the SVM model to predict whether the audio is real or fake

        if hasattr(svm_classifier, "predict_proba"):  # Checks if the model supports probability prediction
            proba = svm_classifier.predict_proba(mfcc_features_scaled)[0]  
            # Gets prediction probabilities for each class
            real_conf = proba[0] * 100.0              # Converts real class probability to percentage
            fake_conf = proba[1] * 100.0              # Converts fake class probability to percentage
        else:
            real_conf = fake_conf = None              # Sets confidence values to None if not supported

        if prediction[0] == 0:                        # If model predicts class 0 (real voice)
            return f"The input audio is classified as real{(f' (confidence {real_conf:.1f}%)' if real_conf is not None else '')}."
            # Returns result string for real audio with confidence if available
        else:
            return f"The input audio is classified as fake{(f' (confidence {fake_conf:.1f}%)' if fake_conf is not None else '')}."
            # Returns result string for fake audio with confidence if available
    else:
        return "Error: Unable to process the input audio."  # Handles failure in feature extraction

if __name__ == "__main__":             # Executes the following code only if this file is run directly
    user_input_file = input("Enter the path of the .wav file to analyze: ")  
    # Takes user input for audio file path
    result = analyze_audio(user_input_file)  # Calls the analyze_audio function
    print(result)                    # Prints the classification result to the terminal
