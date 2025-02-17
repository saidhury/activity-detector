# 🤸 Human Activity Recognition (HAR) Web App 🌐

This project is a real-time Human Activity Recognition (HAR) web app! 🚀 It uses a pre-trained machine learning model (ONNX) to classify your moves (walking, sitting, etc.) using your phone's sensors. 📱 No servers needed - it all happens in your browser! 😎

## ✨ Features

*   **Real-time Recognition:** See your activity classified instantly. 🏃‍♀️🚶‍♂️🧎
*   **Web-Based:** Works in your browser. 🌐
*   **ONNX Runtime Web:** Fast inference in the browser. 💨
*   **Customizable Features:** Easily change features used by the model via a JSON file. ⚙️
*   **Visual Feedback:** See raw sensor data and predictions. 👀
*   **Mobile-Friendly:** Works on phones and tablets. 📱💻
*   **Easy Deployment:** Just HTML, CSS, and JS! Host it anywhere (GitHub Pages, Netlify...). 🌍

## 🤔 How it Works (in a nutshell)

1.  **Model Loading:** Loads the ONNX model and feature definitions (`selected_features.json`).
2.  **Data Collection:** Starts listening to your device's accelerometer and gyroscope when you hit "Start".
3.  **Feature Extraction:**  Calculates features (like mean, standard deviation, etc.) from the sensor data.
4.  **Inference:**  The ONNX model predicts your activity based on the calculated features.
5.  **Display:** Shows the prediction on the screen!

## 📁 Project Structure

*   **`index.html`:** The webpage.
*   **`app.js`:**  The JavaScript magic. ✨
*   **`style.css`:**  Making things look good. 💅
*   **`logistic_model.onnx`:** Your pre-trained model. (Replace with your own!)
*   **`selected_features.json`:**  Defines the features to use.  Example:
    ```json
    {
        "names": ["tBodyAcc-mean()-X", "tBodyAcc-std()-Y", ...]
    }
    ```
*   **`README.md`:**  You're reading it! 😉
*   **`trainer/`**:  Contains scripts for model training and conversion.
    *   **`main.py`:** Script for training the model (using the UCI HAR Dataset).
    *   **`convert.py`:** Script for converting the trained model to ONNX format.

## 🚀 Getting Started

1.  **Clone:** `git clone <repository_url>`
2.  **Get the UCI HAR Dataset:** Download the dataset from [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and extract it.
3. **Prepare the `trainer` folder**:
      * Copy all files and subfolders from the *extracted* UCI HAR Dataset folder into the `trainer/` folder.  Your `trainer` folder should now contain `main.py`, `convert.py`, and the dataset files (e.g., `train/`, `test/`, `features.txt`, etc.).
4.  **Train and Convert (Optional):**  If you want to train your own model, run `main.py` (to train) and then `convert.py` (to create the ONNX model) within the `trainer` folder. This step requires you to have Python and the necessary libraries installed (see `trainer/main.py` and `trainer/convert.py` for dependencies, such as `scikit-learn`, `tensorflow`, `tf2onnx`).
5.  **Get a Model:**  If you *don't* train your own, you'll need to obtain a pre-trained ONNX model and place it in the main project directory as `logistic_model.onnx`.
6.  **`selected_features.json`:**  Create this file in the main project directory to tell the app which features your model needs.
7.  **Open `index.html`:**  Open it in your browser.
8.  **Permissions:**  Allow the browser to access motion data.
9.  **Start Recording!**

## 📦 Deployment

Host the files (excluding the `trainer` folder) on any static web server (GitHub Pages, Netlify, Vercel, etc.).

## ⚠️ Important Notes

*   **Model Compatibility:** Make sure your `app.js` uses the correct input/output names for your ONNX model (`float_input` and `probabilities` by default).  Use [Netron](https://netron.app/) to check your model.
*   **Features:** You might need to change the feature extraction in `app.js` if your model needs different features.
*   **Privacy**: Be mindful of the user's data.
