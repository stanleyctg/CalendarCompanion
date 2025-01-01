import streamlit as st
import pickle
import pandas as pd
import os

# Constants
MODEL_PATH = 'nb_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
FEEDBACK_CSV = 'feedback.csv'

class TextClassifierApp:
    def __init__(self, model_path, vectorizer_path, feedback_file):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.feedback_file = feedback_file
        self.category_map = {
            0: 'Academics',
            1: 'Fitness',
            2: 'Others',
            3: 'Personal',
            4: 'Social',
            5: 'Work'
        }
        self.model = self.load_model()
        self.vectorizer = self.load_vectorizer()
        self.initialize_feedback_file()

    @st.cache_resource
    def load_model(_self):
        """Load the pre-trained model from a pickle file."""
        with open(_self.model_path, 'rb') as f:
            return pickle.load(f)

    @st.cache_resource
    def load_vectorizer(_self):
        """Load the pre-trained vectorizer from a pickle file."""
        with open(_self.vectorizer_path, 'rb') as f:
            return pickle.load(f)

    def save_model(self):
        """Save the updated model back to the pickle file."""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def initialize_feedback_file(self):
        """Initialize the feedback file if it doesn't exist."""
        if not os.path.exists(self.feedback_file):
            feedback_df = pd.DataFrame(columns=['text', 'correct_label'])
            feedback_df.to_csv(self.feedback_file, index=False)

    def append_feedback(self, text, correct_label):
        """Append user feedback to the CSV file."""
        feedback_entry = pd.DataFrame({'text': [text], 'correct_label': [correct_label]})
        feedback_entry.to_csv(self.feedback_file, mode='a', header=False, index=False)

    def update_model(self, text, correct_label):
        """Update the model with new data using partial_fit."""
        X_new = self.vectorizer.transform([text])
        y_new = [correct_label]
        self.model.partial_fit(X_new, y_new)
        self.save_model()
        self.append_feedback(text, correct_label)

    def predict(self, text):
        """Predict the category of the input text."""
        X_input = self.vectorizer.transform([text])
        prediction = self.model.predict(X_input)[0]
        confidence = None
        if hasattr(self.model, "predict_proba"):
            prediction_proba = self.model.predict_proba(X_input)[0]
            confidence = max(prediction_proba)
        return prediction, confidence

    def display_feedback_logs(self):
        """Display feedback logs in the Streamlit app."""
        if st.checkbox("Show Feedback Logs"):
            feedback_data = pd.read_csv(self.feedback_file)
            st.dataframe(feedback_data)

    def run(self):
        st.title("📝 Text Category Classifier with Feedback")

        # 1) TEXT INPUT
        user_input = st.text_area("Enter your text here:", height=150)

        # 2) PREDICTION BUTTON
        if st.button("Predict"):
            if user_input.strip() == "":
                st.warning("Please enter some text to classify.")
            else:
                prediction, confidence = self.predict(user_input)
                st.session_state["last_prediction"] = prediction
                st.session_state["last_input"] = user_input
                st.session_state["last_confidence"] = confidence

        # 3) SHOW RESULTS FOR PREDICTION
        if "last_prediction" in st.session_state and "last_input" in st.session_state:
            prediction = st.session_state["last_prediction"]
            predicted_category = self.category_map.get(prediction, "Unknown")
            confidence = st.session_state["last_confidence"]

            st.write(f"**Predicted Category:** {predicted_category}")
            if confidence is not None:
                st.write(f"**Confidence:** {confidence:.2f}")

            # 4) FEEDBACK SECTION
            st.markdown("### Is this prediction correct?")
            feedback_choice = st.radio(
                "Select an option:",
                ("✅ Yes, correct", "❌ No, incorrect"),
                key="feedback_radio"
            )

            correct_category = None
            if feedback_choice == "❌ No, incorrect":
                correct_category = st.selectbox(
                    "Select the correct category:",
                    options=list(self.category_map.values()),
                    key="select_correct_category"
                )

            # 5) SUBMIT FEEDBACK BUTTON
            if st.button("Submit Feedback"):
                if feedback_choice == "✅ Yes, correct":
                    st.success("Thank you for confirming!")
                    self.append_feedback(st.session_state["last_input"], prediction)
                else:
                    if correct_category is not None:
                        correct_label_key = [k for k, v in self.category_map.items() if v == correct_category]
                        if correct_label_key:
                            correct_label_key = correct_label_key[0]
                            self.update_model(st.session_state["last_input"], correct_label_key)
                            st.success("Thank you! The model has been updated with your feedback.")
                        else:
                            st.error("Selected category is invalid.")

        # Instructions
        st.markdown("---")
        st.markdown("### Instructions")
        st.write(
            """
            1. **Enter Text**: Input the text you want to classify.
            2. **Predict**: Click the "Predict" button to see the category.
            3. **Feedback**: If the prediction is incorrect, select "❌ No, incorrect" and provide the correct category.
            4. **Submit**: Click "Submit Feedback" to update the model.
            """
        )

        # Feedback Logs
        st.markdown("---")
        st.markdown("### View Feedback Logs")
        self.display_feedback_logs()

# Run the app
if __name__ == "__main__":
    app = TextClassifierApp(MODEL_PATH, VECTORIZER_PATH, FEEDBACK_CSV)
    app.run()
