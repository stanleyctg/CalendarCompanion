import streamlit as st
import pickle
import pandas as pd
import os

MODEL_PATH = 'nb_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
FEEDBACK_CSV = 'feedback.csv'

# Category mapping
category_map = {
    0: 'Academics',
    1: 'Fitness',
    2: 'Others',
    3: 'Personal',
    4: 'Social',
    5: 'Work'
}

# Initialize the feedback file if it doesn't exist
if not os.path.exists(FEEDBACK_CSV):
    feedback_df = pd.DataFrame(columns=['text', 'correct_label'])
    feedback_df.to_csv(FEEDBACK_CSV, index=False)

@st.cache_resource
def load_model():
    """Load the pre-trained model from a pickle file."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_vectorizer():
    """Load the pre-trained vectorizer from a pickle file."""
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

def save_model(model):
    """Save the updated model back to the pickle file."""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

def append_feedback(text, correct_label):
    """Append user feedback to the CSV file."""
    feedback_entry = pd.DataFrame({'text': [text], 'correct_label': [correct_label]})
    feedback_entry.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)

def update_model(model, vectorizer, text, correct_label):
    """
    Update the model with new data using partial_fit.
    """
    X_new = vectorizer.transform([text])
    y_new = [correct_label]
    model.partial_fit(X_new, y_new)
    save_model(model)
    append_feedback(text, correct_label)

def main():
    st.title("📝 Text Category Classifier with Feedback")

    # Load the model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()

    # 1) TEXT INPUT
    user_input = st.text_area("Enter your text here:", height=150)

    # 2) PREDICTION BUTTON
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            # Transform and predict
            X_input = vectorizer.transform([user_input])
            prediction = model.predict(X_input)[0]

            # Save info in session_state so we can access it after rerun
            st.session_state["last_prediction"] = prediction
            st.session_state["last_input"] = user_input

            if hasattr(model, "predict_proba"):
                prediction_proba = model.predict_proba(X_input)[0]
                confidence = max(prediction_proba)
                st.session_state["last_confidence"] = confidence
            else:
                st.session_state["last_confidence"] = None

    # 3) SHOW RESULTS FOR PREDICTION
    if "last_prediction" in st.session_state and "last_input" in st.session_state:
        # Retrieve from session_state
        prediction = st.session_state["last_prediction"]
        predicted_category = category_map.get(prediction, "Unknown")
        confidence = st.session_state["last_confidence"]

        # Display prediction & confidence
        st.write(f"**Predicted Category:** {predicted_category}")
        if confidence is not None:
            st.write(f"**Confidence:** {confidence:.2f}")

        # 4) FEEDBACK SECTION (Always visible once we have a prediction)
        st.markdown("### Is this prediction correct?")
        feedback_choice = st.radio(
            "Select an option:",
            ("✅ Yes, correct", "❌ No, incorrect"),
            key="feedback_radio"
        )

        # If user says "No, incorrect", show a selectbox for the correct category
        if feedback_choice == "❌ No, incorrect":
            correct_category = st.selectbox(
                "Select the correct category:",
                options=list(category_map.values()),
                key="select_correct_category"
            )
        else:
            correct_category = None

        # 5) SUBMIT FEEDBACK BUTTON
        if st.button("Submit Feedback"):
            if feedback_choice == "✅ Yes, correct":
                st.success("Thank you for confirming!")
                # Optionally log the feedback as well
                append_feedback(st.session_state["last_input"], prediction)
            else:
                if correct_category is not None:
                    correct_label_key = [k for k, v in category_map.items() if v == correct_category]
                    if correct_label_key:
                        correct_label_key = correct_label_key[0]
                        # Update the model
                        update_model(model, vectorizer, st.session_state["last_input"], correct_label_key)
                        st.success("Thank you! The model has been updated with your feedback.")
                    else:
                        st.error("Selected category is invalid.")

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

    st.markdown("---")
    st.markdown("### View Feedback Logs")
    if st.checkbox("Show Feedback Logs"):
        feedback_data = pd.read_csv(FEEDBACK_CSV)
        st.dataframe(feedback_data)

if __name__ == "__main__":
    main()
