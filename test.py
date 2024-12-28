import pickle
import pandas as pd

category_map = {
    0: 'Academics',
    1: 'Fitness',
    2: 'Others',
    3: 'Personal',
    4: 'Social',
    5: 'Work'
}

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('nb_model.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.DataFrame({
    # To be changed...
    'combined_text': ["Calling with my parents to talk about my day and time at university"]
})

print((vectorizer.toarray()))