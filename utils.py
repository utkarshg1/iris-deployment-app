# Predict the results
import pandas as pd
from sklearn.pipeline import Pipeline


def predict_species(
        model: Pipeline, 
        sep_len: float,
        sep_wid:float,
        pet_len: float,
        pet_wid: float):
    xnew = pd.DataFrame(
        [
            {
                "sepal_length": sep_len,
                "sepal_width": sep_wid,
                "petal_length": pet_len,
                "petal_width": pet_wid 
            }
        ]
    )
    pred = model.predict(xnew)
    probs = model.predict_proba(xnew)
    probs_df = pd.DataFrame(probs, columns=model.classes_)
    return pred[0], probs_df.round(4)
