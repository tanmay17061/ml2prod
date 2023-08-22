import joblib
import pickle
from pydantic import BaseModel, ConfigDict, FilePath, AfterValidator
from sklearn.svm import SVC
from typing import Annotated, Optional

def validate_sklearn_model_persistent_extension(model_path: str):
    assert model_path[-6:] == 'joblib' or model_path[-3:] == 'pkl'

class ModelServer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_path: Annotated[str,AfterValidator(validate_sklearn_model_persistent_extension)]
    model: Optional[SVC]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = kwargs['model_path']
        with open(model_path,"rb") as f:
            if model_path[-6:] == 'joblib':
                self.model = joblib.load(f)
            elif model_path[-3:] == 'pkl':
                self.model = pickle.load(f)