import models.model
import catboost


class CatboostModel(models.model.Model):
    def __init__(self, filename, feature_generator):
        super(CatboostModel, self).__init__()
        self._name = "Catboost model"
        self._model = catboost.CatBoostClassifier()
        self._model.load_model(filename)
        self._generator = feature_generator

    def get_top(self, candidates, playlist):
        features = self._generator(playlist, candidates)[1]
        scores = self._model.predict(features, prediction_type="Probability")[:, 1]
        return [x[0] for x in sorted(zip([x[0] for x in candidates[0][2]], scores), key=lambda x:x[1], reverse=True)]
