import json
import joblib
import warnings
from split_data import load_data, delete_file
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier
from utilities.evaluation import calculate_metrics, metrics_summary
from utilities.custom_pipeline import ColumnSelector, ConvertDtypes, \
    GetDummies, GetDataFrame, BooleanTransformation
from utilities.config import feature_columns_dtypes, label_column_dtype, to_boolean, label_column, \
    numerical_features, categorical_features, cols_to_modeling, final_features_to_modeling

warnings.filterwarnings(action='ignore')


def define_preprocessing_pipeline():
    """

    :return:
    """
    general_transformations = Pipeline([('boolean', BooleanTransformation(columns=to_boolean)),
                                        ('dtypes', ConvertDtypes(numerical=numerical_features,
                                                                 categorical=categorical_features)),
                                        ('for_modeling', ColumnSelector(columns=cols_to_modeling))])
    numerical_transformations = Pipeline([('numerical_selector', ColumnSelector(columns=numerical_features)),
                                          ('scaler', StandardScaler()),
                                          ('numerical_df', GetDataFrame(columns=numerical_features))])
    categorical_transformations = Pipeline([('categorical_selector', ColumnSelector(columns=categorical_features)),
                                            ('ohe', GetDummies(columns=categorical_features))])
    preprocessor = Pipeline([('general', general_transformations),
                             ('features', FeatureUnion([
                                 ('numerical', numerical_transformations),
                                 ('categorical', categorical_transformations)
                             ])),
                             ('final_df', GetDataFrame(columns=final_features_to_modeling))])
    return preprocessor


def train(transformer_pipeline, train_data):
    """

    :param transformer_pipeline:
    :param train_data:
    :return:
    """
    train_label = train_data.pop(label_column)
    model = Pipeline([('preprocessor', transformer_pipeline),
                      ('estimator', GradientBoostingClassifier(random_state=42))])
    model.fit(train_data, train_label)
    return model


def prediction(model, validation_data):
    """

    :param model:
    :param validation_data:
    :return:
    """
    validation_label = validation_data.pop(label_column)
    y_pred = model.predict(validation_data)
    return y_pred, validation_label


def save_metrics(metrics: dict, path: str):
    with open(path, 'w') as file_path:
        json.dump(metrics, file_path)


def serialize_model(model, model_path: str):
    return joblib.dump(model, filename=model_path)


def main():
    train_data = load_data(filename='data/train.csv', sep=';',
                           dtype=feature_columns_dtypes.update(label_column_dtype))
    validation_data = load_data(filename='data/validation.csv', sep=';',
                                dtype=feature_columns_dtypes.update(label_column_dtype))
    preprocessor = define_preprocessing_pipeline()
    model = train(transformer_pipeline=preprocessor, train_data=train_data)
    serialize_model(model=model, model_path='models/gboosting.pkl')
    y_pred, label = prediction(model=model, validation_data=validation_data)
    metrics = calculate_metrics(y_true=label, y_pred=y_pred)
    save_metrics(metrics, path='metrics/metrics.json')
    metrics_summary(metrics=metrics)
    delete_file(filename='data/Churn_Modelling.csv')
    delete_file(filename='data/train.csv')
    delete_file(filename='data/validation.csv')
    delete_file(filename='data/test.csv')
    delete_file(filename='data/test_label.csv')
    return None


if __name__ == '__main__':
    main()
