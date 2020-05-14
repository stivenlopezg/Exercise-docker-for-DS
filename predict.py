import joblib
import numpy as np
import pandas as pd
from utilities.config import bucket, key
from utilities.aws import download_from_s3, upload_to_s3
from split_data import export_data, load_data, delete_file


def load_model(model_path: str):
    """

    :param model_path:
    :return:
    """
    model = joblib.load(filename=model_path)
    return model


def prediction(model, data):
    """

    :param model:
    :param data:
    :return:
    """
    result = pd.concat([data,
                        pd.DataFrame(np.round(model.predict_proba(data), 3), columns=['probabilidad_sinfuga',
                                                                                      'probabilidad_fuga']),
                        pd.Series(model.predict(data), name='prediccion')], axis=1)
    result.sort_values('probabilidad_fuga', ascending=False, inplace=True)
    return result


def main():
    download_from_s3(bucket=bucket, key=f'{key}/test.csv', dest_pathname='data/test.csv')
    new_data = load_data(filename='data/test.csv', sep=';')
    model = load_model(model_path='models/gboosting.pkl')
    result = prediction(model=model, data=new_data)
    export_data(df=result, path='results/churn_users.csv', with_header=True)
    upload_to_s3(filename='results/churn_users.csv', bucket=bucket, key=f'{key}/results/churn_users.csv')
    delete_file(filename='results/churn_user.csv')
    return None


if __name__ == '__main__':
    main()
