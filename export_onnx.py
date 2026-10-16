import argparse

import mlflow
import mlflow.sklearn
import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Промышленный скрипт экспорта модели из MLflow Model Registry в формат ONNX."
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default="models:/pg_execution_time_predictor/1",
        help="MLflow Model URI (например, models:/<model_name>/Staging или Version)"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="Адрес MLflow Tracking Server"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="pg_execution_model.onnx",
        help="Путь для сохранения итогового ONNX-файла"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=15,
        help="Целевая версия ONNX opset"
    )
    return parser.parse_args()


def verify_onnx_model(
        onnx_path: str,
        sample_input: np.ndarray,
        sklearn_preds: np.ndarray
        ):
    """
    Выполняет верификацию: сравнивает предсказания исходной Scikit-learn модели и ONNX Runtime.
    """

    logger.info("Запуск верификации ONNX Runtime...")

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
        )
    input_name = session.get_inputs()[0].name

    # Инференс через ONNX Runtime
    raw_preds = session.run(
        None,
        {input_name: sample_input}
        )
    onnx_preds = raw_preds[0].ravel()

    # Проверка на совпадение результатов с допустимой погрешностью
    np.testing.assert_allclose(
        sklearn_preds,
        onnx_preds,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Ошибка верификации: Предсказания ONNX и Scikit-learn расходятся!"
    )

    logger.success(
        "Верификация успешно пройдена! " \
        "Предсказания ONNX Runtime идентичны оригинальной модели."
        )


def main():
    args = parse_args()

    logger.info("Инициализация процесса экспорта модели в ONNX...")

    # Настройка подключения к MLflow Server
    mlflow.set_tracking_uri(args.tracking_uri)
    logger.info(f"Подключение к MLflow Tracking URI: {args.tracking_uri}")

    logger.info(f"Загрузка модели из реестра: {args.model_uri}")
    try:
        loaded_model = mlflow.sklearn.load_model(args.model_uri)
    except Exception as e:
        logger.error(f"Не удалось загрузить модель по URI {args.model_uri}: {e}")
        raise e

    logger.success("Модель успешно загружена из MLflow Model Registry.")

    # Определение размерности и типов входных данных (контракт модели)
    # Используется FloatTensorType из skl2onnx для строгой типизации (где None обозначает динамический размер батча)
    n_features = loaded_model.n_features_in_
    initial_type = [(
        'float_input',
        FloatTensorType([None, n_features])
        )]
    logger.info(f"Определен контракт входа: батч динамический (None), признаков: {n_features}")

    # Трансляция графа в ONNX
    logger.info(f"Компиляция графа в формат ONNX (opset={args.opset})...")
    try:
        onnx_model = to_onnx(
            loaded_model,
            initial_types=initial_type,
            target_opset=args.opset
        )
    except Exception as e:
        logger.error(f"Ошибка при конвертации модели в ONNX: {e}")
        raise e

    # Проверка корректности структуры ONNX-модели перед сохранением
    onnx.checker.check_model(onnx_model)
    logger.success("Структура ONNX-модели прошла внутреннюю проверку чекером (onnx.checker).")

    # Сохранение бинарного файла на диск
    logger.info(f"Сохранение скомпилированного графа в файл: {args.output_path}")
    with open(args.output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.success(f"Файл успешно сохранен: {args.output_path}")

    # Проверка инференса
    dummy_input = np.random.rand(5, n_features).astype(np.float32)
    sklearn_preds = loaded_model.predict(dummy_input)

    verify_onnx_model(
        args.output_path,
        dummy_input,
        sklearn_preds
        )

    logger.success("Пайплайн экспорта в ONNX полностью завершен без ошибок!")


if __name__ == "__main__":
    main()
