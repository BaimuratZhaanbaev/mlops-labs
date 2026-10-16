import argparse
import time

import numpy as np
import onnxruntime as ort
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Промышленный скрипт инференса модели через ONNX Runtime."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="pg_execution_model.onnx",
        help="Путь к скомпилированному ONNX файлу модели"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Инициализация инференса (ONNX Runtime)...")

    # Настройка провайдеров выполнения
    providers = ["CPUExecutionProvider"]

    try:
        # Инициализация сессии инференса
        session = ort.InferenceSession(args.model_path, providers=providers)
        logger.success(f"ONNX сессия успешно инициализирована с провайдерами: {providers}")
    except Exception as e:
        logger.error(f"Не удалось загрузить ONNX модель по пути {args.model_path}: {e}")
        raise e

    # Получение контракта входа/выхода из модели
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    input_type = input_meta.type

    output_meta = session.get_outputs()[0]
    output_name = output_meta.name

    logger.info(
        f"Контракт модели [Вход] -> Имя: '{input_name}', Shape: {input_shape}, Type: {input_type}"
        )
    logger.info(
        f"Контракт модели [Выход] -> Имя: '{output_name}', Shape: {output_meta.shape}"
        )

    # Подготовка тестовых данных
    n_features = input_shape[1] if isinstance(input_shape[1], int) else 4

    # Создаем батч из 3 тестовых запросов
    dummy_input = np.random.rand(3, n_features).astype(np.float32)
    logger.info(
        f"Сформирован тестовый батч для предсказания. Размерность: {dummy_input.shape}"
        )

    # Запуск инференса
    logger.info("Выполнение инференса через ONNX Runtime...")

    start_time = time.perf_counter()
    try:
        # Передаем данные в виде словаря: { имя_входного_узла : numpy_array }
        raw_predictions = session.run([output_name], {input_name: dummy_input})
    except Exception as e:
        logger.error(f"Ошибка во время выполнения инференса: {e}")
        raise e

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
    logger.success(f"Инференс успешно выполнен за {elapsed_time_ms:.4f} мс.")

    # Обработка результатов
    predictions = raw_predictions[0]

    logger.info("Результаты инференса (предсказания модели):")
    for i, pred in enumerate(predictions):
        logger.info(f"  Объект #{i+1} -> Предсказанное значение: {pred}")


if __name__ == "__main__":
    main()
