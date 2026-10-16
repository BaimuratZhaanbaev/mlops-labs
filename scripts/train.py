import argparse
import os

import dvc.api
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PostgreSQL execution time predictor."
        )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/dataset.csv",
        help="Path to versioned CSV dataset"
        )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLflow Tracking URI"
        )
    return parser.parse_args()

def main():
    args = parse_args()

    logger.info("Инициализация пайплайна обучения модели...")

    # Загружаем параметры через официальный DVC API
    logger.info("Загрузка параметров из params.yaml через DVC API...")
    params = dvc.api.params_show()
    train_params = params.get("train", {})

    pearent_run_name = train_params.get("run_name", "DecisionTree_Default_Run")
    max_depths = train_params.get("max_depth", [3, 5, 10])
    min_samples_splits = train_params.get("min_samples_split", [2, 5])
    test_size = train_params.get("test_size", 0.2)
    cv_folds = train_params.get("cv_folds", 3)
    random_state = train_params.get("random_state", 42)

    # Настройка MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("pg_execution_time_prediction")
    logger.success(f"Подключение к MLflow Server установлено: {args.tracking_uri}")

    mlflow.sklearn.autolog(
        log_models=False,
        silent=True
        )

    # Проверка и загрузка данных
    if not os.path.exists(args.data_path):
        logger.error(f"Файл данных не найден по пути: {args.data_path}")
        raise FileNotFoundError(f"Файл {args.data_path} не найден. Выполните dvc checkout!")

    logger.info(f"Загрузка датасета из {args.data_path}...")
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["actual_time_ms"])
    y = df["actual_time_ms"]
    logger.success(
        f"Датасет успешно загружен. Размерность: {X.shape[0]} строк, {X.shape[1]} признаков."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    logger.info(
        f"Данные разбиты на выборки: Train={len(X_train)}, Test={len(X_test)}"
        )

    # GridSearchCV
    param_grid = {
        'max_depth': max_depths,
        'min_samples_split': min_samples_splits
    }

    base_model = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=cv_folds,
        n_jobs=-1
    )

    # MLflow Parent Run
    with mlflow.start_run(run_name=pearent_run_name):
        mlflow.log_params({
            "pipeline_type": "GridSearchCV",
            "dataset_version": "v2.0",
            "benchmark": "JOB_and_CEB_IMDb"
        })

        logger.info("Запуск автоматического Grid Search по сетке параметров...")
        grid_search.fit(X_train, y_train)
        logger.success("Grid Search успешно завершен!")

        best_model = grid_search.best_estimator_
        logger.info(f"Лучшие найденные гиперпараметры: {grid_search.best_params_}")

        # Оценка лучшей модели на тестовом наборе
        predictions = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        test_mae = mean_absolute_error(y_test, predictions)
        test_r2 = r2_score(y_test, predictions)

        logger.info(
            f"Метрики на тестовой выборке -> RMSE: {test_rmse:.2f} ms | MAE: {test_mae:.2f} ms | R2: {test_r2:.4f}"
            )

        mlflow.log_metrics({
            "test_rmse_ms": test_rmse,
            "test_mae_ms": test_mae,
            "test_r2_score": test_r2
        })

        signature = mlflow.models.infer_signature(X_test, predictions)

        # Артефакт-график
        logger.info("Генерация графика распределения предсказаний (Sanity Check)...")
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.scatter(
            y_test,
            predictions,
            alpha=0.6,
            color='forestgreen',
            edgecolor='k'
            )
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--',
            lw=2
            )
        plt.xlabel("Фактическое время исполнения (мс)")
        plt.ylabel("Прогнозируемое время исполнения (мл)")
        plt.title("Лучшее исполнение дерева решений")
        plt.grid(
            True,
            linestyle='--',
            alpha=0.5
            )
        plot_path = "plots/best_model_scatter.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)
        logger.success(
            f"График успешно сохранен и отправлен в MLflow артефакты: {plot_path}"
            )

        # Логирование и регистрация модели в Model Registry
        logger.info(
            "Регистрация модели в MLflow Model Registry (реестр моделей)..."
            )
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="pg_time_predictor_model",
            signature=signature,
            registered_model_name="PG_Execution_Time_Tree_Model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        logger.success("Пайплайн обучения и регистрации модели успешно завершен!")

if __name__ == "__main__":
    main()
