# find_best_model.py

import mlflow
import platform
try:
    # Пытаемся импортировать fireducks.pandas только на Linux
    if platform.system().lower() == 'linux':
        import fireducks.pandas as pd
        print("Загружен fireducks.pandas")
    else:
        raise ImportError
except ImportError:
    import pandas as pd
    print("Загружен стандартный pandas")
from loguru import logger
from tabulate import tabulate
import warnings
import datetime
import sys
import os

# Добавляем родительскую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Импортируем конфигурацию
from config import MLflowConfig as MLflowConfig

# Создаем экземпляр конфигурации
settings = MLflowConfig()

warnings.filterwarnings('ignore')

os.makedirs("logs", exist_ok=True)

# Настройка логгера с цветами и эмоджи
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"logs/find_best_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)

# Конфигурация MLflow
mlflow.set_tracking_uri(settings.tracking_uri)
mlflow.set_experiment(settings.name_train_experiment)


def get_all_runs():
    """Получить все запуски из эксперимента"""
    logger.info("🚀 Получение всех запусков из эксперимента")

    # Получаем эксперимент
    experiment = mlflow.get_experiment_by_name("rl_bybit_futures_trading")
    if experiment is None:
        logger.error("❌ Эксперимент 'rl_bybit_futures_trading' не найден")
        return None

    experiment_id = experiment.experiment_id
    logger.info(f"📁 ID эксперимента: {experiment_id}")

    # Получаем все запуски
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        logger.warning("⚠️  В эксперименте нет запусков")
        return None

    logger.success(f"✅ Найдено {len(runs)} запусков")
    return runs, experiment


def filter_runs_with_win_rates(runs):
    """Отфильтровать запуски, оставив только те, где есть метрика train_win_rate"""
    logger.info("🔍 Фильтрация запусков по наличию метрики train_win_rate")

    # Проверяем, какие колонки содержат train_win_rate
    win_rates_cols = [col for col in runs.columns if 'train_win_rate' in col]

    if not win_rates_cols:
        logger.error("❌ Метрика train_win_rate не найдена ни в одном запуске")
        return None

    logger.info(f"📊 Найдены колонки с train_win_rate: {win_rates_cols}")

    # Фильтруем запуски, где есть значения train_win_rate
    filtered_runs = runs.dropna(subset=win_rates_cols, how='all')

    logger.success(f"✅ Отфильтровано {len(filtered_runs)} запусков с метрикой train_win_rate")
    return filtered_runs


def display_runs_preview(runs, experiment, top_n=10):
    """Отобразить превью запусков с использованием tabulate, включая дату запуска и ссылку"""
    logger.info(f"📋 Превью первых {top_n} запусков")

    # Выбираем ключевые колонки для отображения
    key_columns = ['run_id', 'experiment_id', 'start_time']

    # Добавляем колонки с метриками
    metric_columns = [col for col in runs.columns if col.startswith('metrics.')]
    key_columns.extend(metric_columns[:5])  # Показываем первые 5 метрик

    # Добавляем колонки с параметрами
    param_columns = [col for col in runs.columns if col.startswith('params.')]
    key_columns.extend(param_columns[:5])  # Показываем первые 5 параметров

    # Убираем дубликаты и ограничиваем список
    key_columns = list(dict.fromkeys(key_columns))[:15]

    # Создаем DataFrame для отображения
    preview_df = runs[key_columns].head(top_n).copy()

    # Преобразуем дату в читаемый формат
    if 'start_time' in preview_df.columns:
        preview_df['start_time'] = pd.to_datetime(preview_df['start_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Добавляем колонку со ссылками на запуски
    if 'run_id' in preview_df.columns:
        tracking_uri = settings.tracking_uri
        experiment_id = settings.name_train_experiment
        preview_df['run_link'] = preview_df['run_id'].apply(
            lambda run_id: f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
        )

    # Преобразуем для лучшего отображения
    preview_df_display = preview_df.copy()
    for col in preview_df_display.columns:
        if preview_df_display[col].dtype == 'object':
            max_len = preview_df_display[col].astype(str).map(len).max()
            if max_len > 50:
                preview_df_display[col] = preview_df_display[col].astype(str).apply(
                    lambda x: x[:47] + '...' if len(x) > 50 else x
                )

    print(tabulate(preview_df_display, headers='keys', tablefmt='fancy_grid', showindex=False))
    logger.success("✅ Превью запусков отображено")


def find_best_model_by_win_rates(runs, experiment):
    """Найти лучшую модель по метрике train_win_rate"""
    logger.info("🥇 Поиск лучшей модели по метрике train_win_rate")

    # Ищем колонку с train_win_rate
    win_rates_col = None
    for col in runs.columns:
        if 'train_win_rate' in col:
            win_rates_col = col
            break

    if win_rates_col is None:
        logger.error("❌ Не удалось найти колонку с метрикой train_win_rate")
        return None

    logger.info(f"📊 Используется колонка: {win_rates_col}")

    # Находим запуск с максимальным значением train_win_rate
    best_run = runs.loc[runs[win_rates_col].idxmax()]
    best_win_rate = best_run[win_rates_col]
    best_run_id = best_run['run_id']

    # Получаем дату лучшего запуска
    best_run_date = best_run.get('start_time', 'N/A')
    if pd.notna(best_run_date):
        best_run_date = pd.to_datetime(best_run_date).strftime('%Y-%m-%d %H:%M:%S')
    else:
        best_run_date = 'N/A'

    # Формируем ссылку на лучший запуск
    tracking_uri = settings.tracking_uri
    experiment_id = settings.name_train_experiment
    best_run_link = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{best_run_id}"

    logger.success(f"🏆 Лучшая модель найдена!")
    logger.success(f"🆔 ID лучшего запуска: {best_run_id}")
    logger.success(f"📅 Дата лучшего запуска: {best_run_date}")
    logger.success(f"📈 Значение train_win_rate: {best_win_rate}")
    logger.success(f"🔗 Ссылка на запуск: {best_run_link}")

    return best_run, best_run_link


def check_model_artifacts(run_id):
    """Проверить наличие артефактов модели в запуске"""
    logger.info(f"🔍 Проверка наличия артефактов модели для запуска {run_id}")

    try:
        # Получаем клиент MLflow
        client = mlflow.tracking.MlflowClient()

        # Получаем список артефактов
        artifacts = client.list_artifacts(run_id)

        if not artifacts:
            logger.warning(f"⚠️  В запуске {run_id} не найдено артефактов")
            return False

        # Ищем модель в артефактах
        model_artifacts = [artifact for artifact in artifacts if 'model' in artifact.path.lower()]

        if not model_artifacts:
            logger.warning(f"⚠️  В запуске {run_id} не найдено артефактов модели")
            # Выводим все доступные артефакты
            logger.info(f"📄 Доступные артефакты в запуске {run_id}:")
            for artifact in artifacts:
                logger.info(f"   - {artifact.path}")
            return False

        logger.success(f"✅ Найдены артефакты модели: {[a.path for a in model_artifacts]}")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при проверке артефактов: {str(e)}")
        return False


def load_best_model(best_run):
    """Загрузить лучшую модель с улучшенной обработкой ошибок"""
    logger.info("📥 Попытка загрузки лучшей модели")

    try:
        # Получаем run_id
        run_id = best_run['run_id']
        logger.info(f"📦 Загрузка модели из запуска: {run_id}")

        # Проверяем наличие артефактов модели
        if not check_model_artifacts(run_id):
            logger.error(f"❌ В запуске {run_id} отсутствуют необходимые артефакты модели")
            return None

        # Попытка загрузки модели по разным путям
        model_paths = [
            f"runs:/{run_id}/model",
            f"runs:/{run_id}/model/model",
            f"runs:/{run_id}"
        ]

        model = None
        for path in model_paths:
            try:
                logger.info(f"⬇️  Попытка загрузки модели по пути: {path}")
                model = mlflow.pyfunc.load_model(path)
                logger.success(f"✅ Модель успешно загружена по пути: {path}")
                break
            except Exception as e:
                logger.warning(f"⚠️  Не удалось загрузить модель по пути {path}: {str(e)}")
                continue

        if model is None:
            logger.error(f"❌ Не удалось загрузить модель из запуска {run_id} по всем возможным путям")
            return None

        logger.success(f"✅ Модель успешно загружена из запуска {run_id}")
        return model

    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке модели: {str(e)}")
        logger.exception(e)  # Полная трассировка ошибки
        return None


def main():
    """Основная функция"""
    logger.info("🚀 Начало процесса выбора лучшей модели")

    # Получаем все запуски
    result = get_all_runs()
    if result is None:
        return

    runs, experiment = result

    # Фильтруем запуски по наличию train_win_rate
    filtered_runs = filter_runs_with_win_rates(runs)
    if filtered_runs is None:
        return

    # Отображаем превью запусков с ссылками
    display_runs_preview(filtered_runs, experiment)

    # Находим лучшую модель
    result = find_best_model_by_win_rates(filtered_runs, experiment)
    if result is None:
        return

    best_run, best_run_link = result

    # Выводим детальную информацию о лучшем запуске
    logger.info("📄 Детальная информация о лучшем запуске:")
    best_run_info = best_run.to_dict()

    # Выводим ключевую информацию, включая дату и ссылку
    important_fields = ['run_id', 'experiment_id', 'start_time']
    win_rate_fields = [key for key in best_run_info.keys() if 'train_win_rate' in key]
    important_fields.extend(win_rate_fields)

    for key in important_fields:
        value = best_run_info.get(key, 'N/A')
        # Форматируем дату, если это необходимо
        if key == 'start_time' and pd.notna(value):
            try:
                value = pd.to_datetime(value).strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        logger.info(f"   {key}: {value}")

    # Выводим ссылку на эксперимент
    logger.info(f"   experiment_link: {best_run_link}")

    # Загружаем лучшую модель
    model = load_best_model(best_run)
    if model is not None:
        logger.success("🎉 Процесс выбора и загрузки лучшей модели завершен успешно")
        return model
    else:
        logger.error("💥 Процесс завершен с ошибками при загрузке модели")
        return None


if __name__ == "__main__":
    model = main()
    if model:
        logger.info("🏁 Модель готова к использованию")
    else:
        logger.error("🏁 Работа программы завершена с ошибками")