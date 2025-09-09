# ta_strategy.py
import pandas as pd
import numpy as np
import talib
import json
import os
import time  # Импортируем модуль time для измерения времени
from tabulate import tabulate
from loguru import logger

config = 'ta_config_optimized.json'


def load_ta_config(config_path=config):
    """Загрузка конфигурации технических индикаторов"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            logger.info(f"✅ Конфигурация TA загружена из {config_path}")
        else:
            logger.warning(f"⚠️ Файл конфигурации {config_path} не найден, используются значения по умолчанию")
            user_config = {}

        # Значения по умолчанию
        default_config = {
            "atr": {"timeperiod": 14},
            "hma": {"timeperiod": 20},
            "volume_sma": {"timeperiod": 20},
            "bands": {"multiplier": 1.0},
            "risk_management": {
                "take_profit_percent": 5.0,
                "trailing_stop_percent": 2.0,
                "commission_percent": 0.1
            }
        }

        # Объединяем конфигурации (пользовательские значения переопределяют значения по умолчанию)
        config = default_config.copy()
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

        return config
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке конфигурации TA: {str(e)}")
        raise


def hma(series, period):
    """
    Расчет Hull Moving Average (HMA)
    Args:
        series (pd.Series): Входной ряд данных
        period (int): Период HMA
    Returns:
        pd.Series: Значения HMA
    """
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))

    # WMA(period/2)
    wma_half = talib.WMA(series, timeperiod=half_length)
    # WMA(period)
    wma_full = talib.WMA(series, timeperiod=period)

    # 2 * WMA(period/2) - WMA(period)
    raw_hma = 2 * wma_half - wma_full

    # WMA(2 * WMA(period/2) - WMA(period), sqrt(period))
    hma = talib.WMA(raw_hma, timeperiod=sqrt_length)

    return hma


def apply_strategy_single_timeframe(df, config):
    """
    Применение торговой стратегии к одному датафрейму (один таймфрейм)

    Args:
        df (pd.DataFrame): DataFrame с данными OHLC и объемами для одного таймфрейма
        config (dict): Конфигурация индикаторов

    Returns:
        pd.DataFrame: DataFrame с добавленными индикаторами и сигналами
    """
    # Создаем копию для избежания предупреждений SettingWithCopyWarning
    df = df.copy()

    # ATR
    df['atr'] = talib.ATR(
        df['high'],
        df['low'],
        df['close'],
        timeperiod=config['atr']['timeperiod']
    )

    # Подтверждение объемом (если доступно)
    if 'volume' in df.columns:
        df['vol_ma'] = talib.SMA(
            df['volume'],
            timeperiod=config['volume_sma']['timeperiod']
        )
        df['volume_confirmation'] = (df['volume'] > df['vol_ma']).astype(int)
    else:
        df['volume_confirmation'] = 1

    # Расчет Hull Moving Average и каналов на основе конфигурации
    df['hma'] = hma(df['close'], config['hma']['timeperiod'])
    df['upper_band'] = df['hma'] + (df['atr'] * config['bands']['multiplier'])
    df['lower_band'] = df['hma'] - (df['atr'] * config['bands']['multiplier'])

    # Генерация бинарных сигналов (1 - сигнал, 0 - отсутствие сигнала)
    df['entries'] = ((df['close'] > df['upper_band']) & (df['volume_confirmation'] == 1)).astype(int)
    df['exits'] = ((df['close'] < df['lower_band']) | (df['close'] < df['hma'])).astype(int)

    return df


def resample_to_timeframe(df, timeframe_str):
    """
    Ресемплинг 1-минутного датафрейма к заданному таймфрейму.

    Args:
        df (pd.DataFrame): Исходный датафрейм с 1-минутными данными.
        timeframe_str (str): Целевой таймфрейм (например, '5min', '15min').

    Returns:
        pd.DataFrame: Ресемплированный датафрейм.
    """
    try:
        # Работаем с копией
        df_copy = df.copy()

        # Убедимся, что 'date' является столбцом datetime
        if 'date' not in df_copy.columns:
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy['date'] = df_copy.index
            else:
                raise ValueError("Столбец 'date' не найден в DataFrame")

        df_copy['date'] = pd.to_datetime(df_copy['date'])

        # Сортируем по дате
        df_copy = df_copy.sort_values('date')

        # Устанавливаем 'date' как индекс для ресемплинга
        df_copy = df_copy.set_index('date')

        # Ресемплинг
        resampled_df = df_copy.groupby(pd.Grouper(freq=timeframe_str)).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Сброс индекса, чтобы 'date' снова стал столбцом
        resampled_df = resampled_df.reset_index()

        # Убедимся, что 'date' остался столбцом нужного типа
        resampled_df['date'] = pd.to_datetime(resampled_df['date'])

        logger.info(f"✅ Данные успешно ресемплированы до {timeframe_str}")
        return resampled_df

    except Exception as e:
        logger.error(f"❌ Ошибка при ресемплинге до {timeframe_str}: {e}")
        raise


def apply_strategy(df, config_path=config):
    """
    Применение торговой стратегии с техническими индикаторами и риск-менеджментом
    Включает сигналы со старших таймфреймов (5m, 15m) для 1-минутных данных.

    Args:
        df (pd.DataFrame): DataFrame с данными OHLC и объемами (ожидается 1-минутный таймфрейм)
        config_path (str): Путь к конфигурационному файлу

    Returns:
        pd.DataFrame: DataFrame с добавленными индикаторами и сигналами для 1-минутного таймфрейма
    """
    logger.info("📈 Начало применения торговой стратегии с мульти-таймфреймами")

    # --- Записываем время начала выполнения ---
    start_time = time.time()
    # ------------------------------------------

    try:
        # Загрузка конфигурации
        config = load_ta_config(config_path)
        logger.info(f"Загружено {len(df)} записей с {df['date'].min()} по {df['date'].max()}")

        # --- 1. Применение стратегии к 1-минутному таймфрейму ---
        df_1m = df.copy()
        # Убедимся, что 'date' в df_1m является столбцом datetime
        df_1m['date'] = pd.to_datetime(df_1m['date'])
        df_1m_with_signals = apply_strategy_single_timeframe(df_1m, config)
        # Переименуем сигналы для ясности
        df_1m_with_signals.rename(columns={'entries': 'entries_1m', 'exits': 'exits_1m'}, inplace=True)
        logger.info("✅ Сигналы сгенерированы для 1-минутного таймфрейма")

        # --- 2. Ресемплинг до 5-минутного таймфрейма ---
        df_5m = resample_to_timeframe(df, '5min')
        # --- 3. Применение стратегии к 5-минутному таймфрейму ---
        df_5m_with_signals = apply_strategy_single_timeframe(df_5m, config)
        df_5m_with_signals.rename(columns={'entries': 'entries_5m', 'exits': 'exits_5m'}, inplace=True)
        logger.info("✅ Сигналы сгенерированы для 5-минутного таймфрейма")

        # --- 4. Ресемплинг до 15-минутного таймфрейма ---
        df_15m = resample_to_timeframe(df, '15min')
        # --- 5. Применение стратегии к 15-минутному таймфрейму ---
        df_15m_with_signals = apply_strategy_single_timeframe(df_15m, config)
        df_15m_with_signals.rename(columns={'entries': 'entries_15m', 'exits': 'exits_15m'}, inplace=True)
        logger.info("✅ Сигналы сгенерированы для 15-минутного таймфрейма")

        # --- 6. Синхронизация сигналов со старших таймфреймов на 1-минутный ---
        # Убедимся, что 'date' в основном датафрейме является datetime
        df_result = df_1m_with_signals.copy()
        df_result['date'] = pd.to_datetime(df_result['date'])
        df_result = df_result.sort_values('date').reset_index(drop=True)  # Сортировка на всякий случай

        # Подготовка датафреймов для мержа: убедиться, что 'date' - столбец
        df_5m_with_signals['date'] = pd.to_datetime(df_5m_with_signals['date'])
        df_15m_with_signals['date'] = pd.to_datetime(df_15m_with_signals['date'])

        # Синхронизация 5-минутных сигналов
        df_result = df_result.merge(df_5m_with_signals[['date', 'entries_5m', 'exits_5m']],
                                    on='date', how='left')
        # Заполнение вперед (ffill) для распространения последнего сигнала 5m
        df_result['entries_5m'] = df_result['entries_5m'].ffill().fillna(0).astype(int)
        df_result['exits_5m'] = df_result['exits_5m'].ffill().fillna(0).astype(int)

        # Синхронизация 15-минутных сигналов
        df_result = df_result.merge(df_15m_with_signals[['date', 'entries_15m', 'exits_15m']],
                                    on='date', how='left')
        # Заполнение вперед (ffill) для распространения последнего сигнала 15m
        df_result['entries_15m'] = df_result['entries_15m'].ffill().fillna(0).astype(int)
        df_result['exits_15m'] = df_result['exits_15m'].ffill().fillna(0).astype(int)

        # --- 7. Комбинирование сигналов (логическое И для входов, ИЛИ для выходов) ---
        # Вход: сигнал должен быть на всех таймфреймах
        df_result['entries'] = (
                (df_result['entries_1m'] == 1) &
                (df_result['entries_5m'] == 1) &
                (df_result['entries_15m'] == 1)
        ).astype(int)

        # Выход: сигнал хотя бы на одном таймфрейме
        df_result['exits'] = (
                (df_result['exits_1m'] == 1) |
                (df_result['exits_5m'] == 1) |
                (df_result['exits_15m'] == 1)
        ).astype(int)

        # --- 8. Добавление дополнительных признаков (рассчитываются на 1m данных) ---
        # Нормализованные значения индикаторов (используем ATR и полосы от 1m)
        df_result['norm_atr'] = df_result['atr'] / df_result['close']  # Нормализованный ATR
        df_result['price_position'] = (df_result['close'] - df_result['lower_band']) / (
                df_result['upper_band'] - df_result['lower_band'])  # Позиция цены в канале (1m)
        df_result['price_position'] = np.clip(df_result['price_position'], 0, 1)  # Ограничение

        # Признаки риск-менеджмента (используем параметры из конфига, примененные к 1m данным)
        df_result['take_profit_level'] = df_result['close'] * (
                1 + config['risk_management']['take_profit_percent'] / 100)
        df_result['trailing_stop_distance'] = df_result['close'] * (
                config['risk_management']['trailing_stop_percent'] / 100)
        df_result['commission'] = config['risk_management']['commission_percent'] / 100

        # --- Записываем время окончания и вычисляем длительность ---
        end_time = time.time()
        execution_time = end_time - start_time
        logger.success(f"✅ Стратегия с мульти-таймфреймами успешно применена за {execution_time:.2f} секунд")
        # ----------------------------------------------------------

        # Превью результата
        logger.info("📋 Результат стратегии (последние 10 строк):")
        logger.info(
            f"Превью данных: \n{tabulate(df_result.tail(10), headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False)}")

        # Заполняем NaN нулями (на всякий случай, хотя ffill должен помочь)
        df_result = df_result.fillna(0)

        return df_result

    except Exception as e:
        # --- В случае ошибки также можно залогировать время (до возникновения ошибки) ---
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(
            f"❌ Ошибка при применении стратегии с мульти-таймфреймами (выполнено за {execution_time:.2f} секунд): {str(e)}")
        # ----------------------------------------------------------------------------------
        raise
