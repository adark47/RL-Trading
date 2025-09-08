# ta_strategy.py

import pandas as pd
import numpy as np
import talib
import json
import os
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
            # Изменено: параметры для HMA вместо SMA
            "hma": {"timeperiod": 20},
            "volume_sma": {"timeperiod": 20},
            "bands": {"multiplier": 1.0},
            "supertrend": {
                "atr_period": 10,
                "atr_multiplier": 3.0
            },
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

def supertrend(df, atr_period=10, atr_multiplier=3.0):
    """
    Расчет SuperTrend индикатора

    Args:
        df (pd.DataFrame): DataFrame с данными OHLC
        atr_period (int): Период для расчета ATR
        atr_multiplier (float): Множитель ATR

    Returns:
        tuple: (supertrend_values, direction)
    """
    # Рассчитываем ATR
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)

    # Рассчитываем базовые линии
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)

    # Инициализация массивов
    supertrend = np.zeros(len(df))
    direction = np.ones(len(df))  # 1 - вверх, -1 - вниз

    # Начальные значения
    supertrend[0] = upper_band.iloc[0]
    direction[0] = 1 if df['close'].iloc[0] > supertrend[0] else -1

    for i in range(1, len(df)):
        # Если текущая цена выше предыдущего SuperTrend и предыдущий был направлен вверх
        if df['close'].iloc[i] > supertrend[i - 1] and direction[i - 1] == 1:
            supertrend[i] = max(lower_band.iloc[i], supertrend[i - 1])
            direction[i] = 1
        # Если текущая цена ниже предыдущего SuperTrend и предыдущий был направлен вниз
        elif df['close'].iloc[i] < supertrend[i - 1] and direction[i - 1] == -1:
            supertrend[i] = min(upper_band.iloc[i], supertrend[i - 1])
            direction[i] = -1
        # Если цена пересекла SuperTrend вверх
        elif df['close'].iloc[i] > supertrend[i - 1]:
            supertrend[i] = lower_band.iloc[i]
            direction[i] = 1
        # Если цена пересекла SuperTrend вниз
        else:
            supertrend[i] = upper_band.iloc[i]
            direction[i] = -1

    return pd.Series(supertrend, index=df.index), pd.Series(direction, index=df.index)


def apply_strategy(df, config_path=config):
    """
    Применение торговой стратегии с техническими индикаторами и риск-менеджментом

    Args:
        df (pd.DataFrame): DataFrame с данными OHLC и объемами
        config_path (str): Путь к конфигурационному файлу

    Returns:
        pd.DataFrame: DataFrame с добавленными индикаторами и сигналами
    """
    logger.info("📈 Начало применения торговой стратегии")

    try:
        # Загрузка конфигурации
        config = load_ta_config(config_path)

        # Создаем копию для избежания предупреждений SettingWithCopyWarning
        df = df.copy()
        logger.info(f"Загружено {len(df)} записей с {df['date'].min()} по {df['date'].max()}")

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
        # Изменено: используем HMA вместо SMA
        df['hma'] = hma(df['close'], config['hma']['timeperiod'])
        df['upper_band'] = df['hma'] + (df['atr'] * config['bands']['multiplier'])
        df['lower_band'] = df['hma'] - (df['atr'] * config['bands']['multiplier'])

        # SuperTrend для риск-менеджмента
        st_period = config['supertrend']['atr_period']
        st_multiplier = config['supertrend']['atr_multiplier']
        df['supertrend'], df['st_direction'] = supertrend(df, st_period, st_multiplier)

        # Генерация бинарных сигналов (1 - сигнал, 0 - отсутствие сигнала)
        # Изменено: используем hma для сигналов
        df['entries'] = ((df['close'] > df['upper_band']) & (df['volume_confirmation'] == 1)).astype(int)
        df['exits'] = ((df['close'] < df['lower_band']) | (df['close'] < df['hma'])).astype(int)

        # Дополнительные признаки для ML модели
        # Нормализованные значения индикаторов
        df['norm_atr'] = df['atr'] / df['close']  # Нормализованный ATR
        # Изменено: используем hma для расчета позиции цены
        df['price_position'] = (df['close'] - df['lower_band']) / (
                df['upper_band'] - df['lower_band'])  # Позиция цены в канале

        # Ограничение значений от 0 до 1 для предотвращения экстремальных значений
        df['price_position'] = np.clip(df['price_position'], 0, 1)

        # Добавляем признаки риск-менеджмента
        df['take_profit_level'] = df['close'] * (1 + config['risk_management']['take_profit_percent'] / 100)
        df['trailing_stop_distance'] = df['close'] * (config['risk_management']['trailing_stop_percent'] / 100)

        # Комиссия биржи (в процентах)
        df['commission'] = config['risk_management']['commission_percent'] / 100

        logger.success("✅ Стратегия успешно применена")

        # Превью результата
        logger.info("📋 Результат стратегии (последние 10 строк):")
        logger.info(
            f"Превью данных: \n{tabulate(df.tail(10), headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False)}")

        # Заполняем NaN нулями
        df = df.fillna(0)

        return df

    except Exception as e:
        logger.error(f"❌ Ошибка при применении стратегии: {str(e)}")
        raise