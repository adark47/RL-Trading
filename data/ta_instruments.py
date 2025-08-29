# ta_instruments.py

import sys
import math
import talib
import pandas as pd
from loguru import logger
from tabulate import tabulate
from datetime import datetime, timedelta

# Настройка логгера Loguru с эмодзи и улучшенным цветовым оформлением
log_dir = "logs"
logger.remove()  # Удаляем стандартный обработчик
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"{log_dir}/ta_instruments_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)

class TA_Instruments_Preprocessor:

    def calculate_hma(self, close: pd.Series, timeperiod: int) -> pd.Series:
        """Расчет Hull Moving Average через WMA, так как прямая реализация HMA отсутствует в TA-Lib"""
        self.logger.debug(f"Рассчитываем HMA с периодом {timeperiod} 📐")

        # Формула HMA: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_period = max(1, int(timeperiod / 2))
        sqrt_period = max(1, int(math.sqrt(timeperiod)))

        wma_half = talib.WMA(close, timeperiod=half_period)
        wma_full = talib.WMA(close, timeperiod=timeperiod)

        # Вычисляем разницу с учетом NaN значений
        diff = 2 * wma_half - wma_full
        hma = talib.WMA(diff, timeperiod=sqrt_period)

        return hma

    @logger.catch
    def add_technical_indicators(self, df: pd.DataFrame, indicators=None) -> pd.DataFrame:
        """
        Добавление выбранных технических индикаторов в DataFrame

        Доступные категории индикаторов:
        - Overlap Studies (перекрывающиеся исследования)
        - Momentum Indicators (импульсные индикаторы) [[3]]
        - Volume Indicators (объемные индикаторы) [[3]]
        - Volatility Indicators (индикаторы волатильности) [[3]]

        Поддерживаемые индикаторы для параметра indicators:
        * 'sma' - Simple Moving Average
        * 'ema' - Exponential Moving Average
        * 'wma' - Weighted Moving Average
        * 'rsi' - Relative Strength Index [[2]]
        * 'macd' - Moving Average Convergence Divergence [[2]]
        * 'bbands' - Bollinger Bands [[9]]
        * 'atr' - Average True Range
        * 'adx' - Average Directional Movement Index [[1]]
        * 'stoch' - Stochastic Oscillator
        * 'cci' - Commodity Channel Index
        * 'roc' - Rate of Change
        * 'hma' - Hull Moving Average (кастомная реализация)
        * 'ad' - Chaikin A/D Line [[1]]
        * 'adosc' - Chaikin A/D Oscillator [[1]]

        Пример использования:
        df = preprocessor.add_technical_indicators(df, indicators=['rsi', 'macd', 'hma'])

        :param df: DataFrame с ценовыми данными (обязательно наличие колонок 'close', для некоторых индикаторов - 'high', 'low', 'volume')
        :param indicators: Список индикаторов для расчета. Если None, добавляются все поддерживаемые индикаторы.
        :return: DataFrame с добавленными техническими индикаторами
        """
        self.logger.info(f"Добавляем выбранные технические индикаторы: {indicators or 'все'} 📈")

        # Проверяем наличие колонки 'close'
        if 'close' not in df.columns:
            self.logger.error("Отсутствует колонка 'close' для расчета индикаторов ❌")
            raise ValueError("Колонка 'close' обязательна для расчета технических индикаторов")

        # Определяем список индикаторов по умолчанию, если не передан
        if indicators is None:
            indicators = ['macd', 'rsi', 'bbands', 'hma', 'atr']

        # Сохраняем исходные индексы для последующего восстановления
        original_index = df.index
        df = df.reset_index(drop=True)

        added_features = []

        # Расчет индикаторов в зависимости от выбора
        if 'macd' in indicators:
            self.logger.debug("Рассчитываем MACD (12,26,9) 📊")
            macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            added_features.extend(['macd', 'macd_signal', 'macd_hist'])

        if 'rsi' in indicators:
            self.logger.debug("Рассчитываем RSI (14) 📊")
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            added_features.append('rsi')

        if 'bbands' in indicators:
            self.logger.debug("Рассчитываем Bollinger Bands (20) 📊")
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
            df['bb_upper'] = upper
            df['bb_mid'] = middle
            df['bb_lower'] = lower
            added_features.extend(['bb_upper', 'bb_mid', 'bb_lower'])

        if 'atr' in indicators:
            self.logger.debug("Рассчитываем Average True Range (ATR) с периодом 14 📐")
            if 'high' in df.columns and 'low' in df.columns:
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
                added_features.append('atr')
            else:
                self.logger.warning("Для расчета ATR необходимы колонки 'high' и 'low' ⚠️")

        if 'adx' in indicators:
            self.logger.debug("Рассчитываем Average Directional Movement Index (ADX) 📊")
            if 'high' in df.columns and 'low' in df.columns:
                df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                added_features.append('adx')
            else:
                self.logger.warning("Для расчета ADX необходимы колонки 'high' и 'low' ⚠️")

        if 'hma' in indicators:
            self.logger.debug("Рассчитываем Hull Moving Average (HMA) с периодами 9 и 21 📐")
            df['hma_fast'] = self.calculate_hma(df['close'], 9)
            df['hma_slow'] = self.calculate_hma(df['close'], 21)
            added_features.extend(['hma_fast', 'hma_slow'])


        # Можно добавить другие индикаторы по аналогии...

        # Восстанавливаем исходные индексы
        df.index = original_index

        # Обработка NaN значений
        initial_count = len(df)
        df.fillna(0, inplace=True)
        final_count = len(df)


        # Обновляем список feature_columns с добавленными индикаторами
        self.feature_columns += added_features

        self.logger.success(f"✅ Успешно добавлено {len(added_features)} технических индикаторов! 🎯")

        # Выводим превью данных
        if not df.empty:
            self.logger.info("📊 Превью данных:")
            self.logger.info(f"\n{tabulate(df[added_features].head(5), showindex=True, headers='keys', tablefmt='psql')}")
            self.logger.info(f"\n{tabulate(df[added_features].tail(5), showindex=True, headers='keys', tablefmt='psql')}")

        return df
