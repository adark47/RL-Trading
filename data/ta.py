import pandas as pd
import numpy as np
import talib
from tabulate import tabulate
from loguru import logger
import datetime
import sys
import os
import json
from itertools import product

# Создаем директорию для логов
os.makedirs("logs", exist_ok=True)
os.makedirs("optimization_results", exist_ok=True)

# Настройка логгера с цветами и эмоджи
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>",
    level="INFO",
    colorize=True
)
logger.add(
    f"logs/ta_strategy_preprocessing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function}:{line} - {message}",
    level="DEBUG"
)


# -------------------------------
# Функция для вычисления Hull Moving Average (HMA)
# -------------------------------
def hull_moving_average(src, length):
    """Вычисление Hull Moving Average"""
    if length <= 0:
        raise ValueError("Длина HMA должна быть положительной")

    wma1 = talib.WMA(src, length // 2)
    wma2 = talib.WMA(src, length)
    hma = talib.WMA(2 * wma1 - wma2, int(np.sqrt(length)))
    return hma


# -------------------------------
# Функция для получения дополнительной статистики
# -------------------------------
def get_strategy_stats(df):
    """Получение статистики по стратегии"""
    total_signals = len(df[df['decision'] != 0])
    buy_signals = len(df[df['decision'] == 1])
    sell_signals = len(df[df['decision'] == -1])

    stats = {
        'total_signals': total_signals,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'signal_frequency': f"{(total_signals / len(df) * 100):.2f}%"
    }
    return stats


# -------------------------------
# Функция для обнаружения дивергенций RSI
# -------------------------------
def detect_rsi_divergence(df, period=5):
    """Обнаружение дивергенций RSI"""
    if len(df) < period:
        return pd.Series([False] * len(df)), pd.Series([False] * len(df))

    # Бычья дивергенция (цена падает, RSI растет)
    price_lows = df['low'].rolling(period, min_periods=1).min()
    rsi_lows = df['rsi'].rolling(period, min_periods=1).min()

    # Новые минимумы цены, но не новые минимумы RSI
    bullish_div = (df['low'] <= price_lows.shift(1)) & (df['rsi'] >= rsi_lows.shift(1))

    # Медвежья дивергенция (цена растет, RSI падает)
    price_highs = df['high'].rolling(period, min_periods=1).max()
    rsi_highs = df['rsi'].rolling(period, min_periods=1).max()

    bearish_div = (df['high'] >= price_highs.shift(1)) & (df['rsi'] <= rsi_highs.shift(1))

    return bullish_div.fillna(False), bearish_div.fillna(False)


# -------------------------------
# Функция для получения параметров по умолчанию
# -------------------------------
def get_default_params():
    """Параметры по умолчанию"""
    return {
        'hma_fast': 12,
        'hma_slow': 26,
        'rsi_len': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'atr_len': 14
    }


# -------------------------------
# Функция для получения диапазонов параметров для оптимизации
# -------------------------------
def get_parameter_ranges():
    """Диапазоны параметров для оптимизации"""
    return {
        'hma_fast': [7, 9, 12, 15, 21],
        'hma_slow': [14, 21, 26, 34, 50],
        'rsi_len': [9, 14, 21],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'macd_fast': [8, 12, 17],
        'macd_slow': [17, 26, 34],
        'macd_signal': [9, 12],
        'atr_len': [14, 21]
    }


# -------------------------------
# Функция для адаптивных параметров (с оптимизацией)
# -------------------------------
def adaptive_parameters(df, optimization_result_file="optimization_results/best_params.json"):
    """Адаптация параметров под волатильность или использование оптимизированных параметров"""

    # Проверяем наличие файла с оптимизированными параметрами
    if os.path.exists(optimization_result_file):
        try:
            with open(optimization_result_file, 'r') as f:
                best_params = json.load(f)
            logger.info(f"✅ Загружены оптимизированные параметры из {optimization_result_file}")
            logger.info(f"🏆 Лучшие параметры: {best_params}")
            return best_params
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки оптимизированных параметров: {e}")

    # Если файла нет, используем адаптивную логику
    logger.info("🔄 Используется адаптивная настройка параметров")

    if len(df) < 50:
        return get_default_params()

    if 'volatility' not in df.columns:
        df['volatility'] = df['close'].pct_change().rolling(20, min_periods=1).std()

    # Используем последние 50 значений для расчета
    recent_vol = df['volatility'].tail(50).dropna()
    if len(recent_vol) == 0:
        return get_default_params()

    avg_volatility = recent_vol.mean()

    if pd.isna(avg_volatility):
        return get_default_params()

    # Адаптируем параметры
    if avg_volatility > 0.03:  # Высокая волатильность
        params = {
            'hma_fast': 7,
            'hma_slow': 14,
            'rsi_len': 9,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'macd_fast': 8,
            'macd_slow': 17,
            'macd_signal': 9,
            'atr_len': 14
        }
    elif avg_volatility > 0.015:  # Средняя волатильность
        params = {
            'hma_fast': 12,
            'hma_slow': 26,
            'rsi_len': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_len': 14
        }
    else:  # Низкая волатильность
        params = {
            'hma_fast': 21,
            'hma_slow': 50,
            'rsi_len': 14,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_len': 14
        }

    return params


# -------------------------------
# Функция для временных фильтров
# -------------------------------
def add_time_filters(df):
    """Добавление временных фильтров"""
    # Избегать сигналов в определенные часы (если данные по часам)
    if hasattr(df['date'].iloc[0], 'hour'):
        df['hour'] = df['date'].dt.hour
        df['weekday'] = df['date'].dt.weekday

        # Избегать торговли в ночные часы или в понедельник
        df['time_filter'] = (df['hour'] >= 8) & (df['hour'] <= 20) & (df['weekday'] != 0)
    else:
        df['time_filter'] = True

    return df


# -------------------------------
# Функция для управления рисками
# -------------------------------
def add_risk_management(df, stop_loss_pct=0.02, take_profit_pct=0.04):
    """Добавление стоп-лосса и тейк-профита"""
    df['stop_loss_level'] = df['close'] * (1 - stop_loss_pct)
    df['take_profit_level'] = df['close'] * (1 + take_profit_pct)
    return df


# -------------------------------
# УЛУЧШЕННАЯ Функция для расчета метрик производительности с учетом комиссий
# -------------------------------
def calculate_performance_metrics_with_commission(df, commission_rate=0.0004, initial_capital=10000):
    """
    Расчет расширенных метрик производительности с учетом комиссий

    Args:
        df (pd.DataFrame): Данные с сигналами
        commission_rate (float): Комиссия брокера (по умолчанию 0.1% = 0.001)
        initial_capital (float): Начальный капитал для расчета абсолютных метрик

    Returns:
        dict: Метрики производительности
    """
    signals = df[df['decision'] != 0].copy()

    if len(signals) < 2:
        return {}

    # Симуляция сделок с учетом комиссий
    positions = []
    equity_curve = [initial_capital]  # Кривая капитала
    current_position = None
    total_commission_paid = 0
    current_capital = initial_capital

    for idx, row in signals.iterrows():
        if row['decision'] == 1 and current_position is None:  # Покупка
            # Комиссия при покупке
            commission_buy = current_capital * commission_rate
            total_commission_paid += commission_buy

            # Сколько акций можем купить
            capital_after_commission = current_capital - commission_buy
            shares_bought = capital_after_commission / row['close']

            current_position = {
                'entry_price': row['close'],
                'entry_time': row['date'],
                'type': 'long',
                'shares': shares_bought,
                'commission_buy': commission_buy
            }

        elif row['decision'] == -1 and current_position is not None:  # Продажа
            if current_position['type'] == 'long':
                # Комиссия при продаже
                proceeds = current_position['shares'] * row['close']
                commission_sell = proceeds * commission_rate
                total_commission_paid += commission_sell

                # Расчет прибыли с учетом комиссий
                pnl_gross = (row['close'] - current_position['entry_price']) / current_position['entry_price']
                # Вычитаем комиссии (в процентах от начальной цены)
                commission_total_pct = (current_position['commission_buy'] + commission_sell) / (
                            current_position['shares'] * current_position['entry_price'])
                pnl_net = pnl_gross - commission_total_pct

                # Абсолютная прибыль
                absolute_pnl = proceeds - commission_sell - (
                            current_position['shares'] * current_position['entry_price']) - current_position[
                                   'commission_buy']

                # Обновляем капитал
                current_capital = current_capital + absolute_pnl
                equity_curve.append(current_capital)

                positions.append({
                    'entry_price': current_position['entry_price'],
                    'exit_price': row['close'],
                    'pnl_gross': pnl_gross,
                    'pnl_net': pnl_net,
                    'commission_paid': current_position['commission_buy'] + commission_sell,
                    'entry_time': current_position['entry_time'],
                    'exit_time': row['date'],
                    'absolute_pnl': absolute_pnl
                })
                current_position = None

    if not positions:
        return {}

    # Извлекаем чистые и валовые доходности
    returns_gross = [p['pnl_gross'] for p in positions]
    returns_net = [p['pnl_net'] for p in positions]
    absolute_pnls = [p['absolute_pnl'] for p in positions]

    if not returns_net:
        return {}

    # Метрики с учетом комиссий
    win_rate_net = len([r for r in returns_net if r > 0]) / len(returns_net) if returns_net else 0
    avg_return_net = np.mean(returns_net) if returns_net else 0
    std_returns_net = np.std(returns_net) if len(returns_net) > 1 else 0

    # Годовой коэффициент Шарпа (предполагаем 252 торговых дня)
    sharpe_ratio_net = (np.mean(returns_net) / std_returns_net * np.sqrt(252)) if std_returns_net > 0 else 0

    # Максимальная просадка по кривой капитала
    equity_array = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_drawdown_net = np.max(drawdown) if len(drawdown) > 0 else 0

    total_return_net = (equity_curve[-1] - initial_capital) / initial_capital if initial_capital > 0 else 0

    # Абсолютные метрики
    total_absolute_pnl = sum(absolute_pnls)
    avg_absolute_pnl = np.mean(absolute_pnls) if absolute_pnls else 0

    # Метрики без учета комиссий (для сравнения)
    win_rate_gross = len([r for r in returns_gross if r > 0]) / len(returns_gross) if returns_gross else 0
    total_return_gross = np.prod([1 + r for r in returns_gross]) - 1 if returns_gross else 0

    # Общая комиссия
    total_commission = sum([p['commission_paid'] for p in positions])

    # Коэффициент прибыльности (Profit Factor)
    gross_profits = sum([p for p in absolute_pnls if p > 0])
    gross_losses = abs(sum([p for p in absolute_pnls if p < 0]))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    # Максимальное количество последовательных прибыльных/убыточных сделок
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_streak = 0

    for pnl in absolute_pnls:
        if pnl > 0:
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        elif pnl < 0:
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))

    consecutive_wins = max_consecutive_wins
    consecutive_losses = max_consecutive_losses

    return {
        'total_trades': len(positions),
        'win_rate_net': win_rate_net,  # Win rate с учетом комиссий
        'win_rate_gross': win_rate_gross,  # Win rate без комиссий
        'avg_return_per_trade_net': avg_return_net,
        'total_return_net': total_return_net,
        'total_return_gross': total_return_gross,
        'sharpe_ratio_net': sharpe_ratio_net,
        'max_drawdown_net': max_drawdown_net,
        'total_commission_paid': total_commission,
        'net_vs_gross_return': total_return_net - total_return_gross,
        'returns_net': returns_net,
        'returns_gross': returns_gross,
        'profit_factor': profit_factor,
        'total_absolute_pnl': total_absolute_pnl,
        'avg_absolute_pnl': avg_absolute_pnl,
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses,
        'equity_curve': equity_curve
    }


# -------------------------------
# Функция стратегии с улучшенной логикой
# -------------------------------
def apply_strategy(df, params=None, commission_rate=0.004):
    """
    Применение торговой стратегии

    Args:
        df (pd.DataFrame): Данные OHLC
        params (dict): Параметры стратегии
        commission_rate (float): Комиссия брокера (по умолчанию 0.1%)

    Returns:
        pd.DataFrame: Данные с сигналами
    """
    if params is None:
        params = adaptive_parameters(df)

    logger.info("📈 Начало применения торговой стратегии")
    logger.info(f"⚙️ Параметры стратегии: {params}")
    logger.info(f"💰 Комиссия брокера: {commission_rate:.3%}")

    try:
        # Создаем копию для избежания предупреждений SettingWithCopyWarning
        df = df.copy()
        logger.info(f"Загружено {len(df)} записей с {df['date'].min()} по {df['date'].max()}")

        # Вычисление индикаторов HMA
        df['hma_fast'] = hull_moving_average(df['close'], params['hma_fast'])
        df['hma_slow'] = hull_moving_average(df['close'], params['hma_slow'])

        # Пересечение HMA с учетом наклона
        df['hma_fast_slope'] = df['hma_fast'].diff()
        df['hma_slow_slope'] = df['hma_slow'].diff()
        df['hma_fast_prev'] = df['hma_fast'].shift(1)
        df['hma_slow_prev'] = df['hma_slow'].shift(1)

        df['hma_cross_up'] = (
                (df['hma_fast'] > df['hma_slow']) &
                (df['hma_fast_prev'] <= df['hma_slow_prev']) &
                (df['hma_fast_slope'] > 0) &
                (df['hma_slow_slope'] > 0)
        ).astype(int)
        df['hma_cross_down'] = (
                (df['hma_fast'] < df['hma_slow']) &
                (df['hma_fast_prev'] >= df['hma_slow_prev']) &
                (df['hma_fast_slope'] < 0) &
                (df['hma_slow_slope'] < 0)
        ).astype(int)

        # RSI с дивергенциями
        df['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_len'] if 'rsi_len' in params else 14)
        df['rsi_buy'] = ((df['rsi'] < params['rsi_oversold']) & (df['rsi'].shift(1) >= params['rsi_oversold'])).astype(
            int)
        df['rsi_sell'] = (
                (df['rsi'] > params['rsi_overbought']) & (df['rsi'].shift(1) <= params['rsi_overbought'])).astype(
            int)

        # Обнаружение дивергенций
        bullish_div, bearish_div = detect_rsi_divergence(df)
        df['bullish_div'] = bullish_div.astype(int)
        df['bearish_div'] = bearish_div.astype(int)

        # MACD (пересечение линий)
        df['macd'], df['macd_signal'], _ = talib.MACD(
            df['close'],
            fastperiod=params['macd_fast'],
            slowperiod=params['macd_slow'],
            signalperiod=params['macd_signal']
        )
        df['macd_cross_up'] = (
                (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = (
                (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

        # ATR
        df['atr'] = talib.ATR(
            df['high'],
            df['low'],
            df['close'],
            timeperiod=params['atr_len'] if 'atr_len' in params else 14
        )

        # Подтверждение объемом (если доступно)
        if 'volume' in df.columns:
            df['vol_ma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_confirmation'] = (df['volume'] > df['vol_ma']).astype(int)
        else:
            df['volume_confirmation'] = 1

        # Временные фильтры
        df = add_time_filters(df)
        df['time_filter'] = df['time_filter'].astype(int)

        # Сила сигналов (конфлюэнтность)
        df['signal_strength'] = (
                df['hma_cross_up'] * 1 +
                df['rsi_buy'] * 1 +
                df['macd_cross_up'] * 1 +
                df['bullish_div'] * 1
        )

        # Комбинированные сигналы с усилением
        df['signal_buy'] = (
                ((df['signal_strength'] >= 3) |  # 3+ подтверждения
                 (df['bullish_div'] & (df['rsi'] < 50))) &  # Или сильная дивергенция
                (df['volume_confirmation'] == 1) &
                (df['time_filter'] == 1)
        ).astype(int)

        df['signal_sell'] = (
                ((df['hma_cross_down'] == 1) & (df['rsi_sell'] == 1) & (df['macd_cross_down'] == 1)) |
                ((df['bearish_div'] == 1) & (df['rsi'] > 50)) &
                (df['volume_confirmation'] == 1) &
                (df['time_filter'] == 1)
        ).astype(int)

        # Принятие решений
        df['decision'] = 0
        df.loc[df['signal_buy'] == 1, 'decision'] = 1  # Покупка
        df.loc[df['signal_sell'] == 1, 'decision'] = -1  # Продажа

        # Заполняем NaN нулями
        df = df.fillna(0)

        # Добавляем колонки для анализа
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['atr'] / df['close'] if df['atr'].sum() > 0 else df['close'].pct_change().rolling(
            20).std()

        # Управление рисками
        df = add_risk_management(df, stop_loss_pct=0.03, take_profit_pct=0.06)

        logger.success("✅ Стратегия успешно применена")

        # Вывод статистики
        stats = get_strategy_stats(df)
        logger.info(f"📊 Статистика стратегии:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        # Метрики производительности с учетом комиссий
        metrics = calculate_performance_metrics_with_commission(df, commission_rate)
        if metrics:
            logger.info(f"📈 Метрики производительности (с учетом комиссий {commission_rate:.3%}):")
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'rate' in key or 'return' in key or 'drawdown' in key or 'factor' in key:
                        logger.info(f"   {key}: {value:.2%}")
                    else:
                        logger.info(f"   {key}: {value:.4f}")
                elif key not in ['returns_net', 'returns_gross', 'equity_curve']:
                    logger.info(f"   {key}: {value}")

            # Дополнительная информация о влиянии комиссий
            if 'total_return_gross' in metrics and 'total_return_net' in metrics:
                commission_impact = metrics['total_return_gross'] - metrics['total_return_net']
                logger.info(f"💸 Влияние комиссий на доходность: {commission_impact:.2%}")

        # Превью результата
        logger.info("📋 Результат стратегии (последние 10 строк):")
        preview_df = df.tail(10)
        logger.info(f"Превью данных: \n{tabulate(preview_df, headers='keys', tablefmt='psql', floatfmt='.4f', showindex=False)}")

        return df

    except Exception as e:
        logger.error(f"❌ Ошибка при применении стратегии: {str(e)}")
        raise


# -------------------------------
# Функция для оптимизации параметров с учетом комиссий
# -------------------------------
def optimize_parameters(df, commission_rate=0.0004, optimization_result_file="optimization_results/best_params.json"):
    """Оптимизация параметров для максимального net win rate"""

    logger.info("🔍 Начало оптимизации параметров...")
    logger.info(f"💰 Комиссия брокера при оптимизации: {commission_rate:.3%}")

    # Проверяем, есть ли уже сохраненные параметры
    if os.path.exists(optimization_result_file):
        logger.info(f"✅ Оптимизация уже выполнена. Параметры загружены из {optimization_result_file}")
        return

    # Получаем диапазоны параметров
    param_ranges = get_parameter_ranges()

    # Создаем все возможные комбинации параметров
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())

    best_win_rate_net = -1
    best_params = get_default_params()
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    logger.info(f"⚙️ Всего комбинаций для проверки: {total_combinations}")

    # Для ускорения можно ограничить количество комбинаций
    max_combinations = min(1000, total_combinations)  # Ограничиваем для скорости

    tested_combinations = 0

    # Тестирование различных комбинаций
    for combination in product(*param_values):
        if tested_combinations >= max_combinations:
            break

        # Создаем словарь параметров
        params = dict(zip(param_names, combination))

        try:
            # Применяем стратегию с текущими параметрами
            df_test = apply_strategy_optimization(df.copy(), params)

            # Рассчитываем метрики с учетом комиссий
            metrics = calculate_performance_metrics_with_commission(df_test, commission_rate)

            if metrics and 'win_rate_net' in metrics and metrics['total_trades'] > 0:
                win_rate_net = metrics['win_rate_net']
                total_trades = metrics['total_trades']

                # Учитываем количество сделок и чистый win rate для более надежной оценки
                if total_trades >= 3 and win_rate_net > best_win_rate_net:
                    best_win_rate_net = win_rate_net
                    best_params = params.copy()
                    logger.info(f"🏆 Новые лучшие параметры: Net Win Rate = {win_rate_net:.2%}, Trades = {total_trades}")
                    logger.info(f"   Параметры: {params}")

            tested_combinations += 1
            if tested_combinations % 100 == 0:
                logger.info(f"🔄 Протестировано {tested_combinations}/{max_combinations} комбинаций...")

        except Exception as e:
            logger.warning(f"⚠️ Ошибка при тестировании параметров {params}: {e}")
            continue

    # Сохраняем лучшие параметры
    best_params['win_rate_net'] = best_win_rate_net
    best_params['commission_rate_used'] = commission_rate
    with open(optimization_result_file, 'w') as f:
        json.dump(best_params, f, indent=4)

    logger.success(f"✅ Оптимизация завершена! Лучшие параметры сохранены в {optimization_result_file}")
    logger.success(f"🏆 Лучший Net Win Rate: {best_win_rate_net:.2%}")
    logger.success(f"📊 Лучшие параметры: {best_params}")

    return best_params


# -------------------------------
# Упрощенная версия apply_strategy для оптимизации
# -------------------------------
def apply_strategy_optimization(df, params):
    """Упрощенная версия apply_strategy для быстрой оптимизации"""

    # Вычисление индикаторов HMA
    df['hma_fast'] = hull_moving_average(df['close'], params['hma_fast'])
    df['hma_slow'] = hull_moving_average(df['close'], params['hma_slow'])

    # Пересечение HMA
    df['hma_fast_prev'] = df['hma_fast'].shift(1)
    df['hma_slow_prev'] = df['hma_slow'].shift(1)

    df['hma_cross_up'] = (
            (df['hma_fast'] > df['hma_slow']) &
            (df['hma_fast_prev'] <= df['hma_slow_prev'])
    ).astype(int)
    df['hma_cross_down'] = (
            (df['hma_fast'] < df['hma_slow']) &
            (df['hma_fast_prev'] >= df['hma_slow_prev'])
    ).astype(int)

    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_len'])
    df['rsi_buy'] = (df['rsi'] < params['rsi_oversold']).astype(int)
    df['rsi_sell'] = (df['rsi'] > params['rsi_overbought']).astype(int)

    # MACD
    df['macd'], df['macd_signal'], _ = talib.MACD(
        df['close'],
        fastperiod=params['macd_fast'],
        slowperiod=params['macd_slow'],
        signalperiod=params['macd_signal']
    )
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']).astype(int)
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']).astype(int)

    # Комбинированные сигналы
    df['signal_buy'] = (
            (df['hma_cross_up'] == 1) &
            (df['rsi_buy'] == 1) &
            (df['macd_cross_up'] == 1)
    ).astype(int)

    df['signal_sell'] = (
            (df['hma_cross_down'] == 1) &
            (df['rsi_sell'] == 1) &
            (df['macd_cross_down'] == 1)
    ).astype(int)

    # Принятие решений
    df['decision'] = 0
    df.loc[df['signal_buy'] == 1, 'decision'] = 1
    df.loc[df['signal_sell'] == 1, 'decision'] = -1

    # Заполняем NaN нулями
    df = df.fillna(0)

    return df


# -------------------------------
# Функция для загрузки реальных данных (пример)
# -------------------------------
def load_real_data(file_path):
    """Загрузка реальных данных из CSV"""
    try:
        df = pd.read_csv(file_path)
        # Предполагаем, что есть колонки: date, open, high, low, close
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        logger.info(f"📥 Загружено {len(df)} строк данных из {file_path}")
        return df
    except Exception as e:
        logger.warning(f"⚠️ Ошибка загрузки файла {file_path}: {str(e)}")
        return None


# -------------------------------
# Основной процесс
# -------------------------------
if __name__ == "__main__":
    try:
        logger.info("🚀 Запуск торговой стратегии")

        # Попытка загрузить реальные данные
        data_file = "data.csv"  # Замените на путь к вашим данным
        if os.path.exists(data_file):
            df = load_real_data(data_file)
            if df is None:
                raise Exception("Не удалось загрузить данные")
        else:
            logger.error(f"💥 Критическая ошибка: отсутствует файл {data_file}")
            raise Exception(f"Отсутствует файл {data_file}")

        # Комиссия брокера (по умолчанию 0.04%)
        BROKER_COMMISSION = 0.0004  # 0.04%

        # Оптимизация параметров (если файл с результатами еще не существует)
        optimization_file = "optimization_results/best_params.json"
        if not os.path.exists(optimization_file):
            logger.info("🔍 Запуск оптимизации параметров...")
            optimize_parameters(df, BROKER_COMMISSION, optimization_file)
        else:
            logger.info("✅ Оптимизация уже выполнена, используем сохраненные параметры")

        # Применение стратегии с оптимизированными параметрами и учетом комиссий
        logger.info("⚙️ Применение стратегии с оптимизированными параметрами и учетом комиссий...")
        df_result = apply_strategy(df, commission_rate=BROKER_COMMISSION)

        # Финальная статистика
        decision_counts = df_result['decision'].value_counts().to_dict()
        logger.info(f"📉 Итоговое распределение решений: {decision_counts}")

    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {str(e)}")
        raise