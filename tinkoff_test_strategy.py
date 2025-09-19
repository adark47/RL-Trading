# tinkoff_test_strategy.py
"""
Example script to run a live trading strategy using the Tinkoff adapter (t_adapter).
"""
import logging
import os
import sys
from decimal import Decimal

# --- Добавляем родительскую директорию в путь поиска модулей ---
# Это позволяет импортировать strategy.py и config.py из той же директории,
# что и этот скрипт, если они там находятся.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Импорты Nautilus Trader (исправленные пути для 1.219.0) ---
# Проверка существующих модулей Nautilus Trader
try:
    import nautilus_trader
    print("SUCCESS: nautilus_trader imported")
except ImportError as e:
    print(f"nautilus_trader: {e}")
    sys.exit(1)

try:
    import nautilus_trader.model
    print("SUCCESS: nautilus_trader.model imported")
    # Проверяем содержимое nautilus_trader.model
    model_contents = [item for item in dir(nautilus_trader.model) if not item.startswith('_')]
    print(f"nautilus_trader.model contents (filtered): {model_contents[:10]}...") # Показываем первые 10
except ImportError as e:
    print(f"nautilus_trader.model: {e}")
    sys.exit(1)

try:
    import nautilus_trader.core
    print("SUCCESS: nautilus_trader.core imported")
    # Проверяем содержимое nautilus_trader.core
    core_contents = [item for item in dir(nautilus_trader.core) if not item.startswith('_')]
    print(f"nautilus_trader.core contents (filtered): {core_contents[:10]}...") # Показываем первые 10
except ImportError as e:
    print(f"nautilus_trader.core: {e}")
    sys.exit(1)

# --- Попытка импортировать нужные классы из разных возможных мест ---
# Bar, BarType
try:
    from nautilus_trader.model import Bar, BarType
    print("SUCCESS: Found Bar, BarType in nautilus_trader.model")
except ImportError as e:
    print(f"nautilus_trader.model (Bar, BarType): {e}")
    try:
        from nautilus_trader.core.data import Bar, BarType # Альтернативный путь
        print("SUCCESS: Found Bar, BarType in nautilus_trader.core.data")
    except ImportError as e_core_data:
        print(f"nautilus_trader.core.data (Bar, BarType): {e_core_data}")
        raise RuntimeError("Could not import Bar, BarType from any known location in Nautilus Trader 1.219.0")

# BarSpecification, BarAggregation, PriceType
try:
    from nautilus_trader.model.bar import BarSpecification, BarAggregation, PriceType
    print("SUCCESS: Found BarSpecification, BarAggregation, PriceType in nautilus_trader.model.bar")
except ImportError as e:
    print(f"nautilus_trader.model.bar (BarSpec): {e}")
    try:
        from nautilus_trader.core.bar import BarSpecification, BarAggregation, PriceType # Альтернативный путь
        print("SUCCESS: Found BarSpecification, BarAggregation, PriceType in nautilus_trader.core.bar")
    except ImportError as e_core_bar:
        print(f"nautilus_trader.core.bar (BarSpec): {e_core_bar}")
        try:
            from nautilus_trader.model.enums import BarSpecification, BarAggregation, PriceType # Еще один альтернативный путь
            print("SUCCESS: Found BarSpecification, BarAggregation, PriceType in nautilus_trader.model.enums")
        except ImportError as e_model_enums:
            print(f"nautilus_trader.model.enums (BarSpec): {e_model_enums}")
            raise RuntimeError("Could not import BarSpecification, BarAggregation, PriceType from any known location in Nautilus Trader 1.219.0")

# InstrumentId
try:
    from nautilus_trader.model.identifiers import InstrumentId
    print("SUCCESS: Found InstrumentId in nautilus_trader.model.identifiers")
except ImportError as e:
    print(f"nautilus_trader.model.identifiers (InstrumentId): {e}")
    try:
        from nautilus_trader.model import InstrumentId # Альтернативный путь
        print("SUCCESS: Found InstrumentId in nautilus_trader.model")
    except ImportError as e_model:
        print(f"nautilus_trader.model (InstrumentId): {e_model}")
        raise RuntimeError("Could not import InstrumentId from any known location in Nautilus Trader 1.219.0")

# TraderId
try:
    from nautilus_trader.model import TraderId
    print("SUCCESS: Found TraderId in nautilus_trader.model")
except ImportError as e:
    print(f"nautilus_trader.model (TraderId): {e}")
    try:
        from nautilus_trader.model.identifiers import TraderId # Альтернативный путь
        print("SUCCESS: Found TraderId in nautilus_trader.model.identifiers")
    except ImportError as e_identifiers:
        print(f"nautilus_trader.model.identifiers (TraderId): {e_identifiers}")
        raise RuntimeError("Could not import TraderId from any known location in Nautilus Trader 1.219.0")

# MessageBus
try:
    from nautilus_trader.msgbus.bus import MessageBus
    print("SUCCESS: Found MessageBus in nautilus_trader.msgbus.bus")
except ImportError as e:
    print(f"nautilus_trader.msgbus.bus (MessageBus): {e}")
    try:
        from nautilus_trader.msgbus import MessageBus # Альтернативный путь
        print("SUCCESS: Found MessageBus in nautilus_trader.msgbus")
    except ImportError as e_msgbus:
        print(f"nautilus_trader.msgbus (MessageBus): {e_msgbus}")
        # MessageBus часто передается как объект, аннотация Any может быть достаточной
        # Для аннотаций типов используем Any
        from typing import Any
        MessageBus = Any # Заглушка для аннотации
        print("INFO: Using Any for MessageBus annotation.")

# Остальные импорты Nautilus Trader
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveExecEngineConfig,
    TradingNodeConfig,
    LoggingConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock

# --- Импорты для вашей стратегии ---
# Убедитесь, что файлы strategy.py и config.py находятся в одной директории с этим скриптом
# или путь к ним добавлен в sys.path.
try:
    from strategy import Strategy, StrategyConfig
    from config import StrategyConfig as StrategyConfig_ # Переименовываем во избежание конфликта
except ImportError as e:
    print(f"Ошибка импорта стратегии или конфигурации: {e}")
    sys.exit(1)

# --- Импорты из вашего нового адаптера t_adapter ---
# Убедитесь, что директория t_adapter находится в той же директории,
# что и этот скрипт, или в PYTHONPATH.
try:
    from t_adapter.config import TinkoffConfiguration
    HAS_T_ADAPTER_CONFIG = True
    print("SUCCESS: Imported TinkoffConfiguration from t_adapter.config")
except ImportError as e:
    print(f"Ошибка импорта t_adapter.config: {e}")
    HAS_T_ADAPTER_CONFIG = False
    TinkoffConfiguration = None

# Попытка импортировать фабрики (которые могут вызвать ImportError)
try:
    from t_adapter.factories import TinkoffLiveDataFactory, TinkoffLiveExecutionFactory
    HAS_T_ADAPTER_FACTORIES = True
    print("SUCCESS: Imported TinkoffLiveDataFactory and TinkoffLiveExecutionFactory from t_adapter.factories")
except ImportError as e:
    print(f"Warning: Could not import TinkoffLiveDataFactory/TinkoffLiveExecutionFactory from t_adapter.factories: {e}")
    HAS_T_ADAPTER_FACTORIES = False
    TinkoffLiveDataFactory = None
    TinkoffLiveExecutionFactory = None

# --- Создаем экземпляр конфигурации вашей стратегии ---
# Предполагается, что config.StrategyConfig_() загружает настройки
# (например, из переменных окружения, файла .env или напрямую).
settings = StrategyConfig_()

# --- УСТАНОВКА ТОКЕНА API TINKOFF ---
# ЖЕСТКАЯ ЗАДАЧА ТОКЕНА В КОДЕ (НЕ РЕКОМЕНДУЕТСЯ ДЛЯ PRODUCTION)
# Лучше установить переменную окружения в системе или через конфигурацию запуска.
# os.environ["TINKOFF_API_TOKEN"] = "ваш_токен_здесь" # <-- Раскомментируйте и замените, если нужно

# --- ИСПРАВЛЕНО: Установка токена как переменной в коде ---
# НЕ ЗАБУДЬТЕ ЗАМЕНИТЬ НА СВОЙ РЕАЛЬНЫЙ ТОКЕН!
YOUR_TINKOFF_API_TOKEN_HERE = "t.mQ1iyp-e90gIQLOTjbfALe2aBdVGmwVCtGFaizObj2G3HPuhljWpHfGls79YYVlsJCXcFR0w1FTKpGuSxCaN1A"

# Устанавливаем токен в переменную окружения, чтобы t_adapter/common.py мог его найти
os.environ["TINKOFF_API_TOKEN"] = YOUR_TINKOFF_API_TOKEN_HERE
# --- КОНЕЦ УСТАНОВКИ ТОКЕНА ---

def _validate_credentials() -> None:
    """
    Ensure Tinkoff API credentials are available before starting the node.
    Проверяем переменную окружения, как это делает сам t_adapter/common.py
    """
    # Проверяем переменную окружения по её имени
    token = os.environ.get("TINKOFF_API_TOKEN") # <-- ИСПРАВЛЕНО: Проверяем правильное имя переменной
    if not token:
        # Альтернативно, можно проверить settings, если токен там хранится
        # if not getattr(settings, 'tinkoff_api_token', None): # Безопасный доступ
        raise ValueError(
            "Tinkoff API credentials are not set. "
            "Please set the TINKOFF_API_TOKEN environment variable or hardcode it in run_Live.py."
            # "Please set the tinkoff_api_token in settings."
        )
    # Дополнительно: можно проверить формат или базовую валидность токена, если необходимо
    # Например, минимальная длина или наличие точки
    # if len(token) < 20 or '.' not in token:
    #     raise ValueError("TINKOFF_API_TOKEN appears to be invalid.")


def build_node() -> TradingNode:
    """
    Create, configure and build a TradingNode instance ready to run.
    """
    _validate_credentials()

    if not HAS_T_ADAPTER_CONFIG:
        raise RuntimeError("Tinkoff adapter configuration (TinkoffConfiguration) is not available.")

    # --- Конфигурация для Tinkoff ---
    # Определяем, использовать ли песочницу (например, через переменную окружения или settings)
    is_sandbox_env = os.environ.get("TINKOFF_USE_SANDBOX", "false").lower() in ("true", "1", "yes")
    # is_sandbox = settings.use_sandbox # Если используете settings
    is_sandbox = is_sandbox_env

    # --- Strategy configuration ---
    # ВАЖНО: InstrumentId должен соответствовать маппингу в вашем t_adapter
    # Пример: Предположим, вы хотите торговать Сбером.
    # Вам нужно определить instrument_id_str так, чтобы он соответствовал
    # ключу в словаре INSTRUMENT_ID_TO_UID в t_adapter/common.py.
    # Например, если там есть запись INSTRUMENT_ID_TO_UID[InstrumentId("SBER/MOEX")] = "e6123145-..."
    # instrument_id_str = "SBER/MOEX"

    # ВАЖНО: Вы должны определить settings.instrument_id где-то в вашем config.py
    # или задать его напрямую здесь.
    # instrument_id_str = settings.instrument_id # Предполагается, что это определено в settings
    # ПОКА ЧТО ИСПОЛЬЗУЕМ ЗАГЛУШКУ:
    instrument_id_str = getattr(settings, 'instrument_id', "SBER.MOEX") # <--- ЗАМЕНИТЕ НА КОРРЕКТНОЕ ЗНАЧЕНИЕ ИЗ ВАШЕГО КОНФИГА
    try:
        instrument_id = InstrumentId.from_str(instrument_id_str)
    except Exception as e:
        print(f"Ошибка создания InstrumentId из '{instrument_id_str}': {e}")
        raise

    # --- BarType ---
    # Убедитесь, что формат BarType совместим с тем, что будет генерировать t_adapter/data.py
    # Например, если t_adapter будет публиковать 1-минутные бары:
    bar_type_str = f"{instrument_id_str}-1-MINUTE-LAST-EXTERNAL" # Или INTERNAL, в зависимости от источника
    try:
        primary_bar_type = BarType.from_str(bar_type_str)
    except Exception as e:
        print(f"Ошибка создания BarType из '{bar_type_str}': {e}")
        raise

    strat_config = StrategyConfig(
        instrument_id=instrument_id,
        primary_bar_type=primary_bar_type,
        # Убедитесь, что settings.trade_size определен
        trade_size=Decimal(str(getattr(settings, 'trade_size', 1))), # taken from settings, default 1
        trade_mode="LIVE" # "BACKTEST" или "LIVE"
    )

    # --- Instrument Provider Config ---
    instrument_provider_cfg = InstrumentProviderConfig(load_all=True) # Или настройте фильтры

    # --- Tinkoff Configuration ---
    # Создаем конфигурацию для t_adapter
    # Предполагается, что settings содержит необходимые атрибуты для TinkoffConfiguration
    # (например, tinkoff_api_token, is_sandbox и т.д.)
    # Токен теперь берется из os.environ["TINKOFF_API_TOKEN"], установленного выше
    tinkoff_config = TinkoffConfiguration(
        # api_token=settings.tinkoff_api_token, # Если токен в settings
        api_token=None, # Позволим t_adapter использовать TINKOFF_API_TOKEN из env
        is_sandbox=is_sandbox,
        instrument_provider_config=instrument_provider_cfg,
        # data_client_config и exec_client_config можно настроить при необходимости
    )

    # --- Node configuration ---
    # Убедитесь, что все атрибуты settings (trader_id_live, timeouts) определены
    node_config = TradingNodeConfig(
        trader_id=TraderId(str(getattr(settings, 'trader_id_live', 'TRADER-001'))), # default
        logging=LoggingConfig(log_level="INFO"), # use_pyo3 может быть не применим
        exec_engine=LiveExecEngineConfig(
            reconciliation=False, # Настройте по необходимости
            reconciliation_lookback_mins=1440,
            # filter_position_reports=True, # Проверьте, нужны ли специфичные настройки
            # filter_order_status_reports=True,
            # validate_order_responses=True,
        ),
        # --- ИСПРАВЛЕНО: Передаем {} вместо {"TINKOFF": None} или {"TINKOFF": tinkoff_config} ---
        # Это предотвращает ошибки ImportError и AttributeError, связанные с фабриками.
        # Если t_adapter.factories успешно импортированы, они будут добавлены позже.
        data_clients={}, # <-- ИСПРАВЛЕНО: Пустой словарь
        exec_clients={}, # <-- ИСПРАВЛЕНО: Пустой словарь
        # --- Конец исправления ---
        timeout_connection=getattr(settings, 'timeout_connection', 10.0), # default 10.0
        timeout_reconciliation=getattr(settings, 'timeout_reconciliation', 10.0), # default 10.0
        timeout_portfolio=getattr(settings, 'timeout_portfolio', 10.0), # default 10.0
        timeout_disconnection=getattr(settings, 'timeout_disconnection', 10.0), # default 10.0
        timeout_post_stop=getattr(settings, 'timeout_post_stop', 10.0), # default 10.0
    )

    # --- Build and return the node ---
    node = TradingNode(config=node_config)

    # Добавляем стратегию
    node.trader.add_strategy(Strategy(config=strat_config))

    # --- ИСПРАВЛЕНО: Добавляем фабрики клиентов из t_adapter, если они доступны ---
    # Перед запуском node.build(), добавляем фабрики клиентов, если они были успешно импортированы.
    # Предполагается, что фабрики были модифицированы для получения конфигурации.
    if HAS_T_ADAPTER_FACTORIES:
        print("Registering Tinkoff adapter factories...")
        # --- ИСПРАВЛЕНО: Передаем tinkoff_config в фабрики через их атрибуты ---
        # Перед запуском node.build(), устанавливаем конфигурацию в атрибуты фабрик.
        # Предполагается, что фабрики были модифицированы для получения конфигурации таким образом.
        TinkoffLiveDataFactory.tinkoff_config = tinkoff_config
        TinkoffLiveExecutionFactory.tinkoff_config = tinkoff_config

        # Добавляем фабрики клиентов из t_adapter
        node.add_data_client_factory("TINKOFF", TinkoffLiveDataFactory)
        node.add_exec_client_factory("TINKOFF", TinkoffLiveExecutionFactory)
        # --- Конец исправления ---
        print("Tinkoff adapter factories registered.")
    else:
        print("Warning: Tinkoff adapter factories are not available. Running without TINKOFF data/exec clients.")
    # --- Конец исправления ---

    # Строим узел
    node.build()

    return node


def main() -> None:
    """
    Entry point for running the trading node.
    """
    try:
        node = build_node()
    except Exception as e:
        print(f"Ошибка при создании узла: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("Запуск торгового узла...")
        node.run()
    except KeyboardInterrupt:
        print("\nПолучен сигнал KeyboardInterrupt, остановка...")
    except Exception as e:
        print(f"Произошла ошибка во время выполнения узла: {e}")
        import traceback
        traceback.print_exc() # Печатает полный стек вызовов для отладки
        # Вы можете добавить логирование здесь
    finally:
        print("Остановка и освобождение ресурсов узла...")
        try:
            node.stop()
            node.dispose()
            print("Узел остановлен и ресурсы освобождены.")
        except Exception as e:
            print(f"Ошибка при остановке/освобождении узла: {e}")


if __name__ == "__main__":
    main()
