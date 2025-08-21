# Reinforcement Learning Dataset for Volatility-Driven Futures Trading

This dataset collection contains four curated subsets of minute-level Binance Futures market data designed specifically for training, validating, testing, and backtesting reinforcement learning agents in high-volatility environments. The data is used in a production-grade project implementing a Dueling Double Deep Q-Network (D3QN) with Prioritized Experience Replay (PER), tailored for financial markets.

---

## 💡 Project Context

The dataset supports a full RL pipeline aimed at developing an intelligent trading system capable of:
- Making profitable trading decisions in highly volatile conditions;
- Learning from localized market impulses rather than continuous streams;
- Operating in realistic conditions, including slippage and transaction fees.

🔗 Full codebase: [GitHub Repository](https://github.com/YuriyKolesnikov/rl-trading-binance)  
📄 Research article (English): [RL Agent for Algorithmic Trading on Binance Futures — Architecture, Backtest, and Results](https://medium.com/@YuriKolesnikovAI/rl-agent-for-algorithmic-trading-on-binance-futures-architecture-backtest-and-results-63fc4662893d)  
📄 Research article (Russian): [RL-агент для алгоритмической торговли на Binance Futures: архитектура, бэктест, результаты](https://habr.com/ru/articles/934258/)  
🤖 Real-time RL predictions: [Telegram](https://t.me/binance_ai_agent)  

---

## 📁 Dataset Structure

Each data sample is a 150-minute window centered around a strong volatility impulse.

- Shape: `(150, 7)` — 150 minutes × 7 features
- Channels: `open`, `high`, `low`, `close`, `volume`, `volume_weighted_average`, `num_trades`
- Format: `np.ndarray` wrapped in `.npz`
- Metadata: unique keys `(TICKER, datetime)` per session

| Subset      | Samples | Period                    | Purpose                 |
|-------------|---------|---------------------------|-------------------------|
| `Train`     | 24,104  | 2020-01-14 → 2024-08-31   | Training                |
| `Validation`| 1,377 | 2024-09-01 → 2024-12-01   | Hyperparameter tuning   |
| `Test`      | 3,400   | 2024-12-01 → 2025-03-01   | Final evaluation        |
| `Backtest`  | 3,186   | 2025-03-01 → 2025-06-01   | Realistic simulation    |

Each session contains:
- 90 minutes of pre-impulse history (for state construction)
- 60 minutes of post-impulse trading session (for reward computation)

---

## 🧠 Dataset Motivation

This dataset departs from standard continuous sampling. Instead, it focuses only on high-volatility episodes that represent actual market decision points. Specifically:
- Price moves >5% within a 10-minute window
- Preceded by 90 minutes of relative stability
- Selected using a contrast ratio filter to remove noisy signals

These sessions serve as atomic training units for reinforcement learning agents operating in short-term trading strategies.

---

## 🧰 Data Pipeline Tools

All preprocessing logic is encapsulated in reusable utilities (as part of the open project):

- `load_npz_dataset(path)` — loads session list and metadata
- `select_and_arrange_channels(data, channels)` — filters and arranges input
- `calculate_normalization_stats(data)` — computes per-channel stats
- `apply_normalization(data, stats)` — standardizes for agent consumption

📂 Utilities

All preprocessing tools are included in [`data_utils.py`](./data_utils.py).

Data undergoes:
- Channel filtering
- Relative scaling
- Log transforms
- NaN/outlier protection

---

## 📊 Visualizations

Each sample comes with an optional visualization (example graphs available in the project):
- Line plots with volatility impulse marker at minute 90
- Session metadata in title (ticker + UTC timestamp)
- Used to audit signal quality and alignment

---

## 🔐 License

**License**: [MIT License](https://opensource.org/licenses/MIT)  
This dataset is released under the MIT license — you are free to use, modify, and distribute it for any purpose, including commercial use, provided that the original copyright and permission notice are included.

---

## 📦 Download and Usage

You can load the dataset using HuggingFace’s `datasets` library:

```python
from datasets import load_dataset

# Example: load training split
train_dataset = load_dataset("ResearchRL/open-rl-trading-binance-dataset", split="train_data")
