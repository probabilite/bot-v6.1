import asyncio
from datetime import datetime, timedelta
import json
import os
import sys
import talib
import numpy as np
from dotenv import load_dotenv
from binance import AsyncClient
import aiohttp
import logging
import random
from typing import List, Dict, Optional, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from notifier import send_telegram_message
from trade_params import fallback_trade_parameters 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import traceback
import filelock
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.preprocessing import LabelEncoder
from logging.handlers import RotatingFileHandler

def configure_logging():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    file_handler = RotatingFileHandler(
        'bot_debug.log',
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[file_handler, stream_handler]
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return logger

logger = configure_logging()
logger.info("✅ Modeller yüklendi! [DEBUG] Sistem aktif")

REQUIRED_FEATURES = [
    "signal_strength",
    "rsi",
    "ema_diff",
    "macd_direction",
    "bb_position",
    "volume_ratio",
    "atr_percent",
]

DEFAULTS = {
    'ema_diff': 0,
    'bb_position': 50,
    'volume_ratio': 1.0,
    'atr_percent': 1.5
}

def convert_to_builtin_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def prepare_features(trade_data):
    features = {}
    for feature in REQUIRED_FEATURES:
        features[feature] = trade_data.get(feature, DEFAULTS.get(feature, 0))
    
    features['side_encoded'] = 1 if str(trade_data.get('side', '')).lower() == 'long' else 0
    
    if 'volume_ratio' not in trade_data:
        high = trade_data.get('high', 1)
        low = trade_data.get('low', 0.0001)
        features['volume_ratio'] = trade_data.get('volume', 0) / (high - low)
    
    return [features[col] for col in REQUIRED_FEATURES]

def validate_trade_data(trade):
    required_keys = ['symbol', 'side', 'entry_price', 'rsi']
    missing = [k for k in required_keys if k not in trade]
    if missing:
        logger.warning(f"Eksik veri: {trade.get('symbol')} | Eksikler: {missing}")
        return False
    return True

class RiskCalculator:
    @staticmethod
    def calculate_position_size(symbol: str, atr: float, current_price: float, account_balance: float) -> float:
        if None in [atr, current_price, account_balance]:
            logger.warning("Eksik parametreler nedeniyle varsayılan boyut kullanılıyor")
            return TRAINING_POSITION_SIZE
            
        risk_amount = account_balance * (MAX_ACCOUNT_RISK_PERCENT / 100)
        position_size = risk_amount / (atr * 2)
        
        min_size = max(5.0, account_balance * 0.001)
        max_size = account_balance * 0.1
        
        return round(np.clip(position_size, min_size, max_size), 2)

    @staticmethod
    def dynamic_tp_sl(rsi: float, atr: float, probability: float, side: str, current_price: float) -> Tuple[float, float, float]:
        base_multiplier = 1 + (probability * 0.5)
        tp1 = round(atr * 1.5 * base_multiplier / current_price, 4)
        tp2 = round(atr * 3.0 * base_multiplier / current_price, 4)
        sl = round(atr * 1.0 * base_multiplier / current_price, 4)
        return (tp1, tp2, sl) if side == "long" else (-tp1, -tp2, -sl)

class SymbolRanker:
    @staticmethod
    def rank_symbols(symbols: List[str], historical_data: Dict) -> List[Tuple[str, float]]:
        ranked = []
        for sym in symbols:
            try:
                vol = historical_data[sym]['volume_24h']
                atr = historical_data[sym]['atr']
                score = (atr * 0.8) + (np.log(vol+1) * 0.2)
                ranked.append((sym, score))
            except Exception as e:
                logger.error(f"Ranker error for {sym}: {e}")
        return sorted(ranked, key=lambda x: x[1], reverse=True)

class ModelMonitor:
    def __init__(self):
        self.performance_log = []

    async def log_performance(self, y_true: np.array, y_pred: np.array):
        acc = accuracy_score(y_true, y_pred)
        self.performance_log.append({
            'timestamp': pd.Timestamp.now(),
            'accuracy': acc,
            'model_version': joblib.hash(self.model)
        })
        
        if len(self.performance_log) > 10:
            decay = self.check_decay()
            if decay:
                await self.retrain_model()

    def check_decay(self) -> bool:
        last_5_entries = self.performance_log[-5:]
        last_5 = np.mean([x['accuracy'] for x in last_5_entries]) if last_5_entries else 0
    
        first_5_entries = self.performance_log[:5]
        first_5 = np.mean([x['accuracy'] for x in first_5_entries]) if first_5_entries else 0
    
        decay_threshold = float(os.getenv("MODEL_DECAY_THRESHOLD", 0.15))
        return (first_5 - last_5) > decay_threshold

    async def check_model_performance(self):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        df = pd.DataFrame(history)
        if len(df) > 50:
            X = df[REQUIRED_FEATURES]
            y = df['profit_usdt'] > 0
            current_acc = model_cls.score(X, y)
            if current_acc < ML_THRESHOLD - 0.1:
                await self.retrain_model()

    async def retrain_model(self):
        try:
            if not os.path.exists(HISTORY_FILE):
                logger.error("History dosyası bulunamadı!")
                return False
            
            if len(history) < 100:
                logger.error(f"Yetersiz veri: {len(history)} kayıt")
                return False

            df = pd.DataFrame(history)
            df = df[df['profit_usdt'].notna()]

            X = df[REQUIRED_FEATURES]
            y = (df['profit_usdt'] > 0).astype(int)

            new_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
            new_model.fit(X, y)

            train_acc = new_model.score(X, y)
            logger.info(f"Retrain başarılı! Doğruluk: {train_acc:.2%}")

            global model_cls
            model_cls = new_model
            joblib.dump(new_model, MODEL_CLASSIFICATION_PATH)

            await send_telegram_message(
                f"🔄 Model yeniden eğitildi!\n"
                f"• Doğruluk: {train_acc:.2%}\n"
                f"• Kullanılan Veri: {len(df)} işlem"
            )

            return True

        except Exception as e:
            error_msg = f"Retrain hatası: {str(e)}"
            logger.error(error_msg)
            await send_telegram_message(f"🔴 {error_msg}")
            return False

async def init_binance_client():
    return await AsyncClient.create(
        api_key=os.getenv('BINANCE_FUTURES_API_KEY'),
        api_secret=os.getenv('BINANCE_FUTURES_SECRET_KEY'),
        testnet=bool(os.getenv('BINANCE_TESTNET'))
    )

POSITION_FILE = os.path.join(BASE_DIR, "positions.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history_reinforced.json")
SYMBOL_CACHE_FILE = os.path.join(BASE_DIR, "symbol_cache.json")
BLACKLIST_FILE = os.path.join(BASE_DIR, "blacklist.json")
MODEL_CLASSIFICATION_PATH = os.path.join(BASE_DIR, "data", "models", "model_cls.pkl")
MODEL_REGRESSION_PATH = os.path.join(BASE_DIR, "data", "models", "model_reg.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data", "models", "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "data", "models", "label_encoder.pkl")

ML_THRESHOLD = float(os.getenv("ML_THRESHOLD", 0.55))
TRAINING_POSITION_SIZE = float(os.getenv("TRAINING_POSITION_SIZE", 3.0))
TRAINING_MIN_PROB = float(os.getenv("TRAINING_MIN_PROB", 0.25))
TRAINING_MAX_PROB = float(os.getenv("TRAINING_MAX_PROB", 0.4))

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MIN_SIGNAL_STRENGTH = int(os.getenv("MIN_SIGNAL_STRENGTH", 1))
MAX_SHORT_POSITIONS = int(os.getenv("MAX_SHORT_POSITIONS", 15))
MIN_PRICE = float(os.getenv("MIN_PRICE", 0.02))
MAX_HISTORY_LOAD = 1_000_000
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 30))
INTERVAL = int(os.getenv("INTERVAL", 300))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 30))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", 10))
PNL_REPORT_INTERVAL = int(os.getenv("PNL_REPORT_INTERVAL", 1800))
SYMBOL_CACHE_TTL = int(os.getenv("SYMBOL_CACHE_TTL_MINUTES", 60))
LIQUIDITY_THRESHOLD = float(os.getenv("LIQUIDITY_THRESHOLD", 100))
MAX_ACCOUNT_RISK_PERCENT = float(os.getenv("MAX_ACCOUNT_RISK_PERCENT", 1.5))

cooldown_tracker = {}
last_positions_time = {}
SYMBOL_CACHE_FILE = "symbol_cache.json"
model_cls = None
model_reg = None
scaler = None
model_monitor = ModelMonitor()
bot_start_time = datetime.now()

def predict_with_ml(symbol_data):
    try:
        features = prepare_features(symbol_data)
        features_scaled = scaler.transform([features])
        
        proba = model_cls.predict_proba(features_scaled)[0][1]
        predicted_pnl = model_reg.predict(features_scaled)[0]
        
        atr_percent = symbol_data.get('atr_percent', 1.5)
        if symbol_data['side_encoded'] == 1:
            tp1 = 0.015 * atr_percent
            tp2 = 0.03 * atr_percent
            sl = 0.01 * atr_percent
        else:
            tp1 = -0.015 * atr_percent
            tp2 = -0.03 * atr_percent
            sl = -0.01 * atr_percent
            
        return {
            'probability': float(proba),
            'predicted_pnl': float(predicted_pnl),
            'signal': 1 if proba >= ML_THRESHOLD else 0,
            'tp_sl': (tp1, tp2, sl)
        }
    except Exception as e:
        logger.error(f"ML tahmin hatası: {e}")
        return None

def calculate_position_size(probability, atr_percent, balance):
    try:
        if probability < TRAINING_MIN_PROB:
            return 0
        
        risk_factor = np.clip((probability - TRAINING_MIN_PROB) / (TRAINING_MAX_PROB - TRAINING_MIN_PROB), 0, 1)
        risk_amount = balance * 0.01 * risk_factor
        
        position_size = risk_amount / (atr_percent * 0.01)
        
        if os.getenv("TRAINING_MODE", "false").lower() == "true":
            return min(TRAINING_POSITION_SIZE, position_size)
            
        return round(position_size, 2)
    except Exception as e:
        logger.error(f"Pozisyon boyutu hesaplama hatası: {str(e)}")
        return TRAINING_POSITION_SIZE

def generate_signal(symbol_data):
    ml_result = predict_with_ml(symbol_data)
    if not ml_result:
        return None
    
    position_size = calculate_position_size(
        probability=ml_result['probability'],
        atr_percent=symbol_data['atr_percent'],
        balance=get_current_balance()
    )
    
    return {
        'symbol': symbol_data['symbol'],
        'side': 'long' if ml_result['signal'] == 1 else 'short',
        'entry_price': symbol_data['close'],
        'probability': ml_result['probability'],
        'predicted_pnl': float(predicted_pnl),
        'position_size': position_size,
        'timestamp': datetime.now().isoformat()
    }

def load_models():
    global model_cls, model_reg, scaler, label_encoder
    
    try:
        # Load models with feature names
        model_cls = joblib.load(MODEL_CLASSIFICATION_PATH)
        model_reg = joblib.load(MODEL_REGRESSION_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # Set feature names if not present
        if not hasattr(model_cls, 'feature_names_in_'):
            model_cls.feature_names_in_ = np.array(REQUIRED_FEATURES)
        if not hasattr(model_reg, 'feature_names_in_'):
            model_reg.feature_names_in_ = np.array(REQUIRED_FEATURES)
        if not hasattr(scaler, 'feature_names_in_'):
            scaler.feature_names_in_ = np.array(REQUIRED_FEATURES)
            
        # Load label encoder
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        logger.info("Models loaded successfully with feature names")
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise RuntimeError("Modeller yüklenemedi! Lütfen train_models.py'yi çalıştırın.")

def get_current_balance():
    return 10000

def initialize_files():
    """Initialize or clean corrupted files"""
    for file in [POSITION_FILE, HISTORY_FILE, SYMBOL_CACHE_FILE, BLACKLIST_FILE]:
        try:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    json.dump([] if file != BLACKLIST_FILE else [], f)
            else:
                # Verify file is valid JSON
                with open(file, 'r') as f:
                    json.load(f)
        except (json.JSONDecodeError, IOError):
            with open(file, 'w') as f:
                json.dump([] if file != BLACKLIST_FILE else [], f)
            logger.warning(f"Reset corrupted file: {file}")
                
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
        logger.info("Yeni history dosyası oluşturuldu. Shadow mod başlatılıyor...")

def add_to_blacklist(symbol):
    try:
        if not os.path.exists(BLACKLIST_FILE):
            blacklist = []
        else:
            with open(BLACKLIST_FILE, "r") as f:
                blacklist = json.load(f)
        if symbol not in blacklist:
            blacklist.append(symbol)
            with open(BLACKLIST_FILE, "w") as f:
                json.dump(blacklist, f, indent=4)
            logger.info(f"{symbol} blacklist'e eklendi.")
        else:
            logger.info(f"{symbol} zaten blacklist'te.")
    except Exception as e:
        logger.error(f"Blacklist ekleme hatası: {e}")

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

async def send_telegram_message(text: str):
    with open('telegram_sent.log', 'a', encoding='utf-8') as f:
        f.write(text + "\n")
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram error: {await resp.text()}")
    except Exception as e:
        logger.error(f"Telegram send error: {str(e)}")

def train_models():
    try:
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
        
        if len(data) < 100:
            raise ValueError("Yetersiz veri (min 100 işlem)")
        
        df = pd.DataFrame(data)
        df = df[df['profit_usdt'].notna()]
        
        df['side_encoded'] = df['side'].apply(lambda x: 1 if x == 'long' else 0)
        features = ['rsi', 'ema_diff', 'volume_ratio', 'atr_percent']
        X = df[features]
        
        y_cls = (df['profit_usdt'] > 0).astype(int)
        y_reg = df['pnl_percent']
        
        clf = RandomForestClassifier(n_estimators=50).fit(X, y_cls)
        reg = RandomForestRegressor(n_estimators=50).fit(X, y_reg)
        
        joblib.dump(clf, MODEL_CLASSIFICATION_PATH)
        joblib.dump(reg, MODEL_REGRESSION_PATH)
        
        logger.info(f"✅ Modeller güncellendi | Veri: {len(df)} işlem")
        
    except Exception as e:
        logger.error(f"Model eğitim hatası: {str(e)}")

def check_data_quality():
    if not os.path.exists(HISTORY_FILE):
        return False
        
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
        
    if len(history) < 20:
        return True
        
    win_rate = sum(1 for x in history if x['profit_usdt'] > 0) / len(history)
    return 0.2 < win_rate < 0.8

def load_positions():
    try:
        if not os.path.exists(POSITION_FILE) or os.stat(POSITION_FILE).st_size == 0:
            return []
            
        with open(POSITION_FILE, "r") as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading positions: {e}")
        return []

def save_positions(positions):
    try:
        with open('positions.json', 'w') as f:
            json.dump(positions, f)
        with open('position_count.txt', 'w') as f:
            f.write(str(len(positions)))
    except Exception as e:
        print("⚠️ POSITION YAZMA HATASI:", str(e))
    
def load_blacklist():
    try:
        with open("blacklist.json", "r") as f:
            return json.load(f)    
    
    except Exception as e:
        logger.error(f"Monitor Loop Kritik Hata: {str(e)}\n{traceback.format_exc()}")
async def my_async_function():
    await asyncio.sleep(60)
def load_symbol_cache():
    if not os.path.exists(SYMBOL_CACHE_FILE):
        return None
    try:
        with open(SYMBOL_CACHE_FILE, "r") as f:
            cache = json.load(f)
        
        if "timestamp" not in cache or "symbols" not in cache:
            logger.warning("Symbol cache yapısı eksik. Yeniden oluşturulacak.")
            return None
        
        timestamp = datetime.fromisoformat(cache["timestamp"])
        time_diff = (datetime.utcnow() - timestamp).total_seconds()
        if time_diff > SYMBOL_CACHE_TTL * 60:
            return None
        
        return cache["symbols"]
    except Exception as e:
        logger.error(f"Cache yükleme hatası: {e}")
        return None

def record_closed_trade(pos, exit_price, reason):
    entry = pos["entry_price"]
    size = pos["size"]
    side = pos["side"]

    if side == "long":
        profit_usdt = (exit_price - entry) * size
        pnl_percent = ((exit_price - entry) / entry) * 100
    else:
        profit_usdt = (entry - exit_price) * size
        pnl_percent = ((entry - exit_price) / entry) * 100

    trade = {
        "symbol": pos["symbol"],
        "side": side,
        "entry_price": entry,
        "exit_price": exit_price,
        "pnl_percent": round(pnl_percent, 2),
        "size": size,
        "profit_usdt": round(profit_usdt, 4),
        "closed_reason": reason,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_strength": pos.get("signal_strength", 0),
        "rsi": pos.get("rsi", 0),
        "ml_probability": pos.get("ml_probability", 0),
        "mode": pos.get("mode", "real")
    }

    for feature in REQUIRED_FEATURES:
        trade[feature] = pos.get(feature, 0)

    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([convert_numpy(trade)], f, indent=4)
    else:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        history.append(trade)
        with open(HISTORY_FILE, "w") as f:
            json.dump(convert_numpy(history), f, indent=4)

def create_position(
    symbol: str,
    side: str,
    entry_price: float,
    signal_strength: int,
    rsi: float,
    probability: float,
    atr: float = None,
    current_price: float = None,
    account_balance: float = None,
    force_train_mode: bool = False
) -> Optional[Dict]:
    try:
        if atr and account_balance and current_price:
            position_size = RiskCalculator.calculate_position_size(
                symbol, atr, current_price, account_balance
            )
        else:
            position_size = DEFAULT_TRAINING_SIZE

        mode = "training" if force_train_mode else "real"

        tp1 = entry_price * (1.01 if side == "long" else 0.99)
        tp2 = entry_price * (1.02 if side == "long" else 0.98)
        sl  = entry_price * (0.99 if side == "long" else 1.01)

        return {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "signal_strength": signal_strength,
            "rsi": rsi,
            "ml_probability": probability,
            "size": round(position_size, 3),
            "mode": mode,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "atr": atr,
            "account_risk": 0,
            "tp1": round(tp1, 4),
            "tp2": round(tp2, 4),
            "sl": round(sl, 4),
            "tp1_hit": False,
            "tp2_hit": False,
            "trailing_active": False,
            "peak_price": entry_price,
            "ema_diff": None,
            "macd_direction": None,
            "bb_position": None,
            "volume_ratio": None,
            "atr_percent": None,
            "side_encoded": 1 if side == "long" else 0,
            "symbol_encoded": 0,
            "partial_tp1_done": False,
            "size_closed": 0.0
        }

    except Exception as e:
        logger.error(f"{symbol} için pozisyon oluşturulurken hata: {e}")
        return None

def log_shadow_trade(symbol, side, signal_strength, rsi, probability):
    trade = {
        "symbol": symbol,
        "side": side,
        "entry_price": 0,
        "exit_price": 0,
        "pnl_percent": None,
        "size": 0,
        "profit_usdt": 0,
        "closed_reason": "shadow",
        "timestamp": datetime.utcnow().isoformat(),
        "signal_strength": signal_strength,
        "rsi": rsi,
        "ml_probability": probability,
        "mode": "shadow",
        "features": {
            "ema_diff": (current_price - ema[-1])/ema[-1]*100,
            "bb_position": (current_price - lower_bb[-1])/(upper_bb[-1] - lower_bb[-1]),
            "volume_ratio": volume[-1]/np.mean(volume[-20:])
        }
    }

    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except:
        history = []

    history.append(trade)

    with open(HISTORY_FILE, "w") as f:
        json.dump(convert_numpy(history), f, indent=4)

    logger.info(f"👻 Shadow işlem gözlemi kaydedildi: {symbol} (%{probability*100:.1f})")

def predict_signal(symbol_data):
    try:
        features = prepare_features(symbol_data)
        if len(features) != len(REQUIRED_FEATURES):
            logger.error(f"Özellik boyutu uyumsuz: {symbol_data['symbol']}")
            return None
        
        prediction = model_cls.predict([features])
        return prediction[0]
    except Exception as e:
        logger.error(f"Tahmin hatası ({symbol_data['symbol']}): {str(e)}")
        return None

async def check_signal(symbol, rsi, ema, close_prices, volume, open_prices, high_prices, low_prices,
                      macd, macd_signal, upper_bb, lower_bb, historical_data):
    try:
        logger.info(f"🔍 Sinyal kontrolü: {symbol}")
        
        if close_prices[-1] < MIN_PRICE or not symbol.endswith("USDT"):
            logger.info(f"⏩ Atlandı: {symbol} (fiyat < MIN_PRICE veya USDT değil)")
            return None, 0, {}, 0

        last_close = close_prices[-1] if len(close_prices) > 0 else 0                                                    
        last_rsi = rsi[-1] if len(rsi) > 0 else 50
        last_ema = ema[-1] if len(ema) > 0 else last_close

        ema_diff = ((last_close - last_ema) / last_ema * 100) if last_ema != 0 else 0
        volume_window = volume[-20:-1] if len(volume) >= 20 else []
        avg_volume = np.mean(volume_window) if len(volume_window) > 0 else 1
        vol_ratio = volume[-1] / avg_volume if avg_volume != 0 else 1

        atr_percent = 0
        if symbol in historical_data and "atr" in historical_data[symbol]:
            atr = historical_data[symbol]["atr"]
            atr_percent = atr / last_close * 100 if last_close != 0 else 0

        features = {
            "signal_strength": 1,
            "rsi": last_rsi,
            "ema_diff": ema_diff,
            "macd_direction": 1 if macd[-1] > macd_signal[-1] else 0,
            "bb_position": (last_close - lower_bb[-1]) / (upper_bb[-1] - lower_bb[-1]) * 100 
                          if (upper_bb[-1] - lower_bb[-1]) != 0 else 50,
            "volume_ratio": vol_ratio,
            "atr_percent": atr_percent,      
        }

        try:
            features["symbol_encoded"] = label_encoder.transform([symbol])[0]
        except ValueError:
            if not hasattr(label_encoder, 'classes_'):
                label_encoder.fit(["BTCUSDT"])
            features["symbol_encoded"] = 0
            logger.warning(f"⚠️ Yeni sembol: {symbol}. BTCUSDT kodu kullanıldı")

        ml_result = predict_with_ml(features)
        if not ml_result:
            return None, 0, {}, 0

        probability = ml_result['probability']
        side = 'long' if ml_result['signal'] == 1 else 'short'
        features["signal_strength"] = max(1, int(probability * 5))

        return side, features["signal_strength"], features, probability

    except Exception as e:
        logger.error(f"Sinyal işleme hatası: {symbol} - {str(e)}")
        return None, 0, {}, 0

def save_symbol_cache(symbols):
    try:
        cache = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbols": symbols
        }
        with open(SYMBOL_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        logger.error(f"Symbol cache kaydetme hatası: {e}")

async def open_position(
    positions, symbol, side, price, rsi, ema_val, high, low, close, volume,
    strength, features, probability,
    atr=None, account_balance=None, current_price=None
):
    if any(p['symbol'] == symbol for p in positions):
        return
    if len(positions) >= MAX_POSITIONS:
        return

    tp1 = tp2 = sl = None
    if model_reg:
        try:
            X = pd.DataFrame([features], columns=REQUIRED_FEATURES)
            preds = model_reg.predict(X)[0]
            tp1, tp2, sl = preds[0], preds[1], preds[2]
        except Exception as e:
            logger.error(f"Regresyon modeli hatası: {e}")
            tp1, tp2, sl = fallback_trade_parameters()
    else:
        tp1, tp2, sl = fallback_trade_parameters()

    startup_mode = datetime.now() - bot_start_time < timedelta(hours=24)
    pos = create_position(
    symbol, 
    side, 
    price, 
    strength, 
    rsi, 
    probability,
    atr=atr,
    current_price=price,
    account_balance=get_current_balance(),
    force_train_mode=startup_mode
)

    if not pos:
        return

    pos.update({
        "entry_price": price,
        "tp1": round(price * (1 + tp1 if side == "long" else 1 - tp1), 6),
        "tp2": round(price * (1 + tp2 if side == "long" else 1 - tp2), 6),
        "sl": round(price * (1 - sl if side == "long" else 1 + sl), 6),
        "tp1_hit": False,
        "tp2_hit": False,
        "trailing_active": False,
        "peak_price": price,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_strength": strength,
        "ml_probability": probability
    })
    
    for feature in REQUIRED_FEATURES:
        pos[feature] = features.get(feature, 0)

    positions.append(pos)
    save_positions(positions)
    
    mode_emoji = "🧪" if pos["mode"] == "training" else "🎯"
    mode_text = "EĞİTİM MODU" if pos["mode"] == "training" else "GERÇEK POZİSYON"
    
    entry_msg = f"""
{mode_emoji} *YENİ POZİSYON* {'🟢 LONG' if side == 'long' else '🔴 SHORT'} ({mode_text})
-----------------------------
• *Coin*: `{symbol}`
• *Giriş Fiyatı*: `{price}`
• *RSI*: `{round(rsi, 2)}`
• *EMA Farkı*: `{round((price - ema_val)/ema_val*100, 2)}%`
• *Hedefler*: TP1=`{pos['tp1']}` | TP2=`{pos['tp2']}`
• *Stop Loss*: `{pos['sl']}`
• *Büyüklük*: `{pos['size']} USDT`
• *Sinyal Gücü*: `{strength}/5`
• *Tahmin Olasılığı*: `{round(probability*100, 2)}%`
• *Göstergeler*: MACD {'↑' if features.get('macd_direction', 0) == 1 else '↓'} | BB {'↑' if price > talib.MA(close, timeperiod=20)[-1] else '↓'}
-----------------------------
⏳ `{pos["timestamp"]} UTC`"""
    
    await send_telegram_message(entry_msg)

async def check_positions(positions):
    if not positions:
        return positions

    updated_positions = []

    for pos in positions:
        data = await fetch_klines(pos["symbol"])
        if not data:
            updated_positions.append(pos)
            continue

        current_price = float(data[-1]['close'])
        previous_price = float(data[-2]['close'])
        side = pos["side"]

        price_jump = abs(current_price - previous_price) / previous_price
        if price_jump >= 0.05:
            record_closed_trade(pos, current_price, "Ani Fiyat Hareketi")
            await send_telegram_message(f"""
🚨 *Acil Kapatma: Ani Hareket (> %5)*
--------------------------------
• *Coin*: `{pos['symbol']}`
• *Fiyat Değişimi*: `{price_jump * 100:.2f}%`
• *Kapanış Fiyatı*: `{current_price}`
• *Zaman*: `{datetime.utcnow().strftime('%H:%M:%S')} UTC`
""")
            continue

        pos.setdefault("partial_tp1_done", False)
        pos.setdefault("size_closed", 0.0)
        pos.setdefault("peak_price", pos["entry_price"])

        if not pos["tp1_hit"]:
            if (side == "long" and current_price >= pos["tp1"]) or (side == "short" and current_price <= pos["tp1"]):
                pos["tp1_hit"] = True
                pos["partial_tp1_done"] = True
                partial_close = pos["size"] * 0.3
                pos["size_closed"] += partial_close
                pos["size"] -= partial_close
                pos["sl"] = pos["entry_price"]

                await send_telegram_message(f"""
✅ *TP1 Hedef (%30 kapandı)* {'🟢' if side == 'long' else '🔴'}
-----------------------------
• *Coin*: `{pos['symbol']}`
• *Fiyat*: `{current_price}`
• *Yeni SL*: `{pos['sl']}`
• *Kalan Büyüklük*: `{pos['size']} USDT`
• *Zaman*: `{datetime.utcnow().strftime('%H:%M:%S')} UTC`
""")

        elif not pos["tp2_hit"]:
            if (side == "long" and current_price >= pos["tp2"]) or (side == "short" and current_price <= pos["tp2"]):
                pos["tp2_hit"] = True
                pos["trailing_active"] = True
                pos["peak_price"] = current_price

                await send_telegram_message(f"""
🎯 *TP2 Hedef (Trailing SL aktif)* {'🟢' if side == 'long' else '🔴'}
-----------------------------
• *Coin*: `{pos['symbol']}`
• *Fiyat*: `{current_price}`
• *Trailing Başladı*
• *Zaman*: `{datetime.utcnow().strftime('%H:%M:%S')} UTC`
""")

        if pos.get("trailing_active"):
            if (side == "long" and current_price > pos["peak_price"]) or (side == "short" and current_price < pos["peak_price"]):
                pos["peak_price"] = current_price

            trail_offset = 0.005
            trail_sl = pos["peak_price"] * (1 - trail_offset) if side == "long" else pos["peak_price"] * (1 + trail_offset)
            if (side == "long" and current_price <= trail_sl) or (side == "short" and current_price >= trail_sl):
                record_closed_trade(pos, current_price, "Trailing SL")
                await send_telegram_message(f"""
🟡 *Trailing SL Devreye Girdi* {'🟢' if side == 'long' else '🔴'}
-----------------------------
• *Coin*: `{pos['symbol']}`
• *Peak*: `{pos['peak_price']}`
• *Kapanış Fiyatı*: `{current_price}`
• *Kalan Miktar*: `{pos['size']} USDT`
• *Zaman*: `{datetime.utcnow().strftime('%H:%M:%S')} UTC`
""")
                continue

        if (side == "long" and current_price <= pos["sl"]) or (side == "short" and current_price >= pos["sl"]):
            loss_percent = abs(pos['entry_price'] - current_price) / pos['entry_price'] * 100
            record_closed_trade(pos, current_price, "Stop Loss")
            await send_telegram_message(f"""
❌ *STOP LOSS* {'🟢' if side == 'long' else '🔴'}
--------------------------------
• *Coin*: `{pos['symbol']}`
• *Fiyat*: `{current_price}`
• *Kayıp*: `{loss_percent:.2f}%`
• *SL Türü*: {'Trailing' if pos.get('trailing_active') else 'Normal'}
• *Zaman*: `{datetime.utcnow().strftime('%H:%M:%S')} UTC`
""")
            continue

        updated_positions.append(pos)

    save_positions(updated_positions)
    return updated_positions

async def fetch_klines(symbol, interval='5m', limit=300, retries=3):
    for attempt in range(retries):
        client = None
        try:
            client = await init_binance_client()
            klines = await asyncio.wait_for(
                client.get_klines(symbol=symbol, interval=interval, limit=limit),
                timeout=10
            )
            
            if not klines or len(klines) < 20:
                logger.warning(f"Insufficient klines data for {symbol}")
                return None
                
            validated = []
            for k in klines:
                if len(k) >= 6:
                    validated.append({
                        'timestamp': datetime.fromtimestamp(k[0]/1000),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    })
            
            return validated
            
        except asyncio.TimeoutError:
            logger.warning(f"{symbol} timeout (attempt {attempt+1}/{retries})")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"{symbol} kline hatası: {str(e)}")
            await asyncio.sleep(5)
        finally:
            if client:
                await client.close_connection()
    
    logger.error(f"{symbol} verisi alınamadı after {retries} attempts")
    return None

async def fetch_liquidity_data(symbol):
    client = None
    try:
        client = await init_binance_client()
        ticker = await client.get_ticker(symbol=symbol)
        return {
            'volume_24h': float(ticker['quoteVolume']),
            'price_change': float(ticker['priceChangePercent'])
        }
    except Exception as e:
        logger.error(f"Likidite verisi alınamadı {symbol}: {e}")
        add_to_blacklist(symbol)
        return {'volume_24h': 0, 'price_change': 0}
    finally:
        if client:
            await client.close_connection()

async def fetch_historical_data(symbols: list) -> dict:
    historical_data = {}
    for symbol in symbols:
        try:
            klines = await fetch_klines(symbol, interval='4h', limit=100)
            if not klines:
                continue
                
            closes = np.array([k['close'] for k in klines])
            volumes = np.array([k['volume'] for k in klines])
            
            historical_data[symbol] = {
                'volume_24h': sum(volumes[-6:]),
                'atr': talib.ATR(
                    np.array([k['high'] for k in klines]),
                    np.array([k['low'] for k in klines]),
                    closes,
                    14
                )[-1]
            }
        except Exception as e:
            logger.error(f"Tarihsel veri hatası {symbol}: {e}")
            historical_data[symbol] = {'volume_24h': 0, 'atr': 0}
    
    return historical_data

async def fetch_symbols():
    try:
        client = await init_binance_client()
        exchange_info = await client.futures_exchange_info()
        symbols = [
            s["symbol"] for s in exchange_info["symbols"]
            if s["contractType"] == "PERPETUAL" and s["symbol"].endswith("USDT")
        ]
        await client.close_connection()

        blacklist = load_blacklist()
        symbols = [s for s in symbols if s not in blacklist]

        save_symbol_cache(symbols)
        return symbols
    except Exception as e:
        logger.error(f"Binance sembolleri alınamadı: {e}")
        return []

async def monitor_positions_loop():
    while True:
        try:
            positions = load_positions()
            if not positions:
                await asyncio.sleep(15)
                continue
                
            try:
                client = await init_binance_client()
                await client.futures_ping()
                await client.close_connection()
            except Exception as e:
                logger.error(f"API Ping Hatası: {str(e)}")
                await asyncio.sleep(30)
                continue
                
            positions = await check_positions(positions)
            save_positions(positions)
            
        except Exception as e:
            logger.error(f"Monitor Loop Kritik Hata: {str(e)}\n{traceback.format_exc()}")
            await asyncio.sleep(60)

async def calculate_open_pnl(positions):
    if not positions:
        return None, None, []
    
    total_pnl = 0
    pnl_details = []
    position_counts = {"long": 0, "short": 0}
    
    for pos in positions:
        data = await fetch_klines(pos["symbol"], limit=1)
        if not data:
            continue
            
        current_price = float(data[-1]['close'])
        entry = pos["entry_price"]
        side = pos["side"]
        
        position_counts[side] += 1
        
        pnl = (current_price - entry)/entry*100 if side == "long" else (entry - current_price)/entry*100
        total_pnl += pnl * (pos["size"]/10)
        
        pnl_details.append({
            "symbol": pos["symbol"],
            "side": side,
            "pnl": round(pnl, 2),
            "size": pos["size"]
        })
    
    avg_pnl = total_pnl / len(positions) if positions else 0
    return avg_pnl, position_counts, sorted(pnl_details, key=lambda x: abs(x["pnl"]), reverse=True)

async def periodic_pnl_report_loop():
    last_open_report = None
    while True:
        now = datetime.utcnow()

        if now.hour == 0 and now.minute == 0:
            daily = generate_pnl_report(days=1)
            weekly = generate_pnl_report(days=7)
            monthly = generate_pnl_report(days=30)
            await send_telegram_message(daily)
            await send_telegram_message(weekly)
            await send_telegram_message(monthly)
            await asyncio.sleep(60 * 60)

        if (last_open_report is None or (now - last_open_report).total_seconds() > PNL_REPORT_INTERVAL):
            positions = load_positions()
            if positions:
                avg_pnl, position_counts, pnl_details = await calculate_open_pnl(positions)
                report = f"""
📊 *Açık Pozisyonlar PnL Özeti*
------------------------------
• Uzun Pozisyon: {position_counts['long']}
• Kısa Pozisyon: {position_counts['short']}
• Ortalama PnL: {avg_pnl:.2f}%
• Toplam Açık Pozisyon: {len(positions)}
------------------------------
"""
                for detail in pnl_details:
                    report += (
                        f"• `{detail['symbol']}` | {'🟢' if detail['side']=='long' else '🔴'} | PnL: `{detail['pnl']}%` | Size: `{detail['size']}`\n"
                    )
                await send_telegram_message(report)
                last_open_report = now
            else:
                last_open_report = now
        await asyncio.sleep(30)

def generate_pnl_report(days=1):
    if not os.path.exists(HISTORY_FILE):
        return "📉 *Hiç kapanmış işlem bulunamadı.*"

    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    now = datetime.now()
    cutoff = now - timedelta(days=days)
    filtered = [t for t in history if datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff]

    if not filtered:
        return f"📉 *Son {days} gün içinde işlem yok.*"

    total_pnl = sum(t["profit_usdt"] for t in filtered)
    win_count = sum(1 for t in filtered if t["profit_usdt"] > 0)
    loss_count = sum(1 for t in filtered if t["profit_usdt"] < 0)

    label = "Günlük" if days == 1 else "Haftalık" if days == 7 else "Aylık" if days == 30 else f"Son {days} Günlük"

    report = f"""
📊 *{label} İşlem Özeti*
-----------------------------
✅ Kar Eden İşlem: {win_count}
❌ Zarar Eden İşlem: {loss_count}
💰 Net PnL: {total_pnl:.2f} USDT
📈 Toplam İşlem: {len(filtered)}
🕒 Rapor Zamanı: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
    return report

async def training_monitor():
    while True:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        
        if len(history) % 50 == 0:
            df = pd.DataFrame(history)
            if len(df) > 100:
                win_rate = (df['profit_usdt'] > 0).mean()
                if 0.25 < win_rate < 0.75:
                    train_models()
                    
        await asyncio.sleep(3600)

async def trading_strategy_loop():
    while True:
        try:
            logger.info("Starting strategy loop iteration")
            positions = load_positions()
            logger.info(f"Loaded {len(positions)} current positions")
            
            symbols = await fetch_symbols()
            logger.info(f"Fetched {len(symbols)} symbols from exchange")

            blacklist = load_blacklist()
            filtered_symbols = [s for s in symbols if s not in blacklist]
            if len(filtered_symbols) < len(symbols):
                print(f"Blacklist filtered {len(symbols) - len(filtered_symbols)} symbols")

            liquid_symbols = []
            for s in filtered_symbols:
                data = await fetch_liquidity_data(s)
                if data['volume_24h'] > LIQUIDITY_THRESHOLD:
                    liquid_symbols.append(s)
                else:
                    print(f"{s} | Skipped: Low volume ({data['volume_24h']} < {LIQUIDITY_THRESHOLD})")

                if model_monitor.check_decay():
                    await model_monitor.retrain_model()

            historical_data = await fetch_historical_data(liquid_symbols)
            ranked_symbols = SymbolRanker.rank_symbols(liquid_symbols, historical_data)
            
            print(f"Top {len(ranked_symbols[:20])} ranked symbols to process:")
            for symbol, score in ranked_symbols[:20]:
                print(f"Processing {symbol} (score: {score:.2f})")
                
                if len(positions) >= MAX_POSITIONS:
                    print("Max positions reached, breaking loop")
                    break
                    
                if any(p['symbol'] == symbol for p in positions):
                    print(f"{symbol} | Skipped: Already in positions")
                    continue

                if symbol in cooldown_tracker:
                    remaining = (datetime.now() - cooldown_tracker[symbol]).total_seconds() / 60
                    if remaining < COOLDOWN_MINUTES:
                        print(f"{symbol} | Skipped: In cooldown ({remaining:.1f}m remaining)")
                        continue

                klines = await fetch_klines(symbol)
                if not klines or len(klines) < 30:
                    print(f"{symbol} | Skipped: Insufficient kline data")
                    continue

                close = np.array([k['close'] for k in klines])
                high = np.array([k['high'] for k in klines])
                low = np.array([k['low'] for k in klines])
                open_ = np.array([k['open'] for k in klines])
                volume = np.array([k['volume'] for k in klines])
                
                rsi = talib.RSI(close, 14)
                ema = talib.EMA(close, 20)
                macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                upper_bb, middle_bb, lower_bb = talib.BBANDS(close, timeperiod=20)
                atr = talib.ATR(high, low, close, 14)[-1]

                side, strength, features, probability = await check_signal(
                    symbol, rsi, ema, close, volume, open_, high, low,
                    macd, macd_signal, upper_bb, lower_bb, historical_data
                )

                if side:
                    account_balance = 10000
                    position_size = RiskCalculator.calculate_position_size(symbol, atr, close[-1], account_balance)
                    
                    await open_position(
                        positions, symbol, side, close[-1], rsi[-1], ema[-1],
                        high, low, close, volume, strength, features, probability,
                        atr=atr,
                        current_price=close[-1],
                        account_balance=account_balance
                    )
                    await asyncio.sleep(0.2)

            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)

            if len(history) % 100 == 0 and check_data_quality():
                train_models()

            if len(history) > 1000 and check_data_quality():
                os.environ['FULL_AUTO_MODE'] = 'true'

            await asyncio.sleep(COOLDOWN_MINUTES * 60)
            
        except Exception as e:
            await send_telegram_message(f"🔴 *STRATEJI HATASI*: {repr(e)}")
            logger.error(f"Strategy loop error: {repr(e)} | Type: {type(e)}")
            print(f"⛔ STRATEGY LOOP ERROR: {repr(e)}")
            await asyncio.sleep(30)

async def main():
    initialize_files()
    load_models()

    if not all([model_cls, model_reg, scaler]):
        logger.error("Kritik hata: Modeller yüklenemedi")
        await send_telegram_message("🔴 Kritik Hata: Modeller yüklenemedi! Bot kapatılıyor...")
        return

    print("\033[92m" + "="*50)
    print("🤖 BOT AKTİF (V6.1 - Güncellenmiş ML Entegrasyonu)")
    print(f"• Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"• Maks. Pozisyon: {MAX_POSITIONS} | Cooldown: {COOLDOWN_MINUTES}dk")
    print(f"• ML Threshold: {ML_THRESHOLD} | Training Boyutu: {TRAINING_POSITION_SIZE} USDT")
    print("="*50 + "\033[0m")
    
    await send_telegram_message(
        f"🤖 *Bot Başladı (V6.1 - Güncellenmiş ML Entegrasyonu)*\n"
        f"• Maks. Pozisyon: `{MAX_POSITIONS}`\n"
        f"• Cooldown: `{COOLDOWN_MINUTES}dk`\n"
        f"• ML Threshold: `{ML_THRESHOLD}`\n"
        f"• Training Boyutu: `{TRAINING_POSITION_SIZE} USDT`\n"
        f"• ML Modeller: {'Aktif' if model_cls and model_reg else 'Pasif'}"
    )

    try:
        positions = load_positions()
        if positions:
            message = ["🔍 *Mevcut Pozisyonlar Tespit Edildi:*"]
            for pos in positions:
                side_emoji = "🟢" if pos["side"] == "long" else "🔴"
                mode_emoji = "🧪" if pos.get("mode") == "training" else "🎯"
                message.append(f"""
{mode_emoji} {side_emoji} `{pos['symbol']}` ({pos['side'].upper()}):
• Entry: `{pos['entry_price']}`
• TP1: `{pos['tp1']}` | TP2: `{pos['tp2']}`
• SL: `{pos['sl']}`
""")
            await send_telegram_message("\n".join(message))
    except Exception as e:
        logger.error(f"Pozisyon yükleme hatası: {e}")

    tasks = [
        asyncio.create_task(trading_strategy_loop()),
        asyncio.create_task(monitor_positions_loop()),
        asyncio.create_task(training_monitor()),
        asyncio.create_task(periodic_pnl_report_loop())  
    ]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

if __name__ == "__main__":    
    asyncio.run(main())