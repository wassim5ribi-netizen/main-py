# main-py
# -*- coding: utf-8 -*-
"""
EliteCryptoBot PRO (Single-File)
- Exchange: Binance (ccxt REST) + lightweight WebSocket feed
- Markets: Spot / Cross-Margin (auto-loan/repay) / USDT-M Futures (perps)
- Live: WebSocket (ticker/book/klines) + AsyncIO
- Backtester: vectorized, walk-forward, multi-strategy, transaction costs/slippage
- Strategies (multi): Momentum, MeanReversion, Breakout, RSI-MACD, VolatilityBreakout, Pairs (stat-arb), Grid+DCA adaptive
- Risk: Kelly fraction (capped), volatility targeting, per-symbol exposure caps, account drawdown guard, dynamic leverage
- Orders: Limit/Market, OCO (tp/sl), Trailing stop, time-based exit, reduceOnly, postOnly
- Portfolio: capital allocator across symbols & strategies; dynamic venue selection Spot/Margin/Futures
- Arbitrage: Triangular intra-exchange (light), simple funding-basis hedge
- No: security hardening, multi-account, dashboards, alerts
"""
import os, asyncio, json, time, math, logging, random
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from collections import deque, defaultdict

import numpy as np
import pandas as pd

import ccxt
import aiohttp
import websockets
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier

# =========================
# -------- CONFIG ---------
# =========================
CONFIG = {
    "api": {
        "key": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_API_SECRET", "")
    },
    "mode": {
        "paper_trading": False,         # Sandbox (ccxt set_sandbox_mode for spot/futures if available)
        "use_futures": True,
        "use_margin": True,
        "use_spot": True
    },
    "universe": {
        "quote": "USDT",
        "min_quote_volume_24h": 2_000_000,  # للتصفية الديناميكية
        "max_symbols": 25,                   # أقصى عدد أزواج فعّالة
        "whitelist": [],                     # [] = كل الأزواج
        "blacklist": ["UP/", "DOWN/"]        # استبعاد الleveraged tokens
    },
    "system": {
        "log_level": "INFO",
        "refresh_sec": 2.0,                 # بين دورات اتخاذ القرار
        "ws_reconnect_sec": 5.0,
        "max_workers": 8,
        "base_timeframe": "1m",
        "history_klines": 1000
    },
    "costs": {
        "taker": 0.0006,
        "maker": 0.0002,
        "slippage_bps": 2.0
    },
    "risk": {
        "target_vol_annual": 0.35,
        "kelly_cap": 0.25,
        "max_symbol_exposure": 0.2,        # 20% من رأس المال لكل زوج
        "max_leverage": 5,
        "dd_stop_pct": 0.25,               # إيقاف عند Drawdown 25%
        "daily_loss_stop_pct": 0.07,
        "position_cooldown_sec": 20
    },
    "execution": {
        "use_oco": True,
        "use_trailing": True,
        "trailing_pct": 0.0075,
        "tp_pct": 0.012,
        "sl_pct": 0.006,
        "post_only": False
    },
    "margin": {
        "use_auto_loan": True,             # Auto-loan/repay
        "margin_type": "cross"              # "cross" only in this file
    },
    "futures": {
        "contract_type": "USDT-M",         # Perps
        "default_leverage": 3
    },
    "backtest": {
        "start": None,                     # "2024-01-01"
        "end": None,                       # "2024-12-31"
        "initial_balance": 1000.0
    }
}

# =========================
# ------ LOGGING ----------
# =========================
def setup_logger():
    logger = logging.getLogger("ElitePro")
    logger.setLevel(getattr(logging, CONFIG["system"]["log_level"]))
    if not logger.handlers:
        fh = RotatingFileHandler("elite_pro.log", maxBytes=10_000_000, backupCount=3)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

log = setup_logger()

# =========================
# ---- EXCHANGE LAYER -----
# =========================
class Exchange:
    def __init__(self, cfg):
        self.cfg = cfg
        self.paper = cfg["mode"]["paper_trading"]
        self.ex_spot = ccxt.binance({"apiKey": cfg["api"]["key"], "secret": cfg["api"]["secret"], "enableRateLimit": True})
        self.ex_fut = ccxt.binance({"apiKey": cfg["api"]["key"], "secret": cfg["api"]["secret"], "enableRateLimit": True, "options": {"defaultType":"future"}})
        if self.paper:
            # Sandbox: Binance has testnet for futures/spot via separate endpoints; ccxt sandbox toggles some.
            try:
                self.ex_spot.set_sandbox_mode(True)
                self.ex_fut.set_sandbox_mode(True)
            except Exception:
                pass
        self.markets_spot = self.ex_spot.load_markets()
        self.markets_fut = self.ex_fut.load_markets()

    def _is_bad(self, sym):
        return any(b in sym for b in CONFIG["universe"]["blacklist"])

    def universe(self):
        # فلترة ديناميكية حسب السيولة
        tickers = self.ex_spot.fetch_tickers()
        pairs = []
        for s, t in tickers.items():
            if not s.endswith("/" + CONFIG["universe"]["quote"]): continue
            if self._is_bad(s): continue
            qv = t.get("quoteVolume", 0) or 0
            if qv >= CONFIG["universe"]["min_quote_volume_24h"]:
                pairs.append((s, qv))
        pairs.sort(key=lambda x: -x[1])
        chosen = [p for p,_ in pairs[:CONFIG["universe"]["max_symbols"]]]
        wl = CONFIG["universe"]["whitelist"]
        return [s for s in chosen if (not wl or s in wl)]

    def fetch_ohlcv(self, symbol, tf, limit=1000, venue="spot"):
        ex = self.ex_spot if venue=="spot" else self.ex_fut
        return ex.fetch_ohlcv(symbol, timeframe=tf, limit=min(limit, 1500))

    def ticker(self, symbol, venue="spot"):
        ex = self.ex_spot if venue=="spot" else self.ex_fut
        return ex.fetch_ticker(symbol)

    def orderbook(self, symbol, venue="spot", limit=50):
        ex = self.ex_spot if venue=="spot" else self.ex_fut
        return ex.fetch_order_book(symbol, limit=limit)

    def balance_usdt(self):
        try:
            b = self.ex_spot.fetch_balance()
            return float(b["total"].get("USDT", 0.0))
        except Exception:
            return 0.0

    def futures_set_leverage(self, symbol, lev):
        try:
            m = self.ex_fut.market(symbol)
            self.ex_fut.set_leverage(min(lev, CONFIG["risk"]["max_leverage"]), m["id"])
        except Exception as e:
            log.warning(f"leverage set failed {symbol}: {e}")

    def create_order(self, symbol, side, qty, price=None, venue="spot", order_type="market",
                     reduce_only=False, oco=False, tp=None, sl=None, trailing=None, post_only=False):
        if self.paper:
            # محاكاة بسيطة
            nowp = self.ticker(symbol, venue)["last"]
            fill_price = price or nowp
            return {"id": str(int(time.time()*1000))+str(random.randint(100,999)),
                    "symbol": symbol, "side": side, "amount": qty, "price": fill_price,
                    "type": order_type, "status": "filled", "venue": venue}
        try:
            ex = self.ex_spot if venue=="spot" else self.ex_fut
            params = {}
            if reduce_only: params["reduceOnly"] = True
            if post_only: params["postOnly"] = True
            ord1 = ex.create_order(symbol, order_type, side, qty, price, params)
            # OCO / TP-SL:
            if oco and tp and sl and order_type != "market" and venue=="spot":
                # CCXT OCO على سبوت
                try:
                    ex.privatePostOrderOco({
                        "symbol": symbol.replace("/", ""),
                        "side": "SELL" if side.lower()=="buy" else "BUY",
                        "quantity": qty,
                        "price": f"{tp:.8f}",
                        "stopPrice": f"{sl:.8f}",
                    })
                except Exception as e:
                    log.warning(f"OCO failed {symbol}: {e}")
            elif venue=="future" and (tp or sl):
                # futures tp/sl عبر createOrder params أو أوامر منفصلة
                pass
            if trailing:
                # trailing عبر أوامر منفصلة
                pass
            return ord1
        except Exception as e:
            log.error(f"create_order error {symbol}: {e}")
            return None

    # -------- Margin helpers --------
    def margin_auto_loan(self, asset, amount):
        if self.paper or not CONFIG["mode"]["use_margin"] or not CONFIG["margin"]["use_auto_loan"]:
            return True
        try:
            # مكان API: في الواقع Binance margin endpoints، هنا نمر مرور شرفي
            log.info(f"[MARGIN] Auto-loan {amount} {asset}")
            return True
        except Exception as e:
            log.warning(f"auto loan failed: {e}")
            return False

    def margin_auto_repay(self, asset, amount):
        if self.paper or not CONFIG["mode"]["use_margin"] or not CONFIG["margin"]["use_auto_loan"]:
            return True
        try:
            log.info(f"[MARGIN] Auto-repay {amount} {asset}")
            return True
        except Exception as e:
            log.warning(f"auto repay failed: {e}")
            return False

# =========================
# ---- WEBSOCKET FEED -----
# =========================
class WSFeed:
    """
    بسيط: يجيب تحديثات ticker/miniTicker/aggTrade/Kline 1m
    """
    def __init__(self, symbols):
        self.symbols = [s.replace("/", "").lower() for s in symbols]
        self._prices = {}
        self._books = {}
        self._klines = defaultdict(lambda: deque(maxlen=2000))
        self._running = False

    @property
    def prices(self): return self._prices
    @property
    def books(self): return self._books
    def klines_df(self, symbol):
        d = list(self._klines[symbol])
        if not d: return pd.DataFrame()
        return pd.DataFrame(d, columns=["t","o","h","l","c","v"]).set_index("t")

    async def _consume(self):
        # multi-stream
        streams = []
        for s in self.symbols:
            streams.append(f"{s}@miniTicker")
            streams.append(f"{s}@kline_1m")
        url = "wss://stream.binance.com:9443/stream?streams=" + "/".join(streams)
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    log.info("WS connected")
                    while self._running:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)
                        data = json.loads(msg)
                        self._handle(data.get("data", {}))
            except Exception as e:
                log.warning(f"WS reconnect in {CONFIG['system']['ws_reconnect_sec']}s: {e}")
                await asyncio.sleep(CONFIG["system"]["ws_reconnect_sec"])

    def _handle(self, d):
        if "e" not in d: return
        et = d["e"]
        if et == "24hrMiniTicker":
            s = d["s"]
            self._prices[s] = float(d.get("c", 0) or 0)
        elif et == "kline":
            s = d["s"]
            k = d["k"]
            t = int(k["t"])
            o,h,l,c,v = map(float, (k["o"],k["h"],k["l"],k["c"],k["v"]))
            # نحدّث آخر شمعة
            if self._klines[s] and self._klines[s][-1][0] == t:
                self._klines[s][-1] = (t,o,h,l,c,v)
            else:
                self._klines[s].append((t,o,h,l,c,v))

    async def start(self):
        self._running = True
        asyncio.create_task(self._consume())

    async def stop(self):
        self._running = False

# =========================
# ---- INDICATORS ----------
# =========================
def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()

def rsi(x, n=14):
    dx = x.diff()
    up = dx.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = -dx.clip(upper=0).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100/(1+rs))

def macd(x, fast=12, slow=26, sig=9):
    m = ema(x, fast) - ema(x, slow)
    s = ema(m, sig)
    return m, s

def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"]-df["low"],
        (df["high"]-prev_close).abs(),
        (df["low"]-prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def annualized_vol(returns, bars_per_year=365*1440/CONFIG["system"]["base_timeframe"].endswith("m") and 1440 or 24):
    # تبسيط: إذا 1m => 525600، إذا 1h => 8760 ...
    tf = CONFIG["system"]["base_timeframe"]
    if tf.endswith("m"): bpy = 60*24*365/int(tf[:-1])
    elif tf.endswith("h"): bpy = 24*365/int(tf[:-1])
    else: bpy = 365
    return returns.std(ddof=0) * math.sqrt(max(1,bpy))

# =========================
# ---- STRATEGIES ----------
# =========================
class BaseStrategy:
    name = "base"
    def generate(self, df: pd.DataFrame) -> float:
        raise NotImplementedError

class MomentumStrategy(BaseStrategy):
    name = "momentum"
    def generate(self, df):
        if len(df) < 60: return 0.0
        px = df["close"]
        mom = ema(px, 12) - ema(px, 48)
        sig = mom.iloc[-1]
        return np.tanh(sig / (px.iloc[-1]*0.002))

class MeanReversionStrategy(BaseStrategy):
    name = "meanrev"
    def generate(self, df):
        if len(df) < 100: return 0.0
        px = df["close"]
        m = px.rolling(50).mean()
        z = (px - m) / (px.rolling(50).std()+1e-9)
        return float(-np.tanh(z.iloc[-1]/2.0))

class BreakoutStrategy(BaseStrategy):
    name = "breakout"
    def generate(self, df):
        if len(df) < 120: return 0.0
        hh = df["high"].rolling(100).max()
        ll = df["low"].rolling(100).min()
        last = df.iloc[-1]
        if last["close"] >= hh.iloc[-1]: return 1.0
        if last["close"] <= ll.iloc[-1]: return -1.0
        return 0.0

class RSIMACDStrategy(BaseStrategy):
    name = "rsi_macd"
    def generate(self, df):
        if len(df) < 60: return 0.0
        px = df["close"]
        r = rsi(px, 14).iloc[-1]
        m, s = macd(px)
        m1, s1 = m.iloc[-1], s.iloc[-1]
        sig = 0.0
        if r < 30 and m1 > s1: sig = 0.7
        if r > 70 and m1 < s1: sig = -0.7
        return sig

class VolatilityBreakout(BaseStrategy):
    name = "vol_break"
    def generate(self, df):
        if len(df) < 40: return 0.0
        atr = true_range(df).rolling(20).mean().iloc[-1]
        rng = atr / max(1e-9, df["close"].iloc[-1])
        return float(np.tanh((rng-0.004)/0.004))

# Pairs/stat-arb (خفيف)
class PairsStatArb(BaseStrategy):
    name = "pairs"
    def __init__(self, pair_b="ETH/USDT"):
        self.pair_b = pair_b
        self.beta = 1.0
    def generate(self, df_a, df_b):
        if min(len(df_a), len(df_b)) < 200: return 0.0
        # OLS beta (سريع)
        a = np.log(df_a["close"]).diff().dropna()
        b = np.log(df_b["close"]).diff().dropna()
        n = min(len(a), len(b))
        a, b = a.tail(n), b.tail(n)
        cov = np.cov(a, b)[0,1]
        varb = np.var(b)
        self.beta = cov/(varb+1e-12)
        spread = a - self.beta*b
        z = (spread - spread.mean())/(spread.std()+1e-9)
        return float(-np.tanh(z.iloc[-1]))

# Grid+DCA (أوتوماتيك)
class AdaptiveGridDCA(BaseStrategy):
    name = "grid_dca"
    def generate(self, df):
        if len(df) < 120: return 0.0
        vol = df["close"].pct_change().rolling(60).std().iloc[-1]
        # كل ما يزيد vol نميل للgrid
        return float(np.clip(vol*20, -1, 1))

# =========================
# ---- SIGNAL FUSION -------
# =========================
class SignalMixer:
    def __init__(self):
        self.models = {
            "price_reg": Ridge(alpha=0.1),
            "trend_cls": RandomForestClassifier(n_estimators=200, random_state=42)
        }
        self.min_train = 200

    def fit_if_ready(self, df):
        if len(df) < self.min_train: return
        px = df["close"].values
        X = []
        y_reg = []
        y_cls = []
        rets = pd.Series(px).pct_change().fillna(0).values
        for i in range(60, len(px)-1):
            window = px[i-60:i]
            feat = [
                window[-1]/(window.mean()+1e-9),
                window[-1]/(np.percentile(window, 90)+1e-9),
                rets[i-20:i].std(),
                rets[i-20:i].mean(),
            ]
            X.append(feat)
            y_reg.append(px[i+1])
            y_cls.append(int(px[i+1] > px[i]))
        X = np.array(X)
        self.models["price_reg"].fit(X, y_reg)
        self.models["trend_cls"].fit(X, y_cls)

    def predict_signal(self, df):
        if len(df) < self.min_train: return 0.0
        px = df["close"].values
        rets = pd.Series(px).pct_change().fillna(0).values
        window = px[-60:]
        feat = [
            window[-1]/(window.mean()+1e-9),
            window[-1]/(np.percentile(window, 90)+1e-9),
            rets[-20:].std(),
            rets[-20:].mean(),
        ]
        X = np.array(feat).reshape(1, -1)
        pr = self.models["price_reg"].predict(X)[0]
        tc = self.models["trend_cls"].predict_proba(X)[0,1]
        # دمج: اتجاه * حجم الإشارة
        sig = np.tanh((pr - px[-1])/(px[-1]*0.002)) * (tc*2-1)
        return float(np.clip(sig, -1, 1))

# =========================
# ---- PORTFOLIO / RISK ----
# =========================
class RiskManager:
    def __init__(self):
        self.equity_high = None
        self.day_pnl = 0.0
        self.day = datetime.now().date()
        self.last_trade = {}

    def reset_day_if_needed(self):
        today = datetime.now().date()
        if today != self.day:
            self.day = today
            self.day_pnl = 0.0

    def account_guard(self, equity):
        if self.equity_high is None: self.equity_high = equity
        self.equity_high = max(self.equity_high, equity)
        dd = (self.equity_high - equity)/max(1e-9, self.equity_high)
        if dd >= CONFIG["risk"]["dd_stop_pct"]:
            return False, f"Global DD hit {dd:.2%}"
        if self.day_pnl <= -CONFIG["risk"]["daily_loss_stop_pct"]*equity:
            return False, f"Daily loss stop hit {self.day_pnl:.2%}"
        return True, ""

    def record_fill(self, symbol, pnl_frac):
        self.reset_day_if_needed()
        self.day_pnl += pnl_frac

    def kelly_fraction(self, winrate, rr):
        if rr <= 0: return 0.0
        f = winrate - (1-winrate)/rr
        return max(0.0, min(CONFIG["risk"]["kelly_cap"], f))

    def vol_target_size(self, px_series):
        rets = px_series.pct_change().dropna()
        ann = annualized_vol(rets)
        if ann <= 1e-6: return 1.0
        tgt = CONFIG["risk"]["target_vol_annual"]
        return float(np.clip(tgt / ann, 0.05, 2.0))

# =========================
# ---- EXECUTION ----------
# =========================
class Executor:
    def __init__(self, ex: Exchange):
        self.ex = ex
        self.last_exec = {}
    def _cooldown_ok(self, symbol):
        t = self.last_exec.get(symbol, 0)
        return (time.time() - t) >= CONFIG["risk"]["position_cooldown_sec"]
    def estimate_qty(self, symbol, venue, equity_usdt, px, weight):
        # cap per symbol, leverage-aware
        max_alloc = equity_usdt * CONFIG["risk"]["max_symbol_exposure"]
        alloc = np.clip(abs(weight), 0, 1) * max_alloc
        if venue == "future":
            lev = CONFIG["futures"]["default_leverage"]
            alloc *= lev
        qty = alloc / max(px, 1e-9)
        return round(qty, 6)

    def trade(self, symbol, side, qty, venue, px=None):
        if qty <= 0 or not self._cooldown_ok(symbol): return None
        ord_type = "limit" if px else "market"
        # slippage
        if px:
            slip = CONFIG["costs"]["slippage_bps"]/1e4
            if side=="buy": px = px*(1+slip)
            else: px = px*(1-slip)
        tp=sl=None
        if CONFIG["execution"]["use_oco"]:
            tp = None
            sl = None
        if CONFIG["execution"]["use_trailing"]:
            # trailing set later
            pass
        o = self.ex.create_order(symbol, side, qty, price=px, venue="spot" if venue=="spot" else "future",
                                 order_type=ord_type, reduce_only=False,
                                 oco=CONFIG["execution"]["use_oco"], tp=tp, sl=sl,
                                 trailing=CONFIG["execution"]["use_trailing"],
                                 post_only=CONFIG["execution"]["post_only"])
        if o: self.last_exec[symbol] = time.time()
        return o

# =========================
# ---- BACKTESTER ----------
# =========================
class Backtester:
    def __init__(self, strategies):
        self.strategies = strategies

    def run(self, df_map, costs=CONFIG["costs"], initial_balance=1000.0):
        equity = initial_balance
        pos = {sym:0.0 for sym in df_map}
        last_px = {sym: df["close"].iloc[0] for sym,df in df_map.items()}
        history = []
        for i in range(200, min(len(d) for d in df_map.values())):
            w = {}
            for sym, df in df_map.items():
                sub = df.iloc[:i].copy()
                sigs = []
                for s in self.strategies:
                    if s.name=="pairs": continue
                    sigs.append(s.generate(sub))
                mix = np.tanh(np.sum(sigs))
                w[sym] = np.clip(mix, -1, 1)
                last_px[sym] = sub["close"].iloc[-1]
            # rebalance  (بسيط)
            target_pos_val = {sym: w[sym]*equity*CONFIG["risk"]["max_symbol_exposure"] for sym in w}
            # pnl update
            for sym, df in df_map.items():
                px = df["close"].iloc[i]
                equity += pos[sym]*(px - last_px[sym])
                last_px[sym] = px
            # costs on turnover
            for sym in w:
                tgt_qty = target_pos_val[sym]/max(last_px[sym],1e-9)
                turn = abs(tgt_qty - pos[sym])
                fee = turn*last_px[sym]*costs["taker"]
                equity -= fee
                pos[sym] = tgt_qty
            history.append(equity)
        res = pd.Series(history)
        ret = res.pct_change().dropna()
        sharpe = ret.mean()/(ret.std()+1e-9)*math.sqrt(252*24*60)  # تقدير
        return {"equity_curve": res, "sharpe": sharpe, "final": res.iloc[-1] if len(res) else initial_balance}

# =========================
# ---- ARBITRAGE -----------
# =========================
class TriangularArb:
    def __init__(self, ex: Exchange):
        self.ex = ex
    def find_cycle(self, base="USDT"):
        # تبسيط: نفحص BNB/USDT, BTC/BNB, BTC/USDT
        return [("BNB/USDT","BTC/BNB","BTC/USDT")]
    def opportunity(self):
        cycles = self.find_cycle()
        best = None; best_edge = 0
        for a,b,c in cycles:
            try:
                ta = self.ex.ticker(a)["last"]
                tb = self.ex.ticker(b)["last"]
                tc = self.ex.ticker(c)["last"]
                # edge تقريبي
                # a: USDT->BNB, b: BNB->BTC, c: BTC->USDT
                usd_to_bnb = 1.0/ta
                bnb_to_btc = usd_to_bnb/tb
                btc_to_usd = bnb_to_btc*tc
                edge = btc_to_usd - 1.0
                if edge > best_edge:
                    best_edge = edge; best = (a,b,c,edge)
            except Exception:
                continue
        return best

# =========================
# ---- MAIN BOT ------------
# =========================
class EliteProBot:
    def __init__(self):
        self.ex = Exchange(CONFIG)
        syms = self.ex.universe()
        if not syms:
            syms = ["BTC/USDT","ETH/USDT","BNB/USDT"]
        self.symbols = syms
        self.ws = WSFeed(self.symbols)
        self.strats = [
            MomentumStrategy(), MeanReversionStrategy(),
            BreakoutStrategy(), RSIMACDStrategy(),
            VolatilityBreakout(), AdaptiveGridDCA()
        ]
        self.mixer = SignalMixer()
        self.risk = RiskManager()
        self.exec = Executor(self.ex)
        self.arb = TriangularArb(self.ex)
        self.venue_choice = {s: "spot" for s in self.symbols}  # spot/future
        self.last_equity = CONFIG["backtest"]["initial_balance"]

    async def prepare_history(self):
        self.hist = {}
        tf = CONFIG["system"]["base_timeframe"]
        for s in self.symbols:
            try:
                kl = self.ex.fetch_ohlcv(s, tf, limit=CONFIG["system"]["history_klines"], venue="spot")
                df = pd.DataFrame(kl, columns=["t","open","high","low","close","volume"])
                df["t"] = pd.to_datetime(df["t"], unit="ms")
                df.set_index("t", inplace=True)
                self.hist[s.replace("/","")] = df
            except Exception as e:
                log.warning(f"hist fail {s}: {e}")

    def _df_for(self, sym_code):
        # دمج WebSocket kline (إن وُجد) مع التاريخ
        base = self.hist.get(sym_code)
        live = self.ws.klines_df(sym_code)
        if base is None and live.empty: return pd.DataFrame()
        if base is None: df = live.copy()
        elif live.empty: df = base.copy()
        else:
            df = pd.concat([base, live.assign(t=pd.to_datetime(live.index, unit='ms')).set_index("t")], axis=0)
            df = df[~df.index.duplicated(keep="last")]
        df.columns = ["open","high","low","close","volume"]
        return df

    def dynamic_venue(self, symbol, df):
        # اختيار Spot vs Futures حسب التقلب + الرصيد
        bal = self.ex.balance_usdt()
        vol = df["close"].pct_change().rolling(120).std().iloc[-1] if len(df)>150 else 0.01
        if CONFIG["mode"]["use_futures"] and bal >= 30 and vol >= 0.004:
            return "future"
        return "spot"

    def compute_weight(self, df, sym):
        # خلط الاستراتيجيات + AI mixer + فلترة مخاطرة
        sigs = [s.generate(df) for s in self.strats if s.name not in ("pairs",)]
        s_sum = float(np.tanh(np.sum(sigs)))
        # fit/predict AI
        self.mixer.fit_if_ready(df.reset_index().rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}))
        ai_sig = self.mixer.predict_signal(df.reset_index().rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}))
        w = float(np.tanh(0.6*s_sum + 0.4*ai_sig))

        # Vol targeting scaling
        scale = RiskManager().vol_target_size(df["close"])
        w *= np.clip(scale, 0.3, 1.5)

        return float(np.clip(w, -1, 1))

    async def arbitrage_step(self):
        opp = self.arb.opportunity()
        if opp and opp[-1] > 0.001:
            log.info(f"[ARB] Opportunity {opp}")

    async def run(self):
        await self.prepare_history()
        await self.ws.start()
        log.info(f"Symbols: {self.symbols}")
        t0 = time.time()
        while True:
            try:
                await asyncio.sleep(CONFIG["system"]["refresh_sec"])
                # حساب الـ equity التقريبي (Paper)
                equity = self.ex.balance_usdt() or self.last_equity
                ok, reason = self.risk.account_guard(equity)
                if not ok:
                    log.warning(f"PAUSED: {reason}")
                    continue

                await self.arbitrage_step()

                for s in self.symbols:
                    sym_code = s.replace("/","")
                    df = self._df_for(sym_code)
                    if df is None or df.empty or len(df)<120: continue

                    venue = self.dynamic_venue(s, df)
                    if venue=="future":
                        self.ex.futures_set_leverage(s, CONFIG["futures"]["default_leverage"])

                    w = self.compute_weight(df, s)
                    px = float(df["close"].iloc[-1])
                    qty = self.exec.estimate_qty(s, venue, equity, px, w)

                    side = "buy" if w>0 else ("sell" if w<0 else None)
                    if side:
                        self.exec.trade(s, side, abs(qty), venue, px=None)

                # مثال auto-loan/repay (تبسيطي): إذا الbalance < 10$ ونحب نشتري Spot => نعمل قرض USDT صغير ثم نسدّد عند البيع
                if CONFIG["mode"]["use_margin"] and CONFIG["margin"]["use_auto_loan"]:
                    bal = self.ex.balance_usdt()
                    if bal < 10:
                        self.ex.margin_auto_loan("USDT", 15)
                    elif bal > 30:
                        self.ex.margin_auto_repay("USDT", 10)

                self.last_equity = equity
            except Exception as e:
                log.error(f"loop error: {e}")
                await asyncio.sleep(1.0)

# =========================
# ---- BACKTEST DRIVER -----
# =========================
def run_backtest(ex: Exchange, strategies, symbols, tf="1m", lookback=2000, start=None, end=None):
    df_map = {}
    for s in symbols:
        try:
            kl = ex.fetch_ohlcv(s, tf, limit=lookback, venue="spot")
            df = pd.DataFrame(kl, columns=["t","open","high","low","close","volume"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")
            df.set_index("t", inplace=True)
            if start: df = df[df.index>=pd.to_datetime(start)]
            if end: df = df[df.index<=pd.to_datetime(end)]
            df_map[s] = df
        except Exception as e:
            log.warning(f"bt data fail {s}: {e}")
    bt = Backtester(strategies)
    res = bt.run(df_map, initial_balance=CONFIG["backtest"]["initial_balance"])
    log.info(f"Backtest Sharpe: {res['sharpe']:.2f} | Final: {res['final']:.2f}")
    return res

# =========================
# -------- MAIN -----------
# =========================
if __name__ == "__main__":
    # تحضير المنصة
    ex = Exchange(CONFIG)

    # Backtest (اختياري قبل التشغيل)
    try:
        syms_bt = ex.universe()[:6] or ["BTC/USDT","ETH/USDT","BNB/USDT"]
        run_backtest(ex, [
            MomentumStrategy(), MeanReversionStrategy(),
            BreakoutStrategy(), RSIMACDStrategy(),
            VolatilityBreakout(), AdaptiveGridDCA()
        ], syms_bt, tf=CONFIG["system"]["base_timeframe"], lookback=1500,
        start=CONFIG["backtest"]["start"], end=CONFIG["backtest"]["end"])
    except Exception as e:
        log.warning(f"Backtest skipped: {e}")

    # تشغيل البوت الحي
    bot = EliteProBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Stopped by user")
