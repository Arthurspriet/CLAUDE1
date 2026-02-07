"""Trader tools — LLM-facing tools that wrap the trader module.

These tools provide the agent with abilities to:
- Scan markets for trading signals (momentum, volume, spread, yield)
- Analyze and approve trades through risk management
- Open and manage positions with stop-losses
- Monitor positions and generate trading performance reports

All decisions are logged to DecisionLogger (Google Sheet + local JSON).

Signal Cache
~~~~~~~~~~~~
A module-level cache stores the latest signals from ``quant_analyze`` so
that downstream tools (``analyze_trade``, ``open_position``) can auto-
look them up by token name.  This eliminates the need for the LLM to
manually extract and re-pass entry/target/stop prices between tool calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from claude1.tools.base import BaseTool
from claude1.spending.decision_logger import DecisionLogger

if TYPE_CHECKING:
    from claude1.trader.strategy import TradingStrategy, TradeDecision
    from claude1.trader.signals import Signal, SignalScanner
    from claude1.trader.positions import PositionManager
    from claude1.trader.risk_manager import RiskManager
    from claude1.wallet import SolanaWallet

logger = logging.getLogger("claude1.tools.trader_tools")

# ── Signal cache ─────────────────────────────────────────────────────────────
# Keyed by token symbol (lowercase).  Each entry is the *best* approved Signal
# from the latest quant_analyze run.  Downstream tools can pull from here when
# the LLM doesn't (or can't) pass the exact numeric values.

_signal_cache: dict[str, "Signal"] = {}
_decision_cache: dict[str, "TradeDecision"] = {}

# ── Near-miss cache ─────────────────────────────────────────────────────────
# Stores rejected signals that were *close* to approval (R:R >= 1.0 or
# confidence >= 0.4).  The agent can use these for limit orders at better
# entry prices or smart-swap fallback decisions.  Keyed by token symbol (lowercase).
_near_miss_cache: dict[str, tuple["Signal", "TradeDecision"]] = {}


def _cache_signal(signal: "Signal", decision: "TradeDecision | None" = None) -> None:
    """Store the best signal per token in the module-level cache."""
    key = signal.token.lower().strip()
    existing = _signal_cache.get(key)
    # Keep the signal with higher confidence × R:R
    if existing is None or (
        signal.confidence * max(signal.risk_reward_ratio, 0.1)
        > existing.confidence * max(existing.risk_reward_ratio, 0.1)
    ):
        _signal_cache[key] = signal
        if decision is not None:
            _decision_cache[key] = decision


def _cache_near_miss(signal: "Signal", decision: "TradeDecision") -> None:
    """Store a rejected-but-close signal for alternative actions (limit orders, smart swap)."""
    key = signal.token.lower().strip()
    rr = signal.risk_reward_ratio
    conf = signal.confidence
    # Only cache if reasonably close to approval
    if rr >= 0.8 or conf >= 0.4:
        existing = _near_miss_cache.get(key)
        if existing is None or (
            conf * max(rr, 0.1) > existing[0].confidence * max(existing[0].risk_reward_ratio, 0.1)
        ):
            _near_miss_cache[key] = (signal, decision)


def _get_cached_signal(token: str) -> "Signal | None":
    """Look up the best cached signal for a token.

    Falls back to the near-miss cache so that ``analyze_trade`` can still
    reference prices from rejected-but-close signals (useful for limit orders).
    """
    key = token.lower().strip()
    sig = _signal_cache.get(key)
    if sig is not None:
        return sig
    # Fallback: near-miss cache
    near = _near_miss_cache.get(key)
    return near[0] if near else None


def _get_cached_decision(token: str) -> "TradeDecision | None":
    """Look up the cached trade decision for a token."""
    return _decision_cache.get(token.lower().strip())


def _get_near_misses() -> list[tuple[str, "Signal", "TradeDecision"]]:
    """Return all near-miss signals sorted by composite score (best first)."""
    items = [
        (tok, sig, dec) for tok, (sig, dec) in _near_miss_cache.items()
    ]
    items.sort(
        key=lambda x: x[1].confidence * max(x[1].risk_reward_ratio, 0.1),
        reverse=True,
    )
    return items


def _clear_cache() -> None:
    """Clear the signal cache (e.g. between scan cycles)."""
    _signal_cache.clear()
    _decision_cache.clear()
    _near_miss_cache.clear()


def _decision_logger() -> DecisionLogger:
    """Get the singleton DecisionLogger instance."""
    return DecisionLogger.get_instance()


class MarketScanTool(BaseTool):
    """Scan markets for trading signals across multiple signal types."""

    def __init__(
        self,
        working_dir: str,
        scanner: "SignalScanner",
        strategy: "TradingStrategy",
    ) -> None:
        super().__init__(working_dir)
        self._scanner = scanner
        self._strategy = strategy

    @property
    def name(self) -> str:
        return "market_scan"

    @property
    def description(self) -> str:
        return (
            "Scan markets for trading signals. Detects momentum, mean-reversion, "
            "volume spikes, spread opportunities, and yield. Provide price data "
            "and/or lending rates to analyze. Returns ranked signals with trade decisions."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "signal_type": {
                    "type": "string",
                    "enum": ["price_action", "yield", "spread", "all"],
                    "description": "Type of signals to scan for (default: all)",
                },
                "price_data": {
                    "type": "object",
                    "description": (
                        "Price data: {token: {price, change_1h, change_24h, volume_24h, avg_volume}}. "
                        "Get this from token_price or market APIs."
                    ),
                },
                "lending_data": {
                    "type": "object",
                    "description": "Lending rates: {token: {apy, protocol}} or {token: apy_number}",
                },
                "spread_data": {
                    "type": "object",
                    "description": "Spread data: {token: {dex_price, cex_price, spread_pct}}",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max signals to return (default: 10)",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        signal_type = kwargs.get("signal_type", "all")
        price_data = kwargs.get("price_data", {})
        lending_data = kwargs.get("lending_data", {})
        spread_data = kwargs.get("spread_data", {})
        limit = kwargs.get("limit", 10)

        try:
            # Auto-fetch price data from Jupiter if none provided
            if not price_data and signal_type in ("price_action", "all"):
                price_data = self._auto_fetch_prices()

            signals = []

            if signal_type == "all" and price_data:
                # Use multi-token, multi-strategy scanning
                signals.extend(self._scanner.scan_multi_token(price_data))
            elif signal_type == "price_action" and price_data:
                signals.extend(self._scanner.scan_price_action(price_data))

            if signal_type in ("yield", "all") and lending_data:
                signals.extend(self._scanner.scan_yield_opportunities(lending_data))

            if signal_type in ("spread", "all") and spread_data:
                signals.extend(self._scanner.scan_spread_opportunities(spread_data))

            ranked = self._scanner.get_ranked(limit=limit)

            if not ranked:
                # Even with no signals, return useful market data + guidance
                if price_data:
                    lines = ["📊 Market scan complete — no strong signals from classic scanner.\n"]
                    lines.append(f"Scanned {len(price_data)} tokens:")
                    for token, data in sorted(price_data.items()):
                        if isinstance(data, dict):
                            price = data.get("price", "?")
                            c24h = data.get("change_24h", 0)
                            emoji = "🟢" if c24h > 2 else "🔴" if c24h < -2 else "⚪"
                            lines.append(
                                f"  {emoji} {token.upper():>8}: ${price:>12,.6f} ({c24h:+.2f}% 24h)"
                            )
                    lines.append("")
                    lines.append("💡 NEXT STEP: Call quant_analyze to run ALL strategies (momentum, ")
                    lines.append("   mean-reversion, breakout, scalping, correlation) across all tokens.")
                    lines.append("   Scalping works in flat markets. Correlation detects altcoin divergence.")
                    return "\n".join(lines)
                return (
                    "Could not fetch price data from Jupiter API.\n"
                    "💡 Try: quant_analyze (auto-fetches prices and runs all strategies)"
                )

            lines = [f"🔍 Found {len(ranked)} trading signals:\n"]
            approved_count = 0
            for i, signal in enumerate(ranked, 1):
                decision = self._strategy.evaluate_signal(signal)
                trade_str = "✅ TRADE" if decision.approved else "⛔ SKIP"
                lines.append(f"{i}. [{trade_str}] {signal.summary()}")
                if decision.approved:
                    approved_count += 1
                    lines.append(
                        f"   → Size: {decision.suggested_amount:.4f} "
                        f"(${decision.suggested_cost:.2f})"
                    )
                    # Log each approved trade decision
                    _decision_logger().log_trade_decision(
                        token=signal.token,
                        direction=signal.direction,
                        entry_price=signal.current_price,
                        amount=decision.suggested_amount,
                        cost=decision.suggested_cost,
                        approved=True,
                        confidence=signal.confidence,
                        risk_reward=0,
                        signal_type=signal.source,
                        target_price=signal.target_price,
                        stop_price=signal.stop_price,
                    )
                elif not decision.approved and decision.reason:
                    lines.append(f"   → {decision.reason}")
                    # Log rejected decisions too
                    _decision_logger().log_trade_decision(
                        token=signal.token,
                        direction=signal.direction,
                        entry_price=signal.current_price,
                        approved=False,
                        reason=decision.reason,
                        confidence=signal.confidence,
                        signal_type=signal.source,
                        target_price=signal.target_price,
                        stop_price=signal.stop_price,
                    )

            # Log scan summary
            _decision_logger().log_market_scan(
                signals_found=len(ranked),
                approved_count=approved_count,
                tokens_scanned=list(price_data.keys()) if price_data else [],
                scan_type=signal_type,
            )

            return "\n".join(lines)

        except Exception as exc:
            return f"Error scanning market: {exc}"

    # ── Birdeye OHLCV cache (avoid re-fetching within 60s) ──────────────
    _ohlcv_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    _OHLCV_CACHE_TTL = 60.0  # seconds

    # ── Trending tokens cache ─────────────────────────────────────────
    _trending_cache: tuple[float, dict[str, str]] | None = None
    _TRENDING_CACHE_TTL = 120.0  # seconds

    @staticmethod
    def _fetch_trending_tokens() -> dict[str, str]:
        """Fetch trending Solana tokens from Birdeye and merge with static SCAN_TOKENS.

        Returns a ``{symbol: mint_address}`` dict that is a superset of
        the static ``SCAN_TOKENS`` list.  Birdeye failures are gracefully
        handled — the static list is always returned as a baseline.
        """
        import time
        import httpx
        from claude1.config import (
            BIRDEYE_API_KEY, BIRDEYE_BASE_URL, BIRDEYE_HTTP_TIMEOUT, SCAN_TOKENS,
        )

        # Check cache first
        if MarketScanTool._trending_cache is not None:
            ts, cached = MarketScanTool._trending_cache
            if time.time() - ts < MarketScanTool._TRENDING_CACHE_TTL:
                return cached

        # Always start with static tokens
        merged: dict[str, str] = dict(SCAN_TOKENS)

        if not BIRDEYE_API_KEY:
            return merged

        try:
            headers = {
                "X-API-KEY": BIRDEYE_API_KEY,
                "x-chain": "solana",
                "accept": "application/json",
            }
            resp = httpx.get(
                f"{BIRDEYE_BASE_URL}/defi/token_trending",
                headers=headers,
                params={"sort_by": "rank", "sort_type": "asc", "offset": 0, "limit": 20},
                timeout=BIRDEYE_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", {}).get("items", []) if isinstance(data, dict) else []

            # Build a reverse lookup of known mints → symbols
            known_mints = {v: k for k, v in SCAN_TOKENS.items()}

            for item in items:
                if not isinstance(item, dict):
                    continue
                mint = item.get("address", "")
                symbol = (item.get("symbol") or "").lower().strip()
                if not mint or not symbol:
                    continue
                # Skip stablecoins and wrapped tokens
                if symbol in ("usdc", "usdt", "dai", "wsol"):
                    continue
                # Use the known symbol if we already track this mint
                if mint in known_mints:
                    symbol = known_mints[mint]
                # Add to merged (won't overwrite existing entries)
                if symbol not in merged:
                    merged[symbol] = mint

            logger.info(
                "Trending tokens: %d from Birdeye + %d static = %d total",
                max(0, len(merged) - len(SCAN_TOKENS)), len(SCAN_TOKENS), len(merged),
            )
        except Exception as exc:
            logger.debug("Birdeye trending fetch failed: %s", exc)

        # Update cache
        import time as _t
        MarketScanTool._trending_cache = (_t.time(), merged)
        return merged

    @staticmethod
    def _fetch_birdeye_ohlcv(mint: str, symbol: str) -> dict[str, float] | None:
        """Fetch real 1h and 4h OHLCV data from Birdeye for a single token.

        Returns ``{change_1h, change_4h}`` computed from actual candle
        open/close prices, or ``None`` if Birdeye is unavailable.
        Results are cached for 60 seconds.
        """
        import time
        import httpx
        from claude1.config import BIRDEYE_API_KEY, BIRDEYE_BASE_URL, BIRDEYE_HTTP_TIMEOUT

        if not BIRDEYE_API_KEY:
            return None

        cache_key = mint
        now = time.time()

        # Check cache
        if cache_key in MarketScanTool._ohlcv_cache:
            ts, cached_data = MarketScanTool._ohlcv_cache[cache_key]
            if now - ts < MarketScanTool._OHLCV_CACHE_TTL:
                return cached_data

        headers = {
            "X-API-KEY": BIRDEYE_API_KEY,
            "x-chain": "solana",
            "accept": "application/json",
        }

        result: dict[str, float] = {}

        try:
            # Fetch 1h candles (last 2 candles to compute 1h change)
            time_to = int(now)
            time_from_1h = time_to - 2 * 3600  # 2h back for at least 2 candles

            resp = httpx.get(
                f"{BIRDEYE_BASE_URL}/defi/ohlcv",
                headers=headers,
                params={
                    "address": mint,
                    "type": "1H",
                    "time_from": time_from_1h,
                    "time_to": time_to,
                },
                timeout=BIRDEYE_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            items_1h = data.get("data", {}).get("items", []) if isinstance(data, dict) else []

            if len(items_1h) >= 2:
                # Compute change from the open of the previous candle to the close of the latest
                prev_open = float(items_1h[-2].get("o", 0))
                latest_close = float(items_1h[-1].get("c", 0))
                if prev_open > 0:
                    result["change_1h"] = ((latest_close - prev_open) / prev_open) * 100
            elif len(items_1h) == 1:
                candle = items_1h[0]
                o = float(candle.get("o", 0))
                c = float(candle.get("c", 0))
                if o > 0:
                    result["change_1h"] = ((c - o) / o) * 100

            # Fetch 4h candles (last 2 candles)
            time_from_4h = time_to - 8 * 3600

            resp4 = httpx.get(
                f"{BIRDEYE_BASE_URL}/defi/ohlcv",
                headers=headers,
                params={
                    "address": mint,
                    "type": "4H",
                    "time_from": time_from_4h,
                    "time_to": time_to,
                },
                timeout=BIRDEYE_HTTP_TIMEOUT,
            )
            resp4.raise_for_status()
            data4 = resp4.json()
            items_4h = data4.get("data", {}).get("items", []) if isinstance(data4, dict) else []

            if len(items_4h) >= 2:
                prev_open = float(items_4h[-2].get("o", 0))
                latest_close = float(items_4h[-1].get("c", 0))
                if prev_open > 0:
                    result["change_4h"] = ((latest_close - prev_open) / prev_open) * 100
            elif len(items_4h) == 1:
                candle = items_4h[0]
                o = float(candle.get("o", 0))
                c = float(candle.get("c", 0))
                if o > 0:
                    result["change_4h"] = ((c - o) / o) * 100

        except Exception as exc:
            logger.debug("Birdeye OHLCV failed for %s (%s): %s", symbol, mint[:8], exc)
            # Cache the failure too (to avoid hammering)
            MarketScanTool._ohlcv_cache[cache_key] = (now, result if result else {})
            return result if result else None

        # Cache the result
        MarketScanTool._ohlcv_cache[cache_key] = (now, result)
        return result if result else None

    @staticmethod
    def _auto_fetch_prices() -> dict[str, Any]:
        """Fetch live price data for all scanned + trending Solana tokens.

        Returns data in the format expected by SignalScanner.scan_price_action().
        Scans the full SCAN_TOKENS list plus dynamically discovered trending
        tokens from Birdeye.

        Enriches price data with real 1h/4h candle changes from Birdeye OHLCV
        when available, falling back to the ``change_24h * 0.4`` estimation
        when Birdeye is not configured or the request fails.
        """
        import httpx
        from claude1.config import JUP_API_KEY, JUP_BASE_URL, JUP_HTTP_TIMEOUT

        # ── Step 1: Get full token universe (static + trending) ───────
        all_tokens = MarketScanTool._fetch_trending_tokens()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if JUP_API_KEY:
            headers["x-api-key"] = JUP_API_KEY

        price_data: dict[str, Any] = {}

        try:
            # Jupiter allows up to 100 mints per request
            ids_param = ",".join(all_tokens.values())
            resp = httpx.get(
                f"{JUP_BASE_URL}/price/v3",
                headers=headers,
                params={"ids": ids_param, "showExtraInfo": "true"},
                timeout=JUP_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            prices = data.get("data", data) if isinstance(data, dict) else {}

            for symbol, mint in all_tokens.items():
                info = prices.get(mint, {})
                if isinstance(info, dict):
                    usd_price = info.get("usdPrice", info.get("price", 0))
                    change_24h = float(info.get("priceChange24h", 0) or 0)

                    # Extract volume from extraInfo if available
                    extra = info.get("extraInfo", {}) or {}
                    volume_24h = 0.0
                    avg_volume = 0.0

                    if usd_price:
                        # Default: estimate 1h from 24h
                        estimated_1h = change_24h * 0.4

                        price_data[symbol] = {
                            "price": float(usd_price),
                            "change_1h": estimated_1h,
                            "change_24h": change_24h,
                            "volume_24h": volume_24h,
                            "avg_volume": avg_volume,
                        }

            # ── Step 2: Enrich with real Birdeye OHLCV data ──────────
            for symbol, mint in all_tokens.items():
                if symbol not in price_data:
                    continue
                ohlcv = MarketScanTool._fetch_birdeye_ohlcv(mint, symbol)
                if ohlcv:
                    if "change_1h" in ohlcv:
                        price_data[symbol]["change_1h"] = ohlcv["change_1h"]
                    if "change_4h" in ohlcv:
                        price_data[symbol]["change_4h"] = ohlcv["change_4h"]

        except Exception:
            pass  # Silently fail — will return empty dict

        return price_data


class QuantAnalyzeTool(BaseTool):
    """Run all trading strategies across all tokens and return ranked setups.

    This tool does the quantitative computation so the LLM does not have
    to guess entry/target/stop prices.  It runs momentum, mean-reversion,
    breakout, scalping, and correlation strategies across all scanned
    Solana ecosystem tokens and returns structured trade setups.

    Pre-filters signals through the TradingStrategy risk checks so only
    actionable, approved setups are shown.  Approved signals are cached
    so that subsequent analyze_trade / open_position calls can auto-lookup
    prices by token name alone.
    """

    def __init__(
        self,
        working_dir: str,
        strategy: "TradingStrategy" | None = None,
        risk_manager: "RiskManager" | None = None,
        wallet: "Any | None" = None,
    ) -> None:
        super().__init__(working_dir)
        self._strategy = strategy
        self._risk = risk_manager
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "quant_analyze"

    @property
    def description(self) -> str:
        return (
            "Run all quantitative trading strategies (momentum, micro-momentum, mean-reversion, "
            "breakout, scalping, correlation) across all Solana ecosystem tokens PLUS "
            "dynamically discovered trending tokens from Birdeye. Returns ONLY "
            "risk-approved trade setups with pre-computed entry, target, stop, and R:R. "
            "Uses real 1h/4h candle data from Birdeye when available for better signals. "
            "Position sizes are capped to actual wallet balance. "
            "Approved setups are cached — after calling this, just use "
            "analyze_trade(token='sol') to proceed (prices are auto-filled)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "price_data": {
                    "type": "object",
                    "description": (
                        "Optional price data: {token: {price, change_1h, change_24h, volume_24h}}. "
                        "If omitted, auto-fetches from Jupiter API for all scanned tokens."
                    ),
                },
                "strategies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional: list of strategy names to run (e.g. ['momentum', 'scalping']). "
                        "If omitted, runs all enabled strategies."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max setups to return (default: 10)",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            from claude1.trader.strategies import StrategyRegistry

            price_data = kwargs.get("price_data", {})
            strategy_filter = kwargs.get("strategies", [])
            limit = int(kwargs.get("limit", 10))

            # Auto-fetch if no data provided
            if not price_data:
                price_data = MarketScanTool._auto_fetch_prices()

            if not price_data:
                return "❌ Could not fetch price data from Jupiter. Try providing price_data manually."

            # ── Fetch wallet balance for realistic sizing ─────────────────
            available_balance_usd: float | None = None
            balance_info = ""
            if self._wallet:
                try:
                    from claude1.tools.wallet_tools import get_wallet_balance_usd, get_wallet_balance
                    available_balance_usd = get_wallet_balance_usd(self._wallet.public_key)
                    balances = get_wallet_balance(self._wallet.public_key)
                    sol_bal = balances.get("sol", 0.0)
                    usdc_bal = balances.get("usdc", 0.0)
                    balance_info = (
                        f"\n💳 Wallet: {sol_bal:.4f} SOL + {usdc_bal:.2f} USDC "
                        f"(~${available_balance_usd:.2f} total)"
                    )
                except Exception:
                    pass  # Best-effort

            # Build the registry and optionally filter strategies
            registry = StrategyRegistry()
            if strategy_filter:
                for name in list(registry._strategies.keys()):
                    if name not in strategy_filter:
                        registry.disable(name)

            signals = registry.scan_all(price_data)

            if not signals:
                # Still return market summary
                lines = ["📊 Quant analysis complete — no actionable setups found.\n"]
                if balance_info:
                    lines.append(balance_info)
                lines.append("Current market state:")
                for token, data in sorted(price_data.items()):
                    if isinstance(data, dict):
                        p = data.get("price", 0)
                        c1h = data.get("change_1h", 0)
                        c24h = data.get("change_24h", 0)
                        lines.append(f"  {token.upper():>8}: ${p:>12,.6f}  (1h: {c1h:+.2f}%, 24h: {c24h:+.2f}%)")
                lines.append(f"\nTokens scanned: {len(price_data)}")
                lines.append("Strategies run: " + ", ".join(
                    s.name for s in registry._strategies.values() if s.config.enabled
                ))
                lines.append("\nMarket may be too flat or no tokens meet strategy thresholds.")
                return "\n".join(lines)

            # ── Pre-filter through risk management and cache approved ─────
            # Clear stale cache entries before repopulating
            _clear_cache()

            approved_signals: list = []
            rejected_count = 0

            for sig in signals:
                if self._strategy:
                    # Pass available_balance_usd so sizing is realistic
                    if available_balance_usd is not None and self._risk:
                        # Re-compute position size capped to balance
                        capped_amount = self._risk.get_position_size(
                            entry_price=sig.current_price,
                            stop_price=sig.stop_price,
                            available_balance_usd=available_balance_usd,
                        )
                        # Create a modified signal evaluation by passing portfolio_value
                        decision = self._strategy.evaluate_signal(sig)
                        # Override suggested amount/cost with balance-capped values
                        if decision.approved and capped_amount > 0:
                            decision.suggested_amount = capped_amount
                            decision.suggested_cost = capped_amount * sig.current_price
                    else:
                        decision = self._strategy.evaluate_signal(sig)

                    if decision.approved:
                        _cache_signal(sig, decision)
                        approved_signals.append((sig, decision))
                    else:
                        rejected_count += 1
                        # Cache near-miss signals for alternative actions
                        _cache_near_miss(sig, decision)
                else:
                    # No strategy provided — show all, still cache
                    _cache_signal(sig)
                    approved_signals.append((sig, None))

            if not approved_signals:
                lines = [
                    f"📊 Quant analysis complete — {len(signals)} raw setups found "
                    f"but ALL were rejected by risk management.\n"
                ]
                if balance_info:
                    lines.append(balance_info)
                lines.append("Current market state:")
                for token, data in sorted(price_data.items()):
                    if isinstance(data, dict):
                        p = data.get("price", 0)
                        c24h = data.get("change_24h", 0)
                        emoji = "🟢" if c24h > 2 else "🔴" if c24h < -2 else "⚪"
                        lines.append(f"  {emoji} {token.upper():>8}: ${p:>12,.6f} ({c24h:+.2f}% 24h)")
                lines.append(f"\nRejected: {rejected_count} | Tokens: {len(price_data)}")

                # ── Near-miss signals: show the best rejected setups ─────
                near_misses = _get_near_misses()[:3]
                if near_misses:
                    lines.append("\n🔎 NEAR-MISS SETUPS (closest to approval):")
                    for i, (tok, sig, dec) in enumerate(near_misses, 1):
                        rr = sig.risk_reward_ratio
                        reason = dec.reason.replace("Risk check failed: ", "") if dec.reason else "unknown"
                        # Compute the entry price that would meet min R:R
                        min_rr = 1.5
                        if sig.direction == "long" and sig.target_price > 0 and sig.stop_price > 0:
                            needed_entry = (sig.target_price + min_rr * sig.stop_price) / (1 + min_rr)
                            entry_hint = f"Limit buy at ${needed_entry:,.6f} would meet {min_rr}:1 R:R"
                        elif sig.direction == "short" and sig.target_price > 0 and sig.stop_price > 0:
                            needed_entry = (sig.target_price + min_rr * sig.stop_price) / (1 + min_rr)
                            entry_hint = f"Limit sell at ${needed_entry:,.6f} would meet {min_rr}:1 R:R"
                        else:
                            entry_hint = ""
                        lines.append(
                            f"  {i}. {sig.direction.upper()} {sig.token.upper()} — "
                            f"R:R={rr:.2f} | Conf={sig.confidence:.0%} | ❌ {reason}"
                        )
                        if entry_hint:
                            lines.append(f"     💡 {entry_hint}")

                # ── Actionable alternatives ──────────────────────────────
                lines.append("\n═══ ALTERNATIVES (do NOT repeat quant_analyze) ═══")

                # 1. Limit orders from near-misses
                if near_misses:
                    best_tok = near_misses[0][1].token.upper()
                    lines.append(
                        f"1. 📋 LIMIT ORDER: Use create_limit_order for {best_tok} at the suggested entry above."
                    )
                else:
                    lines.append("1. 📋 LIMIT ORDER: No near-miss tokens to target.")

                # 2. Smart swap into strongest momentum token
                momentum_tokens = [
                    (tok, d.get("change_24h", 0))
                    for tok, d in price_data.items()
                    if isinstance(d, dict) and d.get("change_24h", 0) > 2
                ]
                momentum_tokens.sort(key=lambda x: x[1], reverse=True)
                if momentum_tokens:
                    best_m, best_chg = momentum_tokens[0]
                    lines.append(
                        f"2. 🔄 SMART SWAP: {best_m.upper()} has +{best_chg:.1f}% momentum — "
                        f"if the user explicitly asked to trade, use swap_tokens to buy "
                        f"{best_m.upper()} directly (check with token_shield first, cap to requested amount)."
                    )
                else:
                    lines.append("2. 🔄 SMART SWAP: No tokens with >2% momentum for a direct swap right now.")

                # 3. Position management
                lines.append(
                    "3. 📊 POSITIONS: Call position_status to check/manage existing positions "
                    "(trail stops, take partial profits)."
                )

                # 4. Price alerts
                lines.append(
                    "4. 🔔 ALERTS: Use scheduler(action='start_watch') to monitor prices "
                    "and get notified when conditions improve."
                )

                lines.append(
                    "\n⚠️ Do NOT call quant_analyze again — market data hasn't changed. "
                    "Pick an alternative above or report the market state to the user."
                )
                return "\n".join(lines)

            # Build ranked output — only approved setups
            top = approved_signals[:limit]
            lines = [
                f"🔬 Quant Analysis — {len(approved_signals)} APPROVED setups "
                f"(from {len(signals)} raw, {rejected_count} rejected):\n"
            ]
            if balance_info:
                lines.append(balance_info)
                lines.append("")

            def _fmt_price(p: float) -> str:
                """Format price with appropriate precision for display."""
                if p <= 0:
                    return "$0"
                import math as _math
                mag = _math.floor(_math.log10(abs(p)))
                decimals = max(2, -mag + 3)
                return f"${p:,.{decimals}f}"

            for i, (sig, decision) in enumerate(top, 1):
                rr = sig.risk_reward_ratio
                ret_pct = sig.expected_return_pct
                strategy_tag = next(
                    (t.split(":")[1] for t in sig.tags if t.startswith("strategy:")),
                    sig.source,
                )
                size_info = ""
                if decision and decision.suggested_amount > 0:
                    size_info = (
                        f"\n   💰 Size: {decision.suggested_amount:.4f} tokens "
                        f"(${decision.suggested_cost:.2f})"
                    )
                lines.append(
                    f"{i}. ✅ [{strategy_tag.upper()}] {sig.direction.upper()} {sig.token.upper()}\n"
                    f"   📈 Entry: {_fmt_price(sig.current_price)} → Target: {_fmt_price(sig.target_price)} "
                    f"| Stop: {_fmt_price(sig.stop_price)}\n"
                    f"   📊 R:R={rr:.2f} | Return={ret_pct:+.2f}% | Conf={sig.confidence:.0%} "
                    f"| Strength={sig.strength}"
                    f"{size_info}\n"
                    f"   📝 {sig.title}"
                )

            lines.append(f"\nTokens scanned: {len(price_data)} | "
                         f"Strategies: {', '.join(s.name for s in registry._strategies.values() if s.config.enabled)}")
            lines.append(
                "\n💡 NEXT STEP: Pick a token and call analyze_trade(token='TOKEN_NAME').\n"
                "   Prices are auto-filled from the cache — no need to pass entry/target/stop."
            )
            return "\n".join(lines)

        except Exception as exc:
            return f"Error in quant analysis: {exc}"


class AnalyzeTradeTool(BaseTool):
    """Analyze a potential trade with full risk assessment.

    Includes a per-token rejection counter to prevent the LLM from
    retrying the same failing trade with worse parameters.
    """

    # Class-level rejection counter: {token: rejection_count}
    _rejection_counts: dict[str, int] = {}
    _MAX_REJECTIONS_PER_TOKEN = 3

    def __init__(self, working_dir: str, strategy: "TradingStrategy",
                 risk_manager: "RiskManager") -> None:
        super().__init__(working_dir)
        self._strategy = strategy
        self._risk = risk_manager

    @property
    def name(self) -> str:
        return "analyze_trade"

    @property
    def description(self) -> str:
        return (
            "Analyze a potential trade before opening. Runs risk checks, "
            "calculates position size, and provides a trade decision with "
            "detailed reasoning."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "Token to trade (symbol, e.g. 'sol', 'bonk'). Prices are auto-filled from quant_analyze cache.",
                },
                "side": {
                    "type": "string",
                    "enum": ["long", "short"],
                    "description": "Trade direction (default: auto from cached signal, or 'long')",
                },
                "entry_price": {
                    "type": "number",
                    "description": "Optional: override entry price (auto-filled from cache if omitted)",
                },
                "target_price": {
                    "type": "number",
                    "description": "Optional: override target price (auto-filled from cache if omitted)",
                },
                "stop_price": {
                    "type": "number",
                    "description": "Optional: override stop price (auto-filled from cache if omitted)",
                },
                "amount": {
                    "type": "number",
                    "description": "Optional: specific amount of token. If omitted, uses risk-based sizing.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Your confidence in this trade (0.0-1.0, default from cache or 0.5)",
                },
            },
            "required": ["token"],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            from claude1.trader.signals import Signal

            token = kwargs.get("token", "")
            side = kwargs.get("side", "")

            # Auto-lookup from signal cache (populated by quant_analyze)
            cached = _get_cached_signal(token)

            entry = float(kwargs.get("entry_price") or kwargs.get("entry") or 0)
            target = float(kwargs.get("target_price") or kwargs.get("target") or 0)
            stop = float(kwargs.get("stop_price") or kwargs.get("stop") or kwargs.get("stop_loss") or 0)
            confidence = float(kwargs.get("confidence", 0))

            # Fill missing values from cache
            if cached:
                if entry <= 0:
                    entry = cached.current_price
                if target <= 0:
                    target = cached.target_price
                if stop <= 0:
                    stop = cached.stop_price
                # Override side from cache when LLM failed to pass prices
                # (if it can't get the prices right, the side is likely wrong too)
                if not side or (entry <= 0 and stop <= 0):
                    side = cached.direction
                if confidence <= 0:
                    confidence = cached.confidence

            # Defaults
            if not side:
                side = "long"
            if confidence <= 0:
                confidence = 0.5

            # Validate we have prices (either from cache or explicit)
            if entry <= 0 or target <= 0 or stop <= 0:
                missing = []
                if entry <= 0:
                    missing.append("entry_price")
                if target <= 0:
                    missing.append("target_price")
                if stop <= 0:
                    missing.append("stop_price")
                return (
                    f"⛔ Missing prices for {token}: {', '.join(missing)}.\n"
                    f"Run quant_analyze first to populate the signal cache, "
                    f"then call analyze_trade(token='{token}') — prices auto-fill."
                )

            # Check rejection counter — prevent infinite retry loops
            token_key = token.lower().strip()
            rejections = AnalyzeTradeTool._rejection_counts.get(token_key, 0)
            if rejections >= self._MAX_REJECTIONS_PER_TOKEN:
                return (
                    f"⛔ BLOCKED: {token} has been rejected {rejections} times this session.\n"
                    f"Do NOT retry this token with different parameters — the market conditions "
                    f"are not favorable.\n\n"
                    f"Instead:\n"
                    f"1. Use quant_analyze to find better opportunities across ALL tokens\n"
                    f"2. Try a different token entirely\n"
                    f"3. Wait for market conditions to change\n\n"
                    f"NEVER widen the stop-loss to pass risk checks — it makes R:R WORSE."
                )

            # Create a synthetic signal
            signal = Signal(
                token=token,
                direction=side,
                title=f"Manual analysis: {token}",
                description=f"Analyze {side} {token} @ ${entry:.4f}",
                current_price=entry,
                target_price=target,
                stop_price=stop,
                confidence=confidence,
                source="manual",
            )

            decision = self._strategy.evaluate_signal(signal)
            result = decision.summary()

            if decision.risk_check:
                result += f"\n\n{decision.risk_check.summary()}"

            # Track rejections per token
            if not decision.approved:
                AnalyzeTradeTool._rejection_counts[token_key] = rejections + 1
                remaining = self._MAX_REJECTIONS_PER_TOKEN - (rejections + 1)
                if remaining > 0:
                    result += (
                        f"\n\n⚠️ Rejection #{rejections + 1} for {token}. "
                        f"{remaining} attempt(s) remaining before this token is blocked.\n"
                        f"TIP: Use quant_analyze to get pre-computed setups with proper R:R."
                    )
                else:
                    result += (
                        f"\n\n🚫 {token} is now BLOCKED ({self._MAX_REJECTIONS_PER_TOKEN} rejections). "
                        f"Move on to other tokens or use quant_analyze."
                    )
            else:
                # Reset counter on approval
                AnalyzeTradeTool._rejection_counts.pop(token_key, None)

            # Log the analysis decision
            _decision_logger().log_trade_decision(
                token=token,
                direction=side,
                entry_price=entry,
                amount=decision.suggested_amount if decision.approved else 0,
                cost=decision.suggested_cost if decision.approved else 0,
                approved=decision.approved,
                reason=decision.reason or "",
                confidence=confidence,
                risk_reward=(target - entry) / (entry - stop) if entry != stop else 0,
                signal_type="manual_analysis",
                target_price=target,
                stop_price=stop,
            )

            return result

        except Exception as exc:
            return f"Error analyzing trade: {exc}"


class OpenPositionTool(BaseTool):
    """Open a new trade position and execute the swap atomically.

    This tool:
    1. Validates prices (from cache or explicit)
    2. Checks wallet balance to ensure sufficient funds
    3. Runs risk management checks
    4. Creates the position record
    5. Executes the swap on Jupiter
    6. On swap success: marks position OPEN with tx_signature
    7. On swap failure: auto-cancels the position, frees capacity
    """

    def __init__(
        self,
        working_dir: str,
        strategy: "TradingStrategy",
        positions: "PositionManager",
        risk_manager: "RiskManager",
        wallet: "Any | None" = None,
    ) -> None:
        super().__init__(working_dir)
        self._strategy = strategy
        self._positions = positions
        self._risk = risk_manager
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "open_position"

    @property
    def description(self) -> str:
        return (
            "Open a new trade position AND execute the swap on Jupiter atomically. "
            "Prices are auto-filled from the quant_analyze cache — just pass the token name. "
            "Checks wallet balance, runs risk checks, executes the swap, and tracks the position. "
            "If the swap fails, the position is auto-cancelled (no phantom positions). "
            "Set auto_execute=false to skip the swap and only create the tracking record."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "Token to trade (symbol, e.g. 'sol', 'bonk'). Prices auto-filled from quant_analyze cache.",
                },
                "side": {
                    "type": "string",
                    "enum": ["long", "short"],
                    "description": "Trade direction (default: auto from cached signal, or 'long')",
                },
                "entry_price": {
                    "type": "number",
                    "description": "Optional: override entry price (auto-filled from cache if omitted)",
                },
                "target_price": {
                    "type": "number",
                    "description": "Optional: override target price (auto-filled from cache if omitted)",
                },
                "stop_price": {
                    "type": "number",
                    "description": "Optional: override stop price (auto-filled from cache if omitted)",
                },
                "amount": {
                    "type": "number",
                    "description": "Token amount. If omitted, uses risk-based sizing capped to wallet balance.",
                },
                "auto_execute": {
                    "type": "boolean",
                    "description": "Execute the swap on Jupiter automatically (default: true). Set false to only create tracking record.",
                },
                "signal_id": {
                    "type": "string",
                    "description": "Optional: ID of the signal that triggered this trade",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about why this trade was opened",
                },
            },
            "required": ["token"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def _execute_swap(
        self, token: str, side: str, amount: float, entry_price: float,
    ) -> tuple[bool, str, str]:
        """Execute the Jupiter swap for this position.

        Returns (success, message, tx_signature).
        """
        if not self._wallet:
            return False, "No wallet configured — cannot execute swap.", ""
        if not self._wallet.can_sign:
            return False, "Wallet is in read-only mode — cannot sign transactions.", ""

        from claude1.tools.wallet_tools import (
            _resolve_mint, _to_lamports, _from_lamports,
            _jup_get, _sign_and_execute, SOL_MINT, USDC_MINT,
        )

        # Determine swap direction from position side:
        #   LONG  token → buy token  → input=USDC/SOL, output=token
        #   SHORT token → sell token → input=token, output=USDC
        token_mint = _resolve_mint(token)

        if side == "long":
            # Buying the token: we need to spend SOL or USDC to get it
            # If the token IS SOL, the position is "hold SOL" — no swap needed
            # (unless we're converting from USDC)
            if token_mint == SOL_MINT:
                # Long SOL = keep SOL. If we have USDC we swap USDC→SOL.
                # Otherwise the position is just "hold SOL" — no swap needed.
                # Check if we have USDC to swap
                from claude1.tools.wallet_tools import get_wallet_balance
                balances = get_wallet_balance(self._wallet.public_key)
                usdc_bal = balances.get("usdc", 0.0)
                if usdc_bal > 1.0:
                    # Swap USDC for SOL
                    input_mint = USDC_MINT
                    output_mint = SOL_MINT
                    swap_amount = min(amount * entry_price, usdc_bal - 0.5)  # keep small USDC buffer
                    amount_lamports = _to_lamports(swap_amount, input_mint)
                else:
                    # Already holding SOL — position is effectively "hold"
                    return True, "Position is LONG SOL — already holding SOL (no swap needed).", ""
            else:
                # Buying an altcoin: spend SOL (or USDC if available) to buy it
                from claude1.tools.wallet_tools import get_wallet_balance
                balances = get_wallet_balance(self._wallet.public_key)
                usdc_bal = balances.get("usdc", 0.0)
                if usdc_bal >= amount * entry_price:
                    input_mint = USDC_MINT
                    swap_amount = amount * entry_price
                    amount_lamports = _to_lamports(swap_amount, input_mint)
                else:
                    # Use SOL
                    input_mint = SOL_MINT
                    # Convert token amount * price to SOL amount
                    swap_amount = amount  # amount in target token
                    # For Jupiter, we specify input amount, so we need SOL amount
                    # cost_usd = amount * entry_price;  sol_amount = cost_usd / sol_price
                    # But Jupiter does this routing for us — just specify the output amount
                    # Actually, let's use the input side: swap SOL worth $cost
                    sol_bal = balances.get("sol", 0.0)
                    sol_cost = amount * entry_price  # this is in USD
                    # We need to know SOL price to convert
                    try:
                        price_resp = _jup_get("/price/v3", params={"ids": SOL_MINT})
                        sol_price_data = (price_resp.get("data", price_resp) if isinstance(price_resp, dict) else {}).get(SOL_MINT, {})
                        sol_price = float(sol_price_data.get("usdPrice", sol_price_data.get("price", 0)) or 0)
                    except Exception:
                        sol_price = entry_price if token_mint == SOL_MINT else 0
                    if sol_price <= 0:
                        return False, "Cannot determine SOL price for swap.", ""
                    sol_to_spend = sol_cost / sol_price
                    from claude1.tools.wallet_tools import SOL_GAS_RESERVE
                    if sol_to_spend > sol_bal - SOL_GAS_RESERVE:
                        return False, (
                            f"Insufficient SOL for swap: need {sol_to_spend:.6f} SOL "
                            f"(+ {SOL_GAS_RESERVE} gas reserve) but wallet has {sol_bal:.6f} SOL."
                        ), ""
                    swap_amount = sol_to_spend
                    amount_lamports = _to_lamports(swap_amount, input_mint)
                output_mint = token_mint
        else:
            # SHORT: sell the token for USDC/SOL
            input_mint = token_mint
            output_mint = USDC_MINT
            amount_lamports = _to_lamports(amount, input_mint)

        # Call Jupiter Ultra order
        try:
            order = _jup_get("/ultra/v1/order", params={
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount_lamports),
                "taker": self._wallet.public_key,
                "slippageBps": "50",
            })

            if isinstance(order, dict) and "error" in order:
                return False, f"Jupiter order failed: {order['error']}", ""

            tx_b64 = order.get("transaction", "") if isinstance(order, dict) else ""
            if not tx_b64:
                return False, f"No transaction returned from Jupiter.", ""

            request_id = order.get("requestId", "") if isinstance(order, dict) else ""
            in_raw = order.get("inAmount", amount_lamports) if isinstance(order, dict) else amount_lamports
            out_raw = order.get("outAmount", "?") if isinstance(order, dict) else "?"

            in_display = _from_lamports(in_raw, input_mint)
            out_display = _from_lamports(out_raw, output_mint) if out_raw != "?" else "?"

            # Sign and execute
            extra = {"requestId": request_id} if request_id else None
            result = _sign_and_execute(self._wallet, tx_b64, "/ultra/v1/execute", extra_body=extra)

            # Extract tx signature from result
            tx_sig = ""
            for line in result.split("\n"):
                if line.startswith("Tx signature:"):
                    tx_sig = line.split(":", 1)[1].strip()
                    break

            info = f"Swapped {in_display:.6f} → {out_display:.6f}"
            return True, f"{info}\n{result}", tx_sig

        except Exception as exc:
            return False, f"Swap execution error: {exc}", ""

    def execute(self, **kwargs: Any) -> str:
        try:
            from claude1.trader.signals import Signal

            token = kwargs.get("token", "")
            side = kwargs.get("side", "")
            auto_execute = kwargs.get("auto_execute", True)
            if isinstance(auto_execute, str):
                auto_execute = auto_execute.lower() not in ("false", "0", "no")

            # Auto-lookup from signal cache (populated by quant_analyze)
            cached = _get_cached_signal(token)

            entry = float(kwargs.get("entry_price") or kwargs.get("entry") or 0)
            target = float(kwargs.get("target_price") or kwargs.get("target") or 0)
            stop = float(kwargs.get("stop_price") or kwargs.get("stop") or kwargs.get("stop_loss") or 0)
            signal_id = kwargs.get("signal_id", "")
            notes = kwargs.get("notes", "")

            # Fill missing values from cache
            if cached:
                if entry <= 0:
                    entry = cached.current_price
                if target <= 0:
                    target = cached.target_price
                if stop <= 0:
                    stop = cached.stop_price
                # Override side from cache when LLM failed to pass prices
                if not side or (entry <= 0 and stop <= 0):
                    side = cached.direction
                if not signal_id:
                    signal_id = cached.id

            # Defaults
            if not side:
                side = "long"

            # Shorts are paper-only on spot DEXes — never execute a swap
            if side == "short":
                auto_execute = False

            # Validate prices
            if entry <= 0 or target <= 0 or stop <= 0:
                missing = []
                if entry <= 0:
                    missing.append("entry_price")
                if target <= 0:
                    missing.append("target_price")
                if stop <= 0:
                    missing.append("stop_price")
                return (
                    f"⛔ Missing prices for {token}: {', '.join(missing)}.\n"
                    f"Run quant_analyze first to populate the signal cache, "
                    f"then call open_position(token='{token}') — prices auto-fill."
                )

            # ── Pre-check wallet balance ──────────────────────────────────
            available_balance_usd: float | None = None
            if self._wallet and auto_execute:
                try:
                    from claude1.tools.wallet_tools import get_wallet_balance_usd
                    available_balance_usd = get_wallet_balance_usd(self._wallet.public_key)
                    if available_balance_usd <= 1.0:
                        return (
                            f"⛔ Insufficient wallet balance: ${available_balance_usd:.2f} available.\n"
                            f"Need at least $1.00 to trade (including gas reserves)."
                        )
                except Exception:
                    pass  # Best-effort; proceed without balance check

            # Calculate amount if not provided (cap to wallet balance)
            amount = kwargs.get("amount")
            if amount is None:
                amount = self._risk.get_position_size(
                    entry, stop,
                    available_balance_usd=available_balance_usd,
                )
            amount = float(amount)

            if amount <= 0:
                return (
                    f"⛔ Position size computed to zero for {token}.\n"
                    f"Wallet balance: ${available_balance_usd:.2f}" if available_balance_usd else
                    f"⛔ Position size computed to zero for {token}."
                )

            cost = amount * entry

            # Extra balance check: cost vs available
            if available_balance_usd is not None and cost > available_balance_usd - 1.0:
                return (
                    f"⛔ Insufficient funds: position cost ${cost:.2f} exceeds "
                    f"available balance ${available_balance_usd:.2f} (minus $1.00 gas reserve).\n"
                    f"Reduce position size or add funds."
                )

            # Run risk check
            risk_check = self._risk.check_trade(
                token=token, side=side, entry_price=entry,
                target_price=target, stop_price=stop, amount=amount,
            )

            if not risk_check.approved:
                _decision_logger().log_position_action(
                    action="rejected",
                    token=token,
                    direction=side,
                    price=entry,
                    amount=amount,
                    reason=f"Risk rejected: {risk_check.summary()[:200]}",
                )
                return f"Position REJECTED by risk manager:\n{risk_check.summary()}"

            # ── Create position in PLANNED state ──────────────────────────
            pos = self._positions.create(
                token=token, side=side, entry_price=entry,
                target_price=target, stop_price=stop, amount=amount,
                signal_id=signal_id, strategy="manual",
                notes=notes,
            )

            # ── Execute swap (if auto_execute) ────────────────────────────
            tx_sig = ""
            swap_msg = ""

            if auto_execute and self._wallet:
                swap_ok, swap_msg, tx_sig = self._execute_swap(
                    token=token, side=side, amount=amount, entry_price=entry,
                )

                if not swap_ok:
                    # Auto-cancel the position — no phantom positions
                    self._positions.cancel(pos.id, notes=f"Swap failed: {swap_msg[:200]}")
                    _decision_logger().log_position_action(
                        action="cancelled",
                        position_id=pos.id,
                        token=token,
                        direction=side,
                        price=entry,
                        amount=amount,
                        reason=f"Swap failed: {swap_msg[:200]}",
                    )
                    return (
                        f"⛔ Position CANCELLED — swap failed:\n"
                        f"  {swap_msg}\n\n"
                        f"Position {pos.id} auto-cancelled. Portfolio capacity freed.\n"
                        f"Use quant_analyze to find another opportunity."
                    )

            # ── Mark as OPEN ──────────────────────────────────────────────
            pos = self._positions.open_position(
                pos.id, entry_price=entry, amount=amount, tx_signature=tx_sig,
            )

            # ── Post-swap verification ────────────────────────────────────
            verified = False
            verify_msg = ""
            if tx_sig and self._wallet:
                try:
                    import time
                    time.sleep(2)  # Wait for on-chain confirmation
                    from claude1.tools.wallet_tools import get_wallet_balance
                    balances = get_wallet_balance(self._wallet.public_key)
                    held = balances.get(token.lower(), 0.0)
                    if held > 0:
                        pos.verified = True
                        self._positions._save()
                        verified = True
                        verify_msg = f"✅ Verified: wallet holds {held:.4f} {token}"
                    else:
                        verify_msg = (
                            f"⚠️ Token not yet detected in wallet — "
                            f"run position_status(action='reconcile') to verify later"
                        )
                except Exception:
                    verify_msg = "⚠️ Could not verify post-swap — run reconcile to check"

            rr = (target - entry) / (entry - stop) if side == "long" and entry != stop else \
                 (entry - target) / (stop - entry) if side == "short" and stop != entry else 0

            _decision_logger().log_position_action(
                action="open",
                position_id=pos.id,
                token=token,
                direction=side,
                price=entry,
                amount=amount,
                reason=notes or f"R:R={rr:.1f}",
            )

            result = (
                f"📊 Position OPENED:\n"
                f"  ID: {pos.id}\n"
                f"  {side.upper()} {token} — {amount:.4f} tokens @ ${entry:.4f}\n"
                f"  Cost: ${cost:.2f}\n"
                f"  Target: ${target:.4f} | Stop: ${stop:.4f}\n"
                f"  R:R: {rr:.1f}\n"
            )

            if tx_sig:
                result += f"  Tx: {tx_sig}\n"

            if swap_msg:
                result += f"\n✅ Swap executed:\n  {swap_msg}\n"
                if verify_msg:
                    result += f"  {verify_msg}\n"
            elif not auto_execute:
                if side == "short":
                    result += (
                        f"\n📝 SHORT position is paper-only (no swap on spot DEX).\n"
                        f"Tracking for P&L purposes only.\n"
                    )
                elif side == "long":
                    result += (
                        f"\n⚠️  Swap NOT executed (auto_execute=false).\n"
                        f"Run: swap_tokens(input_token='usdc', output_token='{token}', amount={cost:.2f})\n"
                        f"  OR: swap_tokens(input_token='sol', output_token='{token}', amount=<SOL_amount>)\n"
                    )

            result += "\nUse position_status to track this position."
            return result

        except Exception as exc:
            return f"Error opening position: {exc}"


class PositionStatusTool(BaseTool):
    """Check status of trading positions and manage them."""

    def __init__(
        self,
        working_dir: str,
        positions: "PositionManager",
        strategy: "TradingStrategy",
        wallet: "Any | None" = None,
    ) -> None:
        super().__init__(working_dir)
        self._positions = positions
        self._strategy = strategy
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "position_status"

    @property
    def description(self) -> str:
        return (
            "Check and manage trade positions. View active positions, close them, "
            "update stop-losses, or check if any stops/targets were hit."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "position_id": {
                    "type": "string",
                    "description": "Specific position ID to check/manage",
                },
                "action": {
                    "type": "string",
                    "enum": ["status", "close", "stop_out", "cancel", "update_stop", "update_target", "check_prices", "reconcile"],
                    "description": "Action to perform (default: status)",
                },
                "price": {
                    "type": "number",
                    "description": "For close/stop_out: exit price. For update_stop/update_target: new price level.",
                },
                "current_prices": {
                    "type": "object",
                    "description": "For check_prices: {token: current_price} to check stops/targets",
                },
                "auto_close_phantoms": {
                    "type": "boolean",
                    "description": "For reconcile: if true, auto-cancel phantom positions (default: false)",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes for the action",
                },
            },
            "required": [],
        }

    def _fetch_live_prices(self) -> dict[str, float]:
        """Fetch live USD prices from Jupiter for all tokens in active positions.

        Returns ``{token_lower: usd_price}`` dict.
        """
        active = self._positions.list_active()
        if not active:
            return {}

        try:
            from claude1.tools.wallet_tools import _jup_get, _resolve_mint

            tokens = {p.token.lower() for p in active}
            mints = {tok: _resolve_mint(tok) for tok in tokens}

            ids_param = ",".join(mints.values())
            resp = _jup_get("/price/v3", params={"ids": ids_param})
            price_data = resp.get("data", resp) if isinstance(resp, dict) else {}

            prices: dict[str, float] = {}
            for tok, mint in mints.items():
                info = price_data.get(mint, {})
                if isinstance(info, dict):
                    p = float(info.get("usdPrice", info.get("price", 0)) or 0)
                    if p > 0:
                        prices[tok] = p
            return prices
        except Exception as exc:
            logger.debug("Live price fetch failed: %s", exc)
            return {}

    def _get_wallet_holdings(self) -> dict[str, float]:
        """Fetch wallet token holdings.  Returns ``{token_lower: amount}``."""
        if not self._wallet:
            return {}
        try:
            from claude1.tools.wallet_tools import get_wallet_balance
            return get_wallet_balance(self._wallet.public_key)
        except Exception:
            return {}

    @staticmethod
    def _compute_phantom_ids(
        active: list, holdings: dict[str, float],
    ) -> tuple[set[str], float]:
        """Identify phantom positions by aggregating tracked amounts per token.

        Groups active LONG positions by token, sums their amounts, and
        compares against wallet holdings.  If tracked total exceeds held
        amount (with 20% tolerance for slippage/rounding), the **newest**
        positions (by ``opened_at`` / ``created_at``) are flagged as
        phantoms until the excess is covered.

        Returns ``(phantom_ids, phantom_capital)`` where *phantom_capital*
        is the sum of ``cost_basis`` for all identified phantom positions.
        """
        from claude1.trader.positions import PositionSide
        from collections import defaultdict

        # Group LONG positions by token
        by_token: dict[str, list] = defaultdict(list)
        for pos in active:
            if pos.side == PositionSide.LONG.value:
                by_token[pos.token.lower()].append(pos)

        phantom_ids: set[str] = set()
        phantom_capital = 0.0

        for tok, positions in by_token.items():
            tracked_total = sum(p.amount for p in positions)
            held = holdings.get(tok, 0.0)

            # 20% tolerance: only flag if tracked significantly exceeds held
            if tracked_total <= held * 1.2:
                continue

            excess = tracked_total - held

            # Sort newest-first (by opened_at, then created_at as fallback)
            positions.sort(
                key=lambda p: p.opened_at or p.created_at or "",
                reverse=True,
            )

            accumulated = 0.0
            for pos in positions:
                if accumulated >= excess:
                    break
                phantom_ids.add(pos.id)
                phantom_capital += pos.cost_basis
                accumulated += pos.amount

        return phantom_ids, phantom_capital

    def _reconcile(self, auto_close: bool = False) -> str:
        """Compare active positions against wallet holdings and flag discrepancies."""
        from claude1.trader.positions import PositionSide

        active = self._positions.list_active()
        if not active:
            return "No active positions to reconcile."

        prices = self._fetch_live_prices()
        if prices:
            self._positions.update_prices_bulk(prices)

        holdings = self._get_wallet_holdings()

        # Use aggregate phantom detection
        phantom_ids, _ = self._compute_phantom_ids(active, holdings)

        phantoms: list = []
        mismatches: list = []
        verified: list = []

        # Per-token tracked totals for non-phantom positions (for mismatch check)
        from collections import defaultdict
        non_phantom_by_token: dict[str, float] = defaultdict(float)

        for pos in active:
            tok = pos.token.lower()
            # Shorts are paper-only — always "verified" from wallet perspective
            if pos.side == PositionSide.SHORT.value:
                verified.append(pos)
                continue

            if pos.id in phantom_ids:
                phantoms.append(pos)
            else:
                non_phantom_by_token[tok] += pos.amount
                verified.append(pos)

        # Check for per-token mismatches among non-phantom positions
        # Move verified positions to mismatches if the aggregate doesn't match
        checked_tokens: set[str] = set()
        final_verified: list = []
        for pos in verified:
            if pos.side == PositionSide.SHORT.value:
                final_verified.append(pos)
                continue
            tok = pos.token.lower()
            if tok in checked_tokens:
                # Already evaluated this token — keep same classification
                held = holdings.get(tok, 0.0)
                tracked = non_phantom_by_token.get(tok, 0.0)
                if tracked > 0 and abs(held - tracked) / max(tracked, 1e-9) > 0.10:
                    mismatches.append((pos, held))
                else:
                    final_verified.append(pos)
                continue
            checked_tokens.add(tok)
            held = holdings.get(tok, 0.0)
            tracked = non_phantom_by_token.get(tok, 0.0)
            if tracked > 0 and abs(held - tracked) / max(tracked, 1e-9) > 0.10:
                mismatches.append((pos, held))
            else:
                final_verified.append(pos)
        verified = final_verified

        lines = ["═══ Position Reconciliation ═══\n"]

        if verified:
            lines.append(f"✅ Verified ({len(verified)}):")
            for pos in verified:
                tag = " [PAPER]" if pos.side == PositionSide.SHORT.value else ""
                lines.append(f"  [{pos.id}] {pos.token}{tag} — {pos.amount:.4f} tokens")
            lines.append("")

        if mismatches:
            lines.append(f"⚠️ Amount mismatch ({len(mismatches)}):")
            for pos, held in mismatches:
                pct = abs(held - pos.amount) / max(pos.amount, 1e-9) * 100
                lines.append(
                    f"  [{pos.id}] {pos.token} — tracked: {pos.amount:.4f}, "
                    f"wallet: {held:.4f} (off by {pct:.0f}%)"
                )
            lines.append("")

        if phantoms:
            phantom_capital = sum(p.cost_basis for p in phantoms)
            lines.append(f"⚠️ PHANTOM positions ({len(phantoms)}) — ${phantom_capital:.2f} blocked:")
            for pos in phantoms:
                lines.append(
                    f"  [{pos.id}] {pos.token} — ${pos.cost_basis:.2f} "
                    f"(entry=${pos.entry_price:.4f}, {pos.amount:.4f} tokens)"
                )
            lines.append("")

            if auto_close:
                cancelled_ids = []
                for pos in phantoms:
                    self._positions.cancel(pos.id, notes="Auto-cancelled: phantom position (token not in wallet)")
                    cancelled_ids.append(pos.id)
                lines.append(
                    f"🗑️ Auto-cancelled {len(cancelled_ids)} phantom positions: "
                    f"{', '.join(cancelled_ids)}"
                )
                lines.append(f"💰 Freed ${phantom_capital:.2f} portfolio capacity")
            else:
                lines.append(
                    "💡 Run position_status(action='reconcile', auto_close_phantoms=true) "
                    "to clean up phantom positions."
                )
        else:
            lines.append("✅ No phantom positions found.")

        price_note = "(prices updated live)" if prices else "(using cached prices)"
        lines.append(f"\n{price_note}")
        return "\n".join(lines)

    def execute(self, **kwargs: Any) -> str:
        position_id = kwargs.get("position_id", "")
        action = kwargs.get("action", "status")
        price = kwargs.get("price", 0)
        current_prices = kwargs.get("current_prices", {})
        auto_close_phantoms = kwargs.get("auto_close_phantoms", False)
        if isinstance(auto_close_phantoms, str):
            auto_close_phantoms = auto_close_phantoms.lower() in ("true", "1", "yes")
        notes = kwargs.get("notes", "")

        try:
            # ── Reconcile action ──────────────────────────────────────────
            if action == "reconcile":
                return self._reconcile(auto_close=bool(auto_close_phantoms))

            # ── Check prices against all positions ────────────────────────
            if action == "check_prices" and current_prices:
                actions = self._strategy.check_positions(current_prices)
                if not actions:
                    return "✅ All positions within range. No stops or targets hit."

                lines = ["⚠️ Position alerts:"]
                for a in actions:
                    lines.append(f"  [{a['action'].upper()}] {a['token']} — {a['reason']}")
                    lines.append(f"    Position: {a['position_id']}")
                return "\n".join(lines)

            # ── Specific position actions ─────────────────────────────────
            if position_id and action != "status":
                pos = self._positions.get(position_id)
                if pos is None:
                    return f"Position '{position_id}' not found."

                if action == "close":
                    if not price:
                        return "Error: price is required to close a position."
                    pos = self._positions.close_position(position_id, float(price), notes=notes)
                    _decision_logger().log_position_action(
                        action="close",
                        position_id=position_id,
                        token=getattr(pos, "token", ""),
                        direction=getattr(pos, "side", ""),
                        price=float(price),
                        pnl=getattr(pos, "pnl", 0),
                        reason=notes or "Manual close",
                    )
                    return f"Position closed: {pos.summary()}"

                elif action == "stop_out":
                    pos = self._positions.stop_out(
                        position_id, exit_price=float(price) if price else None, notes=notes
                    )
                    _decision_logger().log_position_action(
                        action="stop_out",
                        position_id=position_id,
                        token=getattr(pos, "token", ""),
                        direction=getattr(pos, "side", ""),
                        price=float(price) if price else 0,
                        pnl=getattr(pos, "pnl", 0),
                        reason=notes or "Stop-loss hit",
                    )
                    return f"Position stopped out: {pos.summary()}"

                elif action == "cancel":
                    pos = self._positions.cancel(position_id, notes=notes)
                    _decision_logger().log_position_action(
                        action="cancel",
                        position_id=position_id,
                        token=getattr(pos, "token", ""),
                        reason=notes or "Cancelled",
                    )
                    return f"Position cancelled: {pos.summary()}"

                elif action == "update_stop":
                    if not price:
                        return "Error: price is required to update stop-loss."
                    pos = self._positions.update_stop(position_id, float(price), notes=notes)
                    return f"Stop updated: {pos.summary()}"

                elif action == "update_target":
                    if not price:
                        return "Error: price is required to update target."
                    pos = self._positions.update_target(position_id, float(price), notes=notes)
                    return f"Target updated: {pos.summary()}"

            # ── Fetch live prices for display ─────────────────────────────
            live_prices = self._fetch_live_prices()
            price_note = "(prices updated live)" if live_prices else "(using cached prices)"
            if live_prices:
                self._positions.update_prices_bulk(live_prices)

            # ── Show single position ──────────────────────────────────────
            if position_id:
                pos = self._positions.get(position_id)
                if pos is None:
                    return f"Position '{position_id}' not found."
                return f"{pos.summary()}\n{price_note}"

            # ── Show all positions ────────────────────────────────────────
            from claude1.trader.positions import PositionSide

            active = self._positions.list_active()
            all_positions = self._positions.list_all()

            lines = [f"═══ Trading Positions ({len(active)} active, {len(all_positions)} total) ═══\n"]

            # Check for phantoms if wallet is available
            holdings = self._get_wallet_holdings() if self._wallet else {}
            phantom_ids, phantom_capital = (
                self._compute_phantom_ids(active, holdings)
                if holdings else (set(), 0.0)
            )

            if active:
                # Detect short positions
                has_shorts = any(p.side == PositionSide.SHORT.value for p in active)

                lines.append("📊 Active positions:")
                for pos in active:
                    summary_line = pos.summary()

                    if pos.id in phantom_ids:
                        summary_line += " ⚠️ PHANTOM"

                    lines.append(f"  {summary_line}")

                if has_shorts:
                    lines.append("\n  ℹ️ Short positions are paper-only (no swap on spot DEX)")
                lines.append("")

            # Recent closed positions
            closed = [p for p in all_positions if p.is_closed][-5:]  # Last 5
            if closed:
                lines.append("📋 Recent closed:")
                for pos in closed:
                    lines.append(f"  {pos.summary()}")
                lines.append("")

            # Aggregate stats
            total_pnl = self._positions.total_pnl()
            pnl_str = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
            lines.append(f"💰 Total P&L: {pnl_str} | Win rate: {self._positions.win_rate():.0%}")
            lines.append(f"📦 Capital deployed: ${self._positions.total_invested():.2f}")

            # Phantom warning
            if phantom_capital > 0:
                lines.append(
                    f"\n⚠️ ${phantom_capital:.2f} in phantom positions — "
                    f"run position_status(action='reconcile', auto_close_phantoms=true) to clean up"
                )

            lines.append(f"\n{price_note}")
            return "\n".join(lines)

        except Exception as exc:
            return f"Error checking positions: {exc}"


class TradingReportTool(BaseTool):
    """Generate a comprehensive trading performance report."""

    def __init__(
        self,
        working_dir: str,
        strategy: "TradingStrategy",
        risk_manager: "RiskManager",
    ) -> None:
        super().__init__(working_dir)
        self._strategy = strategy
        self._risk = risk_manager

    @property
    def name(self) -> str:
        return "trading_report"

    @property
    def description(self) -> str:
        return (
            "Generate a comprehensive trading performance report. Shows win rate, "
            "P&L, profit factor, risk metrics, portfolio health, and circuit breaker status."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["all_time", "today", "this_week", "this_month"],
                    "description": "Time period for the report (default: all_time)",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        period = kwargs.get("period", "all_time")

        try:
            report = self._strategy.generate_report(period)
            result = report.summary()

            # Add portfolio health details
            health = self._risk.portfolio_health()
            if health and health.get("status") != "no_positions_manager":
                result += "\n\n═══ Portfolio Health ═══"
                result += f"\n  Active: {health.get('active_positions', 0)} positions"
                result += f"\n  Invested: ${health.get('total_invested', 0):.2f}"
                result += f"\n  Capacity remaining: ${health.get('portfolio_capacity_remaining', 0):.2f}"
                result += f"\n  Drawdown: {health.get('drawdown_pct', 0):.1f}%"
                result += f"\n  Circuit breaker: {health.get('circuit_breaker', '?')}"

                limits = health.get("limits", {})
                if limits:
                    result += "\n\n  Risk limits:"
                    result += f"\n    Max position: ${limits.get('max_position_size', 0):.2f}"
                    result += f"\n    Max portfolio: ${limits.get('max_portfolio_size', 0):.2f}"
                    result += f"\n    Max concurrent: {limits.get('max_concurrent', 0)}"
                    result += f"\n    Min R:R: {limits.get('min_risk_reward', 0):.1f}"
                    result += f"\n    Max daily loss: ${limits.get('max_daily_loss', 0):.2f}"

            return result

        except Exception as exc:
            return f"Error generating trading report: {exc}"


class DecisionLogTool(BaseTool):
    """View the agent's decision history (from Google Sheet + local logs)."""

    def __init__(self, working_dir: str) -> None:
        super().__init__(working_dir)

    @property
    def name(self) -> str:
        return "decision_log"

    @property
    def description(self) -> str:
        return (
            "View the agent's recent decision history. Shows all trading decisions, "
            "market scans, position actions, and agent task results. "
            "Use to review past choices and improve future decisions."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max decisions to return (default: 20)",
                },
                "filter_tool": {
                    "type": "string",
                    "description": "Filter by tool name (e.g., 'trade_decision', 'market_scan', 'position_mgmt')",
                },
                "filter_decision": {
                    "type": "string",
                    "enum": ["APPROVED", "REJECTED", "INFO", "EXECUTED", "FAILED"],
                    "description": "Filter by decision type",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        limit = kwargs.get("limit", 20)
        filter_tool = kwargs.get("filter_tool", "")
        filter_decision = kwargs.get("filter_decision", "")

        try:
            decisions = _decision_logger().get_recent_decisions(limit=limit * 2)  # Fetch extra for filtering

            if filter_tool:
                decisions = [d for d in decisions if d.get("tool") == filter_tool]
            if filter_decision:
                decisions = [d for d in decisions if d.get("decision") == filter_decision]

            decisions = decisions[:limit]

            if not decisions:
                return "📋 No decisions found in the log."

            status = _decision_logger().get_status()
            lines = [
                f"📋 Decision Log ({len(decisions)} entries)"
                f" | GSheet: {'✅' if status['gsheet_configured'] else '❌'}"
                f" | Local: {status['local_log_dir']}\n"
            ]

            for d in decisions:
                ts = d.get("timestamp", "?")[:19]  # Trim microseconds
                tool = d.get("tool", "?")
                action = d.get("action", "?")
                decision = d.get("decision", "?")
                token = d.get("token", "")
                reason = d.get("reason", "")

                icon = {"APPROVED": "✅", "REJECTED": "⛔", "INFO": "ℹ️", "EXECUTED": "📊", "FAILED": "❌"}.get(decision, "•")

                line = f"  {icon} [{ts}] {tool}/{action}"
                if token:
                    line += f" — {token}"
                if d.get("direction"):
                    line += f" ({d['direction']})"
                if d.get("price"):
                    line += f" @ ${d['price']}"
                line += f" → {decision}"
                if reason:
                    line += f"\n     {reason[:100]}"
                lines.append(line)

            return "\n".join(lines)

        except Exception as exc:
            return f"Error reading decision log: {exc}"

