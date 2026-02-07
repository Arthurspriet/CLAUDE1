"""Jupiter (Solana DEX) financial tools.

All tools inherit BaseTool. Transaction tools require user confirmation.
Read-only tools work even in read-only wallet mode.
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

import httpx

from claude1.config import (
    JUP_API_KEY,
    JUP_BASE_URL,
    JUP_HTTP_TIMEOUT,
    SOL_MINT,
    TOKEN_ALIASES,
)
from claude1.tools.base import BaseTool

if TYPE_CHECKING:
    from claude1.wallet import SolanaWallet


# ── Shared helpers ──────────────────────────────────────────────────────────


def _jup_headers() -> dict[str, str]:
    """HTTP headers for Jupiter API requests."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if JUP_API_KEY:
        headers["x-api-key"] = JUP_API_KEY
    return headers


def _jup_get(path: str, params: dict[str, Any] | None = None) -> dict | list:
    """GET request to Jupiter API."""
    url = f"{JUP_BASE_URL}{path}"
    resp = httpx.get(url, headers=_jup_headers(), params=params, timeout=JUP_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _jup_post(path: str, body: dict[str, Any]) -> dict | list:
    """POST request to Jupiter API."""
    url = f"{JUP_BASE_URL}{path}"
    resp = httpx.post(url, headers=_jup_headers(), json=body, timeout=JUP_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _resolve_mint(token: str) -> str:
    """Map a token alias (sol, usdc, usdt) to its mint address, or pass through."""
    return TOKEN_ALIASES.get(token.lower().strip(), token.strip())


def _format_token_amount(raw: int | float, decimals: int) -> str:
    """Convert raw lamports/smallest-unit to human-readable amount."""
    value = float(raw) / (10 ** decimals)
    # Remove trailing zeros but keep at least one decimal
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


# ── Read-only tools ─────────────────────────────────────────────────────────


class WalletBalanceTool(BaseTool):
    """Check wallet token holdings and SOL balance."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "wallet_balance"

    @property
    def description(self) -> str:
        return (
            "Check Solana wallet token holdings and SOL balance. "
            "Returns all tokens held with amounts and USD values."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Wallet address to check. Defaults to the configured wallet.",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            address = kwargs.get("address", self._wallet.public_key)
            data = _jup_get(f"/ultra/v1/holdings/{address}")
            if not data:
                return f"No holdings found for {address}"

            lines = [f"Holdings for {address}:"]
            if isinstance(data, list):
                for item in data:
                    symbol = item.get("symbol", "???")
                    amount = item.get("uiAmount", item.get("amount", "?"))
                    usd = item.get("valueUsd", "")
                    usd_str = f" (${usd})" if usd else ""
                    lines.append(f"  {symbol}: {amount}{usd_str}")
            else:
                lines.append(json.dumps(data, indent=2))
            return "\n".join(lines)
        except Exception as exc:
            return f"Error fetching balance: {exc}"


class TokenPriceTool(BaseTool):
    """Get USD prices for tokens."""

    def __init__(self, working_dir: str) -> None:
        super().__init__(working_dir)

    @property
    def name(self) -> str:
        return "token_price"

    @property
    def description(self) -> str:
        return (
            "Get current USD price for one or more Solana tokens. "
            "Accepts token symbols (sol, usdc) or mint addresses."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tokens": {
                    "type": "string",
                    "description": "Comma-separated token symbols or mint addresses (e.g. 'sol,usdc').",
                },
            },
            "required": ["tokens"],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            raw_tokens = kwargs["tokens"]
            mints = [_resolve_mint(t) for t in raw_tokens.split(",")]
            ids_param = ",".join(mints)
            data = _jup_get("/price/v3", params={"ids": ids_param})

            lines = []
            prices = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(prices, dict):
                for mint, info in prices.items():
                    if isinstance(info, dict):
                        price = info.get("price", "N/A")
                        symbol = info.get("symbol", mint[:8])
                        lines.append(f"{symbol}: ${price}")
                    else:
                        lines.append(f"{mint[:8]}: {info}")
            else:
                lines.append(json.dumps(data, indent=2))
            return "\n".join(lines) if lines else "No price data returned."
        except Exception as exc:
            return f"Error fetching price: {exc}"


class TokenSearchTool(BaseTool):
    """Search for Solana tokens by name or symbol."""

    def __init__(self, working_dir: str) -> None:
        super().__init__(working_dir)

    @property
    def name(self) -> str:
        return "token_search"

    @property
    def description(self) -> str:
        return "Search for Solana tokens by name or symbol. Returns token info including mint address."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Token name or symbol to search for (e.g. 'bonk', 'jupiter').",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            query = kwargs["query"]
            data = _jup_get("/tokens/v2/search", params={"query": query})

            if not data:
                return f"No tokens found matching '{query}'."

            lines = [f"Tokens matching '{query}':"]
            items = data if isinstance(data, list) else [data]
            for token in items[:10]:  # Limit to 10 results
                name = token.get("name", "?")
                symbol = token.get("symbol", "?")
                mint = token.get("address", token.get("mint", "?"))
                lines.append(f"  {symbol} ({name}): {mint}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error searching tokens: {exc}"


class TokenShieldTool(BaseTool):
    """Check token security/risk assessment."""

    def __init__(self, working_dir: str) -> None:
        super().__init__(working_dir)

    @property
    def name(self) -> str:
        return "token_shield"

    @property
    def description(self) -> str:
        return (
            "Check security and risk assessment for a Solana token. "
            "Use before trading unfamiliar tokens to detect scams/rugs."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "mint": {
                    "type": "string",
                    "description": "Token mint address or symbol (e.g. 'sol' or full mint address).",
                },
            },
            "required": ["mint"],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            mint = _resolve_mint(kwargs["mint"])
            data = _jup_get("/ultra/v1/shield", params={"mints": mint})

            if not data:
                return f"No security data for {mint}."

            lines = [f"Token Shield for {mint}:"]
            if isinstance(data, dict):
                for key, val in data.items():
                    lines.append(f"  {key}: {val}")
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        symbol = item.get("symbol", item.get("mint", "?")[:8])
                        risk = item.get("riskLevel", item.get("risk", "unknown"))
                        lines.append(f"  {symbol}: risk={risk}")
                        for k, v in item.items():
                            if k not in ("symbol", "mint", "riskLevel", "risk"):
                                lines.append(f"    {k}: {v}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error checking token shield: {exc}"


class ListLimitOrdersTool(BaseTool):
    """List active and historical limit orders."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "list_limit_orders"

    @property
    def description(self) -> str:
        return "List active and historical limit orders for the configured wallet."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status: 'active', 'completed', 'cancelled', or 'all'. Default: 'active'.",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            status = kwargs.get("status", "active")
            data = _jup_get(
                "/trigger/v1/getTriggerOrders",
                params={"wallet": self._wallet.public_key, "status": status},
            )

            if not data:
                return f"No limit orders found (status={status})."

            orders = data if isinstance(data, list) else data.get("orders", [data])
            lines = [f"Limit orders ({status}): {len(orders)} found"]
            for order in orders[:20]:
                if isinstance(order, dict):
                    oid = order.get("id", order.get("orderId", "?"))
                    input_mint = order.get("inputMint", "?")[:8]
                    output_mint = order.get("outputMint", "?")[:8]
                    amount = order.get("makingAmount", order.get("amount", "?"))
                    price = order.get("triggerPrice", order.get("price", "?"))
                    lines.append(f"  [{oid}] {input_mint}→{output_mint} amount={amount} trigger={price}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error listing limit orders: {exc}"


class ListDCAOrdersTool(BaseTool):
    """List active and historical DCA (Dollar-Cost Average) orders."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "list_dca_orders"

    @property
    def description(self) -> str:
        return "List active and historical DCA (Dollar-Cost Averaging) orders for the configured wallet."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by status: 'active', 'completed', 'cancelled', or 'all'. Default: 'active'.",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            status = kwargs.get("status", "active")
            data = _jup_get(
                "/recurring/v1/getRecurringOrders",
                params={"wallet": self._wallet.public_key, "status": status},
            )

            if not data:
                return f"No DCA orders found (status={status})."

            orders = data if isinstance(data, list) else data.get("orders", [data])
            lines = [f"DCA orders ({status}): {len(orders)} found"]
            for order in orders[:20]:
                if isinstance(order, dict):
                    oid = order.get("id", order.get("orderId", "?"))
                    input_mint = order.get("inputMint", "?")[:8]
                    output_mint = order.get("outputMint", "?")[:8]
                    amount = order.get("inAmount", order.get("amount", "?"))
                    interval = order.get("interval", "?")
                    lines.append(f"  [{oid}] {input_mint}→{output_mint} amount={amount} every {interval}s")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error listing DCA orders: {exc}"


class PortfolioPositionsTool(BaseTool):
    """View all DeFi positions across protocols."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "portfolio_positions"

    @property
    def description(self) -> str:
        return "View all DeFi positions (lending, staking, LP) across Solana protocols."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs: Any) -> str:
        try:
            data = _jup_get(
                "/portfolio/v1/positions",
                params={"wallet": self._wallet.public_key},
            )

            if not data:
                return "No DeFi positions found."

            positions = data if isinstance(data, list) else data.get("positions", [data])
            lines = [f"DeFi positions: {len(positions)} found"]
            for pos in positions[:20]:
                if isinstance(pos, dict):
                    protocol = pos.get("protocol", pos.get("platformName", "?"))
                    ptype = pos.get("type", pos.get("positionType", "?"))
                    value = pos.get("valueUsd", pos.get("value", "?"))
                    lines.append(f"  {protocol} ({ptype}): ${value}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error fetching positions: {exc}"


# ── Transaction tools (require confirmation) ────────────────────────────────


def _require_signing(wallet: SolanaWallet) -> str | None:
    """Return an error string if wallet cannot sign, else None."""
    if not wallet.can_sign:
        return "Error: Wallet is in read-only mode (no private key). Cannot execute transactions."
    return None


def _sign_and_execute(
    wallet: SolanaWallet,
    unsigned_tx_b64: str,
    execute_path: str,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Sign a transaction and POST it to Jupiter's execute endpoint."""
    signed_tx = wallet.sign_transaction(unsigned_tx_b64)
    body: dict[str, Any] = {"signedTransaction": signed_tx}
    if extra_body:
        body.update(extra_body)
    result = _jup_post(execute_path, body)
    if isinstance(result, dict):
        tx_id = result.get("txid", result.get("transactionId", result.get("signature", "")))
        status = result.get("status", "submitted")
        msg = f"Transaction {status}."
        if tx_id:
            msg += f"\nTx signature: {tx_id}"
            msg += f"\nhttps://solscan.io/tx/{tx_id}"
        return msg
    return json.dumps(result, indent=2)


class SwapTokensTool(BaseTool):
    """Swap tokens via Jupiter's best route."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "swap_tokens"

    @property
    def description(self) -> str:
        return (
            "Swap one Solana token for another via Jupiter DEX aggregator. "
            "Finds the best route and price. Requires user confirmation."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input_token": {
                    "type": "string",
                    "description": "Token to sell — symbol (sol, usdc) or mint address.",
                },
                "output_token": {
                    "type": "string",
                    "description": "Token to buy — symbol (sol, usdc) or mint address.",
                },
                "amount": {
                    "type": "number",
                    "description": "Amount of input token to swap (human-readable, e.g. 0.1 for 0.1 SOL).",
                },
                "slippage_bps": {
                    "type": "integer",
                    "description": "Slippage tolerance in basis points. Default 50 (0.5%).",
                },
            },
            "required": ["input_token", "output_token", "amount"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        try:
            err = _require_signing(self._wallet)
            if err:
                return err

            input_mint = _resolve_mint(kwargs["input_token"])
            output_mint = _resolve_mint(kwargs["output_token"])
            amount = kwargs["amount"]
            slippage = kwargs.get("slippage_bps", 50)

            # Step 1: Get order (unsigned tx)
            order = _jup_get("/ultra/v1/order", params={
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "taker": self._wallet.public_key,
                "slippageBps": str(slippage),
            })

            if isinstance(order, dict) and "error" in order:
                return f"Swap order failed: {order['error']}"

            tx_b64 = order.get("transaction", "") if isinstance(order, dict) else ""
            if not tx_b64:
                return f"No transaction returned from Jupiter. Response: {json.dumps(order, indent=2)}"

            # Show swap details
            in_amount = order.get("inAmount", amount)
            out_amount = order.get("outAmount", "?")
            price_impact = order.get("priceImpactPct", "?")

            info = (
                f"Swapping {in_amount} {kwargs['input_token']} → {out_amount} {kwargs['output_token']}\n"
                f"Price impact: {price_impact}%\n"
                f"Slippage: {slippage} bps"
            )

            # Step 2 & 3: Sign and execute
            result = _sign_and_execute(self._wallet, tx_b64, "/ultra/v1/execute")
            return f"{info}\n\n{result}"
        except Exception as exc:
            return f"Error swapping tokens: {exc}"


class CreateLimitOrderTool(BaseTool):
    """Place a limit order on Jupiter."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "create_limit_order"

    @property
    def description(self) -> str:
        return (
            "Place a limit order to buy/sell a token at a specific price. "
            "The order executes automatically when the price is reached."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input_token": {
                    "type": "string",
                    "description": "Token to sell — symbol or mint address.",
                },
                "output_token": {
                    "type": "string",
                    "description": "Token to buy — symbol or mint address.",
                },
                "amount": {
                    "type": "string",
                    "description": "Amount of input token to sell (human-readable).",
                },
                "trigger_price": {
                    "type": "string",
                    "description": "Price at which the order triggers (in output token per input token).",
                },
            },
            "required": ["input_token", "output_token", "amount", "trigger_price"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        try:
            err = _require_signing(self._wallet)
            if err:
                return err

            input_mint = _resolve_mint(kwargs["input_token"])
            output_mint = _resolve_mint(kwargs["output_token"])

            order = _jup_post("/trigger/v1/createOrder", {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "maker": self._wallet.public_key,
                "makingAmount": str(kwargs["amount"]),
                "triggerPrice": str(kwargs["trigger_price"]),
            })

            if isinstance(order, dict) and "error" in order:
                return f"Limit order failed: {order['error']}"

            tx_b64 = order.get("transaction", "") if isinstance(order, dict) else ""
            if not tx_b64:
                return f"No transaction returned. Response: {json.dumps(order, indent=2)}"

            result = _sign_and_execute(self._wallet, tx_b64, "/trigger/v1/execute")
            return f"Limit order: {kwargs['amount']} {kwargs['input_token']} → {kwargs['output_token']} @ {kwargs['trigger_price']}\n\n{result}"
        except Exception as exc:
            return f"Error creating limit order: {exc}"


class CancelLimitOrderTool(BaseTool):
    """Cancel an active limit order."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "cancel_limit_order"

    @property
    def description(self) -> str:
        return "Cancel an active limit order by order ID."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to cancel (from list_limit_orders).",
                },
            },
            "required": ["order_id"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        try:
            err = _require_signing(self._wallet)
            if err:
                return err

            order = _jup_post("/trigger/v1/cancelOrder", {
                "maker": self._wallet.public_key,
                "orderId": kwargs["order_id"],
            })

            if isinstance(order, dict) and "error" in order:
                return f"Cancel failed: {order['error']}"

            tx_b64 = order.get("transaction", "") if isinstance(order, dict) else ""
            if not tx_b64:
                return f"No transaction returned. Response: {json.dumps(order, indent=2)}"

            result = _sign_and_execute(self._wallet, tx_b64, "/trigger/v1/execute")
            return f"Cancelling limit order {kwargs['order_id']}\n\n{result}"
        except Exception as exc:
            return f"Error cancelling limit order: {exc}"


class CreateDCAOrderTool(BaseTool):
    """Set up a Dollar-Cost Averaging (DCA) order."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "create_dca_order"

    @property
    def description(self) -> str:
        return (
            "Set up a recurring buy (Dollar-Cost Averaging). "
            "Automatically buys a token at regular intervals."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input_token": {
                    "type": "string",
                    "description": "Token to spend — symbol or mint address.",
                },
                "output_token": {
                    "type": "string",
                    "description": "Token to buy — symbol or mint address.",
                },
                "total_amount": {
                    "type": "string",
                    "description": "Total amount of input token to spend over all intervals.",
                },
                "num_orders": {
                    "type": "integer",
                    "description": "Number of orders to split into (e.g. 10 for 10 buys).",
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": "Seconds between each buy (e.g. 86400 for daily, 3600 for hourly).",
                },
            },
            "required": ["input_token", "output_token", "total_amount", "num_orders", "interval_seconds"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        try:
            err = _require_signing(self._wallet)
            if err:
                return err

            input_mint = _resolve_mint(kwargs["input_token"])
            output_mint = _resolve_mint(kwargs["output_token"])

            order = _jup_post("/recurring/v1/createOrder", {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "maker": self._wallet.public_key,
                "inAmount": str(kwargs["total_amount"]),
                "numberOfOrders": kwargs["num_orders"],
                "interval": kwargs["interval_seconds"],
            })

            if isinstance(order, dict) and "error" in order:
                return f"DCA order failed: {order['error']}"

            tx_b64 = order.get("transaction", "") if isinstance(order, dict) else ""
            if not tx_b64:
                return f"No transaction returned. Response: {json.dumps(order, indent=2)}"

            result = _sign_and_execute(self._wallet, tx_b64, "/recurring/v1/execute")
            per_order = float(kwargs["total_amount"]) / kwargs["num_orders"]
            interval_h = kwargs["interval_seconds"] / 3600
            return (
                f"DCA: {kwargs['total_amount']} {kwargs['input_token']} → {kwargs['output_token']}\n"
                f"  {kwargs['num_orders']} buys of ~{per_order} every {interval_h:.1f}h\n\n{result}"
            )
        except Exception as exc:
            return f"Error creating DCA order: {exc}"


class CancelDCAOrderTool(BaseTool):
    """Cancel an active DCA order."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "cancel_dca_order"

    @property
    def description(self) -> str:
        return "Cancel an active DCA (Dollar-Cost Averaging) order by order ID."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The DCA order ID to cancel (from list_dca_orders).",
                },
            },
            "required": ["order_id"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        try:
            err = _require_signing(self._wallet)
            if err:
                return err

            order = _jup_post("/recurring/v1/cancelOrder", {
                "maker": self._wallet.public_key,
                "orderId": kwargs["order_id"],
            })

            if isinstance(order, dict) and "error" in order:
                return f"Cancel failed: {order['error']}"

            tx_b64 = order.get("transaction", "") if isinstance(order, dict) else ""
            if not tx_b64:
                return f"No transaction returned. Response: {json.dumps(order, indent=2)}"

            result = _sign_and_execute(self._wallet, tx_b64, "/recurring/v1/execute")
            return f"Cancelling DCA order {kwargs['order_id']}\n\n{result}"
        except Exception as exc:
            return f"Error cancelling DCA order: {exc}"


class LendDepositTool(BaseTool):
    """Deposit tokens to earn yield via Jupiter lending."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "lend_deposit"

    @property
    def description(self) -> str:
        return (
            "Deposit tokens into a lending protocol via Jupiter to earn yield/interest. "
            "Returns the estimated APY and transaction details."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "Token to deposit — symbol or mint address.",
                },
                "amount": {
                    "type": "string",
                    "description": "Amount to deposit (human-readable).",
                },
            },
            "required": ["token", "amount"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        try:
            err = _require_signing(self._wallet)
            if err:
                return err

            mint = _resolve_mint(kwargs["token"])

            order = _jup_post("/lend/v1/earn/deposit", {
                "mint": mint,
                "amount": str(kwargs["amount"]),
                "owner": self._wallet.public_key,
            })

            if isinstance(order, dict) and "error" in order:
                return f"Lend deposit failed: {order['error']}"

            tx_b64 = order.get("transaction", "") if isinstance(order, dict) else ""
            if not tx_b64:
                return f"No transaction returned. Response: {json.dumps(order, indent=2)}"

            apy = order.get("apy", "?")
            protocol = order.get("protocol", order.get("platform", "?"))

            result = _sign_and_execute(self._wallet, tx_b64, "/lend/v1/earn/execute")
            return (
                f"Depositing {kwargs['amount']} {kwargs['token']} to {protocol} (APY: {apy}%)\n\n{result}"
            )
        except Exception as exc:
            return f"Error with lend deposit: {exc}"


class SendTokensTool(BaseTool):
    """Send tokens to another wallet (irreversible)."""

    def __init__(self, working_dir: str, wallet: SolanaWallet) -> None:
        super().__init__(working_dir)
        self._wallet = wallet

    @property
    def name(self) -> str:
        return "send_tokens"

    @property
    def description(self) -> str:
        return (
            "Send tokens to another Solana wallet address. "
            "WARNING: This is irreversible. Double-check the recipient address."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "description": "Token to send — symbol or mint address.",
                },
                "amount": {
                    "type": "string",
                    "description": "Amount to send (human-readable).",
                },
                "recipient": {
                    "type": "string",
                    "description": "Recipient Solana wallet address.",
                },
            },
            "required": ["token", "amount", "recipient"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        try:
            err = _require_signing(self._wallet)
            if err:
                return err

            mint = _resolve_mint(kwargs["token"])

            order = _jup_post("/send/v1/craft-send", {
                "mint": mint,
                "amount": str(kwargs["amount"]),
                "sender": self._wallet.public_key,
                "recipient": kwargs["recipient"],
            })

            if isinstance(order, dict) and "error" in order:
                return f"Send failed: {order['error']}"

            tx_b64 = order.get("transaction", "") if isinstance(order, dict) else ""
            if not tx_b64:
                return f"No transaction returned. Response: {json.dumps(order, indent=2)}"

            result = _sign_and_execute(self._wallet, tx_b64, "/send/v1/execute")
            return (
                f"Sending {kwargs['amount']} {kwargs['token']} → {kwargs['recipient']}\n\n{result}"
            )
        except Exception as exc:
            return f"Error sending tokens: {exc}"
