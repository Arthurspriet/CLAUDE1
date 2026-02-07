"""Solana wallet — key loading and transaction signing.

The private key is NEVER returned, logged, or included in any output.
"""

from __future__ import annotations

import base64
import json
from typing import Any


class SolanaWallet:
    """Manages a Solana keypair for signing Jupiter transactions.

    Supports three private key formats:
      - base58 encoded string
      - JSON byte array (e.g. "[1,2,3,...]")
      - hex encoded string

    Falls back to read-only mode if only SOLANA_WALLET_ADDRESS is set.
    """

    def __init__(self) -> None:
        self._keypair: Any | None = None
        self._public_key_str: str = ""
        self._load_keys()

    def _load_keys(self) -> None:
        """Load keypair from environment variables."""
        from claude1.config import SOLANA_PRIVATE_KEY, SOLANA_WALLET_ADDRESS

        if SOLANA_PRIVATE_KEY:
            self._keypair = self._parse_private_key(SOLANA_PRIVATE_KEY)
            self._public_key_str = str(self._keypair.pubkey())
        elif SOLANA_WALLET_ADDRESS:
            self._public_key_str = SOLANA_WALLET_ADDRESS
        else:
            raise ValueError(
                "No Solana wallet configured. Set SOLANA_PRIVATE_KEY or SOLANA_WALLET_ADDRESS."
            )

    @staticmethod
    def _parse_private_key(raw: str) -> Any:
        """Parse a private key from base58, JSON byte array, or hex format."""
        from solders.keypair import Keypair  # type: ignore[import-untyped]
        import base58 as b58  # type: ignore[import-untyped]

        raw = raw.strip()

        # JSON byte array: [1, 2, 3, ...]
        if raw.startswith("["):
            try:
                byte_list = json.loads(raw)
                return Keypair.from_bytes(bytes(byte_list))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid JSON byte array for private key: {exc}") from exc

        # Hex: 64-128 hex chars
        if all(c in "0123456789abcdefABCDEF" for c in raw) and len(raw) in (64, 128):
            try:
                return Keypair.from_bytes(bytes.fromhex(raw))
            except ValueError as exc:
                raise ValueError(f"Invalid hex private key: {exc}") from exc

        # Base58 (default)
        try:
            decoded = b58.b58decode(raw)
            return Keypair.from_bytes(decoded)
        except Exception as exc:
            raise ValueError(
                f"Could not decode private key (tried base58, JSON array, hex): {exc}"
            ) from exc

    @property
    def public_key(self) -> str:
        """Wallet address (public key) as a string."""
        return self._public_key_str

    @property
    def can_sign(self) -> bool:
        """True if a private key is loaded and transactions can be signed."""
        return self._keypair is not None

    def sign_transaction(self, transaction_base64: str) -> str:
        """Sign a base64-encoded VersionedTransaction and return signed base64.

        This matches Jupiter's execute flow:
        1. Decode base64 → raw bytes
        2. Deserialize into VersionedTransaction
        3. Sign with keypair
        4. Serialize → encode base64
        """
        if not self.can_sign:
            raise RuntimeError(
                "Wallet is in read-only mode (no private key). Cannot sign transactions."
            )

        from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]
        from solders.keypair import Keypair  # type: ignore[import-untyped]

        raw_bytes = base64.b64decode(transaction_base64)
        tx = VersionedTransaction.from_bytes(raw_bytes)

        # Sign the transaction
        signed_tx = VersionedTransaction(tx.message, [self._keypair])

        # Serialize and encode
        return base64.b64encode(bytes(signed_tx)).decode("ascii")

    def __repr__(self) -> str:
        pk = self._public_key_str
        truncated = f"{pk[:6]}...{pk[-4:]}" if len(pk) > 10 else pk
        mode = "full" if self.can_sign else "read-only"
        return f"SolanaWallet({truncated}, {mode})"

    def __str__(self) -> str:
        return self.__repr__()
