"""Tests for position tracking, reconciliation, and data model fixes."""

import json
from pathlib import Path

import pytest

from claude1.trader.positions import (
    Position,
    PositionManager,
    PositionSide,
    PositionStatus,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def pm(tmp_path: Path) -> PositionManager:
    """PositionManager backed by a temp file."""
    return PositionManager(positions_path=tmp_path / "positions.json")


# ── Position data-model tests ────────────────────────────────────────────────


class TestPositionDataModel:
    """Tests for the Position dataclass additions."""

    def test_verified_field_default(self):
        """New 'verified' field defaults to False."""
        pos = Position(token="sol", entry_price=100, amount=1.0)
        assert pos.verified is False

    def test_verified_field_set_true(self):
        pos = Position(token="sol", entry_price=100, amount=1.0)
        pos.verified = True
        assert pos.verified is True

    def test_verified_roundtrip(self, tmp_path: Path):
        """verified field survives save/load cycle."""
        pm = PositionManager(positions_path=tmp_path / "pos.json")
        pos = pm.create(token="bonk", side="long", entry_price=0.00002, amount=1000)
        pos.verified = True
        pm._save()

        pm2 = PositionManager(positions_path=tmp_path / "pos.json")
        reloaded = pm2.get(pos.id)
        assert reloaded is not None
        assert reloaded.verified is True

    def test_backwards_compat_no_verified_field(self, tmp_path: Path):
        """Loading old JSON without 'verified' field still works."""
        path = tmp_path / "old.json"
        old_data = [{
            "id": "abc123",
            "token": "sol",
            "side": "long",
            "status": "open",
            "entry_price": 100.0,
            "current_price": 105.0,
            "amount": 1.0,
            "cost_basis": 100.0,
            "created_at": "2025-01-01T00:00:00+00:00",
        }]
        path.write_text(json.dumps(old_data))
        pm = PositionManager(positions_path=path)
        pos = pm.get("abc123")
        assert pos is not None
        assert pos.verified is False  # default

    def test_summary_long_no_paper_tag(self):
        pos = Position(token="sol", side="long", status="open",
                       entry_price=100.0, current_price=110.0, amount=1.0)
        s = pos.summary()
        assert "[PAPER]" not in s
        assert "sol" in s

    def test_summary_short_has_paper_tag(self):
        pos = Position(token="sol", side="short", status="open",
                       entry_price=100.0, current_price=90.0, amount=1.0)
        s = pos.summary()
        assert "[PAPER]" in s


# ── PositionManager aggregate tests ──────────────────────────────────────────


class TestTotalInvested:
    """total_invested() should exclude short positions."""

    def test_long_only(self, pm: PositionManager):
        pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.create(token="bonk", side="long", entry_price=0.00002, amount=50000)
        assert pm.total_invested() == pytest.approx(100 + 1.0, abs=0.01)

    def test_short_excluded(self, pm: PositionManager):
        pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.create(token="eth", side="short", entry_price=3000, amount=0.5)
        # Only the long position should count
        assert pm.total_invested() == pytest.approx(100.0, abs=0.01)

    def test_only_shorts(self, pm: PositionManager):
        pm.create(token="sol", side="short", entry_price=100, amount=1.0)
        assert pm.total_invested() == 0.0

    def test_empty(self, pm: PositionManager):
        assert pm.total_invested() == 0.0

    def test_closed_positions_excluded(self, pm: PositionManager):
        pos = pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.open_position(pos.id, entry_price=100, amount=1.0)
        pm.close_position(pos.id, exit_price=110)
        assert pm.total_invested() == 0.0


# ── Bulk price update tests ──────────────────────────────────────────────────


class TestUpdatePricesBulk:
    def test_updates_active_positions(self, pm: PositionManager):
        pos1 = pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.open_position(pos1.id, entry_price=100, amount=1.0)
        pos2 = pm.create(token="bonk", side="long", entry_price=0.00002, amount=50000)
        pm.open_position(pos2.id, entry_price=0.00002, amount=50000)

        count = pm.update_prices_bulk({"sol": 110.0, "bonk": 0.000025})
        assert count == 2

        assert pm.get(pos1.id).current_price == pytest.approx(110.0)
        assert pm.get(pos2.id).current_price == pytest.approx(0.000025)

    def test_skips_closed_positions(self, pm: PositionManager):
        pos = pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.open_position(pos.id, entry_price=100, amount=1.0)
        pm.close_position(pos.id, exit_price=110)

        count = pm.update_prices_bulk({"sol": 120.0})
        assert count == 0
        # closed price should not change
        assert pm.get(pos.id).current_price == pytest.approx(110.0)

    def test_unknown_token_ignored(self, pm: PositionManager):
        pos = pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.open_position(pos.id, entry_price=100, amount=1.0)

        count = pm.update_prices_bulk({"eth": 3000.0})
        assert count == 0
        assert pm.get(pos.id).current_price == pytest.approx(100.0)

    def test_zero_price_ignored(self, pm: PositionManager):
        pos = pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.open_position(pos.id, entry_price=100, amount=1.0)

        count = pm.update_prices_bulk({"sol": 0.0})
        assert count == 0

    def test_empty_dict(self, pm: PositionManager):
        pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        count = pm.update_prices_bulk({})
        assert count == 0

    def test_single_save_call(self, pm: PositionManager, monkeypatch):
        """Bulk update should call _save() exactly once."""
        pos = pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.open_position(pos.id, entry_price=100, amount=1.0)

        save_calls = []
        original_save = pm._save
        def tracked_save():
            save_calls.append(1)
            original_save()
        monkeypatch.setattr(pm, "_save", tracked_save)

        pm.update_prices_bulk({"sol": 110.0})
        assert len(save_calls) == 1


# ── Risk manager short bypass tests ─────────────────────────────────────────


class TestRiskManagerShortBypass:
    """Portfolio capacity check should be skipped for short positions."""

    def test_short_bypasses_portfolio_cap(self, pm: PositionManager):
        from claude1.trader.risk_manager import RiskManager

        # Fill up portfolio with a long position
        pos = pm.create(token="sol", side="long", entry_price=50, amount=1.8)
        pm.open_position(pos.id, entry_price=50, amount=1.8)

        rm = RiskManager(
            position_manager=pm,
            max_portfolio_size=100.0,
            max_position_size=25.0,
            min_risk_reward=1.0,
        )

        # Long trade should fail (portfolio nearly full)
        check_long = rm.check_trade(
            token="bonk", side="long", entry_price=0.00002,
            target_price=0.00004, stop_price=0.00001, amount=600000,
        )
        assert not check_long.approved
        assert "portfolio_cap" in " ".join(check_long.checks_failed)

        # Short trade should bypass portfolio cap check
        check_short = rm.check_trade(
            token="bonk", side="short", entry_price=0.00002,
            target_price=0.00001, stop_price=0.000025, amount=100000,
        )
        # Should not fail on portfolio_cap
        assert "portfolio_cap" not in " ".join(check_short.checks_failed)


# ── Reconcile logic tests ────────────────────────────────────────────────────


class TestReconcileLogic:
    """Test the phantom-detection and reconciliation logic."""

    def test_phantom_detection(self, pm: PositionManager):
        """A long position whose token is absent from wallet is phantom."""
        pos = pm.create(token="bonk", side="long", entry_price=0.00002, amount=50000)
        pm.open_position(pos.id, entry_price=0.00002, amount=50000)

        # Simulate wallet with only SOL
        holdings = {"sol": 1.0}
        active = pm.list_active()

        phantoms = []
        for p in active:
            if p.side == PositionSide.LONG.value and holdings.get(p.token.lower(), 0.0) <= 0:
                phantoms.append(p)

        assert len(phantoms) == 1
        assert phantoms[0].id == pos.id

    def test_short_not_phantom(self, pm: PositionManager):
        """Short positions are paper-only and never flagged as phantoms."""
        pos = pm.create(token="bonk", side="short", entry_price=0.00002, amount=50000)
        pm.open_position(pos.id, entry_price=0.00002, amount=50000)

        holdings = {"sol": 1.0}
        active = pm.list_active()

        phantoms = [
            p for p in active
            if p.side == PositionSide.LONG.value
            and holdings.get(p.token.lower(), 0.0) <= 0
        ]
        assert len(phantoms) == 0

    def test_aggregate_phantom_multiple_positions_same_token(self, pm: PositionManager):
        """3 SOL LONGs totaling 0.77 with wallet holding 0.23 → newest flagged as phantom."""
        from claude1.tools.trader_tools import PositionStatusTool

        # Create 3 positions with different timestamps (oldest to newest)
        p1 = pm.create(token="sol", side="long", entry_price=86.0, amount=0.20)
        pm.open_position(p1.id, entry_price=86.0, amount=0.20)
        pm.get(p1.id).opened_at = "2025-01-01T10:00:00+00:00"
        pm._save()

        p2 = pm.create(token="sol", side="long", entry_price=86.0, amount=0.27)
        pm.open_position(p2.id, entry_price=86.0, amount=0.27)
        pm.get(p2.id).opened_at = "2025-01-01T11:00:00+00:00"
        pm._save()

        p3 = pm.create(token="sol", side="long", entry_price=86.0, amount=0.30)
        pm.open_position(p3.id, entry_price=86.0, amount=0.30)
        pm.get(p3.id).opened_at = "2025-01-01T12:00:00+00:00"
        pm._save()

        active = pm.list_active()
        holdings = {"sol": 0.23}

        phantom_ids, phantom_capital = PositionStatusTool._compute_phantom_ids(active, holdings)

        # Total tracked = 0.77, held = 0.23, excess = 0.54
        # Newest first: p3 (0.30), p2 (0.27) → accumulated 0.57 >= 0.54
        assert p3.id in phantom_ids
        assert p2.id in phantom_ids
        assert p1.id not in phantom_ids  # oldest, kept
        assert phantom_capital > 0

    def test_single_position_matching_wallet(self, pm: PositionManager):
        """Single position that matches wallet balance → not phantom."""
        from claude1.tools.trader_tools import PositionStatusTool

        pos = pm.create(token="sol", side="long", entry_price=86.0, amount=0.50)
        pm.open_position(pos.id, entry_price=86.0, amount=0.50)

        active = pm.list_active()
        holdings = {"sol": 0.50}

        phantom_ids, phantom_capital = PositionStatusTool._compute_phantom_ids(active, holdings)
        assert len(phantom_ids) == 0
        assert phantom_capital == 0.0

    def test_token_not_in_wallet_all_phantom(self, pm: PositionManager):
        """Positions for a token the wallet doesn't hold at all → all flagged."""
        from claude1.tools.trader_tools import PositionStatusTool

        p1 = pm.create(token="bonk", side="long", entry_price=0.00002, amount=50000)
        pm.open_position(p1.id, entry_price=0.00002, amount=50000)

        p2 = pm.create(token="bonk", side="long", entry_price=0.00002, amount=30000)
        pm.open_position(p2.id, entry_price=0.00002, amount=30000)

        active = pm.list_active()
        holdings = {"sol": 1.0}  # No bonk at all

        phantom_ids, phantom_capital = PositionStatusTool._compute_phantom_ids(active, holdings)
        assert p1.id in phantom_ids
        assert p2.id in phantom_ids
        assert phantom_capital > 0

    def test_tolerance_allows_small_mismatch(self, pm: PositionManager):
        """Tracked slightly exceeds held but within 20% tolerance → not phantom."""
        from claude1.tools.trader_tools import PositionStatusTool

        pos = pm.create(token="sol", side="long", entry_price=86.0, amount=1.0)
        pm.open_position(pos.id, entry_price=86.0, amount=1.0)

        active = pm.list_active()
        # held * 1.2 = 1.08, tracked = 1.0 → 1.0 <= 1.08, no phantom
        holdings = {"sol": 0.90}

        phantom_ids, _ = PositionStatusTool._compute_phantom_ids(active, holdings)
        assert len(phantom_ids) == 0

    def test_cancel_frees_capacity(self, pm: PositionManager):
        """Cancelling a phantom long position frees portfolio capacity."""
        pos = pm.create(token="sol", side="long", entry_price=50, amount=1.0)
        pm.open_position(pos.id, entry_price=50, amount=1.0)

        assert pm.total_invested() == pytest.approx(50.0)

        pm.cancel(pos.id, notes="phantom")
        assert pm.total_invested() == 0.0


# ── PnL accuracy tests ──────────────────────────────────────────────────────


class TestPnLAccuracy:
    """P&L should reflect updated prices after bulk update."""

    def test_pnl_after_bulk_update(self, pm: PositionManager):
        pos = pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.open_position(pos.id, entry_price=100, amount=1.0)

        # Before update: current_price == entry_price, so P&L == 0
        assert pm.get(pos.id).pnl == pytest.approx(0.0)

        pm.update_prices_bulk({"sol": 110.0})
        assert pm.get(pos.id).pnl == pytest.approx(10.0)  # 1.0 * (110 - 100)
        assert pm.get(pos.id).pnl_pct == pytest.approx(10.0)  # 10%

    def test_pnl_negative(self, pm: PositionManager):
        pos = pm.create(token="sol", side="long", entry_price=100, amount=1.0)
        pm.open_position(pos.id, entry_price=100, amount=1.0)

        pm.update_prices_bulk({"sol": 90.0})
        assert pm.get(pos.id).pnl == pytest.approx(-10.0)
