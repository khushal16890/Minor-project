"""
Unit tests for automated mitigation engine and firewall rules.
"""
import pytest
from mitigation.nftables import NftablesManager
from backend.core.config import MitigationConfig


def test_nftables_simulation_block_and_unblock():
    manager = NftablesManager()
    target_ip = "192.168.1.200"

    # Block IP
    res_block = manager.block_ip(
        ip=target_ip,
        reason="Test unit block",
        duration_minutes=30
    )
    assert res_block["success"] is True
    assert res_block.get("simulated") is True

    # Duplicate block prevention
    res_dup = manager.block_ip(
        ip=target_ip,
        reason="Duplicate test"
    )
    assert res_dup["success"] is False or "already blocked" in res_dup.get("message", "").lower()

    # Unblock IP
    res_unblock = manager.unblock_ip(
        ip=target_ip,
        reason="Test unblock"
    )
    assert res_unblock["success"] is True


def test_nftables_protected_ips():
    manager = NftablesManager()
    # Cannot block loopback or protected
    res = manager.block_ip(ip="127.0.0.1", reason="Malicious loopback")
    assert res["success"] is False
    assert "protected" in res.get("message", "").lower()


def test_nftables_invalid_ip():
    manager = NftablesManager()
    res = manager.block_ip(ip="invalid.ip.string", reason="Bad IP")
    assert res["success"] is False
    assert "invalid" in res.get("message", "").lower()
