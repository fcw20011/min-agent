# 功能：验证 Session 管理的多会话隔离性和消息历史独立
# 设计：创建多个 session，分别发送不同消息，验证各自历史互不干扰

import pytest
from minimal_agent.session import SessionManager


def test_session_create():
    mgr = SessionManager()
    s = mgr.create("测试会话")
    assert s.id.startswith("sess-")
    assert s.title == "测试会话"
    assert s.status == "active"


def test_session_list():
    mgr = SessionManager()
    s1 = mgr.create("会话 A")
    s2 = mgr.create("会话 B")
    all_sessions = mgr.list_all()
    assert len(all_sessions) == 2
    assert {s.id for s in all_sessions} == {s1.id, s2.id}


def test_session_get():
    mgr = SessionManager()
    s = mgr.create("test")
    assert mgr.get(s.id) is s
    assert mgr.get("nonexistent") is None


def test_session_close():
    mgr = SessionManager()
    s = mgr.create("test")
    assert s.status == "active"
    mgr.close(s.id)
    assert s.status == "closed"


def test_session_isolation():
    """验证两个 session 的消息历史完全隔离"""
    mgr = SessionManager()
    s1 = mgr.create("窗口 1")
    s2 = mgr.create("窗口 2")

    # 模拟向两个 session 添加不同消息（直接操作消息列表）
    s1.messages.append({"role": "user", "content": "查天气"})
    s1.messages.append({"role": "assistant", "content": "今日晴"})

    s2.messages.append({"role": "user", "content": "写周报"})
    s2.messages.append({"role": "assistant", "content": "周报已生成"})

    # 验证隔离
    assert len(s1.messages) == 2
    assert len(s2.messages) == 2
    assert "查天气" in str(s1.messages)
    assert "写周报" in str(s2.messages)
    assert "查天气" not in str(s2.messages)
    assert "写周报" not in str(s1.messages)
