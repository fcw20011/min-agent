# 功能：验证 ReadFileTool 的读取、错误处理、行号偏移和截断
# 设计：用临时文件构造不同场景，覆盖正常、不存在、过大、偏移等情况

import os
import tempfile

import pytest

from minimal_agent.tools.read_file import ReadFileTool


@pytest.mark.asyncio
async def test_read_file_basic():
    tool = ReadFileTool()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("line 1\nline 2\nline 3\n")
        tmp = f.name

    try:
        result = await tool.execute({"path": tmp})
        assert not result.is_error
        assert "line 1" in result.content
        assert "line 2" in result.content
    finally:
        os.unlink(tmp)


@pytest.mark.asyncio
async def test_read_file_not_found():
    tool = ReadFileTool()
    result = await tool.execute({"path": "/nonexistent/file.txt"})
    assert result.is_error
    assert "不存在" in result.content


@pytest.mark.asyncio
async def test_read_file_with_offset():
    tool = ReadFileTool()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("a\nb\nc\nd\ne\n")
        tmp = f.name

    try:
        result = await tool.execute({"path": tmp, "offset": 3})
        assert not result.is_error
        assert "line 1" not in result.content  # offset 后的行号从 3 开始
        assert "c" in result.content
        assert "d" in result.content
    finally:
        os.unlink(tmp)


@pytest.mark.asyncio
async def test_read_file_with_limit():
    tool = ReadFileTool()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("1\n2\n3\n4\n5\n")
        tmp = f.name

    try:
        result = await tool.execute({"path": tmp, "limit": 2})
        assert not result.is_error
        assert "1" in result.content
        assert "2" in result.content
        assert "3" not in result.content
    finally:
        os.unlink(tmp)


@pytest.mark.asyncio
async def test_read_file_empty():
    tool = ReadFileTool()
    result = await tool.execute({"path": ""})
    assert result.is_error
