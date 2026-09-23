# 跨来源事件对齐 API（Cross-source Event Alignment）

用 **FastAPI + SQLite** 构建的事件对齐服务：客户端提交两组带时间戳、来源键和载荷摘要的事件，服务在容差窗口内建立候选边，并给出**一对一、总时间差最小**的匹配；平局按事件 ID 稳定决胜；**所有未匹配事件都会保留**并单独列出。结果持久化在 SQLite 中，可用 Docker Compose 一键启动。

## 匹配语义

给定 `left`（A 组）与 `right`（B 组）事件：

1. **候选边**：仅当 `|t_left − t_right| ≤ tolerance_us` 时建立候选边（边界包含，单位为微秒）。
2. **一对一匹配**：每个事件至多出现在一对中。
3. **优化目标（按优先级字典序）**：
   1. 匹配对数最多（最大基数）；
   2. 所有配对的绝对时间差之和最小；
   3. 若仍平局，按事件 ID 稳定决胜——两侧事件各自按 ID 排序，最小化「左侧第 i 个事件所配右侧事件的 ID 名次」的字典序（未配对比任何真实名次都大）。结果与客户端提交顺序无关。
4. **未匹配事件保留**：无法配对的事件原样出现在响应的 `unmatched.left` / `unmatched.right` 中，绝不丢弃。
5. **差值解释**：每对都返回
   - `diff_us = t(left) − t(right)`（带符号，可判断谁先谁后）；
   - `abs_diff_us = |diff_us|`；
   - 人类可读的 `explanation`，例如
     `|t('cam-a') - t('rfid-1')| = 100000 us (diff = t(left) - t(right) = -100000 us; 'cam-a' is 100000 us earlier than 'rfid-1')`。

求解器（`app/alignment.py`）把上述三层目标编码进一个精确整数的**最大权指派问题**，用匈牙利算法一次求出全局最优（纯标准库实现，无浮点误差）。

## 一键启动（Docker Compose）

前置条件：已安装 Docker 与 Docker Compose v2。

```bash
docker compose up --build -d
```

- 服务端口：**8000**（映射到宿主机 `127.0.0.1:8000`，如需对外可改 `docker-compose.yml`）。
- SQLite 数据库文件：容器内 `/data/alignment.db`，持久化在 Docker 卷 `alignment-data` 中。
- 交互式文档（Swagger UI）：<http://localhost:8000/docs>
- ReDoc：<http://localhost:8000/redoc>
- 健康检查：<http://localhost:8000/health>

常用命令：

```bash
docker compose logs -f api      # 查看日志
docker compose ps               # 查看状态（含健康检查）
docker compose down             # 停止并删除容器（保留数据卷）
docker compose down -v          # 停止并同时删除数据库卷
```

## 示例请求

### 1. 创建对齐任务

```bash
curl -X POST http://localhost:8000/alignments \
  -H 'Content-Type: application/json' \
  -d '{
    "tolerance_us": 200000,
    "left": [
      {"id": "cam-a", "timestamp": "2026-09-22T10:00:00Z", "source_key": "camera-1", "payload_summary": "person entered"},
      {"id": "cam-b", "timestamp": "2026-09-22T10:05:00Z", "source_key": "camera-1", "payload_summary": "person left"}
    ],
    "right": [
      {"id": "rfid-1", "timestamp": "2026-09-22T10:00:00.100000+00:00", "source_key": "rfid-gate", "payload_summary": "badge scan"},
      {"id": "rfid-2", "timestamp": "2026-09-22T10:05:00.050000+00:00", "source_key": "rfid-gate", "payload_summary": "badge scan"},
      {"id": "rfid-lonely", "timestamp": "2026-09-22T11:00:00Z", "source_key": "rfid-gate", "payload_summary": "orphan scan"}
    ]
  }'
```

要点：

- `timestamp` 接受任意 ISO-8601 时间（含 `Z`、带偏移量、不带时区按 UTC 处理），内部统一换算为 UTC 整数微秒。
- 每组内 `id` 必须唯一；`tolerance_us` 必须 ≥ 0。
- `payload_summary` 可选，仅作为载荷摘要随事件存储/返回。

响应（HTTP `201`，节选）：

```json
{
  "job_id": "b7c955eefbbd47f580fd0feef48cca20",
  "tolerance_us": 200000,
  "pairs": [
    {
      "left_id": "cam-a",
      "right_id": "rfid-1",
      "diff_us": -100000,
      "abs_diff_us": 100000,
      "explanation": "|t('cam-a') - t('rfid-1')| = 100000 us (diff = t(left) - t(right) = -100000 us; 'cam-a' is 100000 us earlier than 'rfid-1')"
    },
    {
      "left_id": "cam-b",
      "right_id": "rfid-2",
      "diff_us": -50000,
      "abs_diff_us": 50000,
      "explanation": "..."
    }
  ],
  "unmatched": {
    "left": [],
    "right": [
      {"id": "rfid-lonely", "timestamp": "2026-09-22T11:00:00+00:00", "source_key": "rfid-gate", "payload_summary": "orphan scan"}
    ]
  },
  "summary": {
    "candidate_edges": 2,
    "matched_pairs": 2,
    "unmatched_left": 0,
    "unmatched_right": 1,
    "total_abs_diff_us": 150000
  }
}
```

### 2. 按 ID 取回任务（持久化）

```bash
curl http://localhost:8000/alignments/b7c955eefbbd47f580fd0feef48cca20
```

### 3. 列出最近的任务（最多 100 条，新的在前）

```bash
curl http://localhost:8000/jobs
```

### 4. 健康检查

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## API 一览

| 方法   | 路径                     | 说明                                     |
| ------ | ------------------------ | ---------------------------------------- |
| POST   | `/alignments`            | 提交两组事件，创建并持久化对齐任务        |
| GET    | `/alignments/{job_id}`   | 取回历史任务的完整结果                    |
| GET    | `/jobs`                  | 列出最近任务摘要                          |
| GET    | `/health`                | 存活探针                                  |
| GET    | `/docs`、`/redoc`        | 自动生成的交互文档                        |

校验失败（重复 ID、负容差、无法解析的时间戳等）返回 HTTP `422`，未知 `job_id` 返回 `404`。

## 不用 Docker 的本地运行方式

需要 Python 3.11+：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ALIGNMENT_DB_PATH=./data/alignment.db   # 默认为 /data/alignment.db
uvicorn app.main:app --reload --port 8000
```

## 测试

```bash
pip install -r requirements.txt httpx
python -m unittest discover -s tests -v
```

- `tests/test_alignment.py`：核心算法单测（仅标准库），含边界、矩形规模、ID 决胜、顺序无关性，以及 400 组随机用例对暴力枚举最优解的对照。
- `tests/test_api.py`：基于 TestClient 的端到端测试（建任务、持久化、校验错误、平局决胜）。

## 目录结构

```
.
├── app/
│   ├── __init__.py
│   ├── alignment.py     # 纯标准库：候选边 + 匈牙利全局最优 + ID 决胜
│   ├── schemas.py       # Pydantic 请求/响应模型、ISO-8601 时间解析
│   ├── db.py            # SQLite 连接、建表脚本
│   └── main.py          # FastAPI 路由与结果组装
├── tests/               # 单元测试与 API 集成测试
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
