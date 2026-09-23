# 跨来源事件对齐 API（Cross-source Event Alignment）

用 **FastAPI + SQLite** 构建的事件对齐服务：客户端提交两组带时间戳、来源键和载荷摘要的事件，服务在容差窗口内建立候选边，并给出**一对一、总时间差最小**的匹配；平局按事件 ID 稳定决胜；**所有未匹配事件都会保留**并单独列出。还可对每个 **来源组合 `(left.source_key, right.source_key)` 设置每组最多匹配数（容量）**，求解器会在容量约束下求全局最优；未声明的组合不限额。结果持久化在 SQLite 中，可用 Docker Compose 一键启动。

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
6. **可选来源配对容量**：请求可通过 `source_capacities` 为来源组合声明每组最多匹配数；同一三层字典序目标改为在**满足全部容量的匹配集合**中取全局最优，而不是先匹配再贪心裁剪。

求解器（`app/alignment.py`）把上述三层目标编码进一个精确整数的**最大权指派问题**：

- **未声明容量**时，用一次匈牙利算法（纯标准库、无浮点误差）得到全局最优；
- **声明了容量**时，可行匹配是三个划分拟阵（左侧事件、右侧事件、各来源组合配额）的交，无法表示成普通指派，求解器改用**精确分支限界搜索**：以匈牙利最优解为初始 incumbent，按「最受限行优先」展开，并用增广路基数界、最小费用流的「残量差 + ID 决胜」松弛界逐层剪枝，因此仍然是全局最优。该问题在大规模下属于 NP 难，容量约束请求的候选边数上限为 **60 条**（超出返回 `422`，建议收紧容差或按来源拆分）；不带容量的请求不受此限制。

## 一键启动（Docker Compose）

前置条件：已安装 Docker 与 Docker Compose v2。

```bash
docker compose up --build -d
```

- 服务端口：默认仅绑定到宿主机回环地址 **127.0.0.1:8000**；可用环境变量改端口，例如 `ALIGNMENT_PORT=8080 docker compose up --build -d`（如需对局域网开放，把 `docker-compose.yml` 里的 `127.0.0.1:` 前缀去掉）。
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
- `source_capacities` 可选：不带时行为与旧版完全一致；`summary.source_combos` 始终返回，未声明容量的组合 `capacity` 为 `null`。

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
    "total_abs_diff_us": 150000,
    "source_combos": [
      {
        "left_source": "camera-1",
        "right_source": "rfid-gate",
        "capacity": null,
        "candidate_edges": 2,
        "matched_pairs": 2,
        "rejected_by_capacity": 0
      }
    ]
  }
}
```

#### 带来源配对容量的请求

在请求体中加入可选的 `source_capacities`，为任意来源组合指定 `max_matches`（`0` 表示完全禁止该组合配对；未列出的组合不限额）：

```bash
curl -X POST http://localhost:8000/alignments \
  -H 'Content-Type: application/json' \
  -d @examples/request_with_capacity.json
```

其中 `source_capacities` 为：

```json
"source_capacities": [
  {"left_source": "camera-1", "right_source": "rfid-gate", "max_matches": 1}
]
```

上例两条候选边都属于被限流的 `(camera-1, rfid-gate)` 组合，因此全局最优只保留一条；按第 2 层「总时间差最小」，保留差更小（50000 us）的 `cam-b ↔ rfid-2`（若时间差也相同，才会在第 3 层按 ID 选更小者）。响应（节选）：

```json
{
  "pairs": [
    {
      "left_id": "cam-b",
      "right_id": "rfid-2",
      "diff_us": -50000,
      "abs_diff_us": 50000,
      "explanation": "..."
    }
  ],
  "unmatched": {
    "left": [ { "id": "cam-a", "...": "..." } ],
    "right": [ { "id": "rfid-1", "...": "..." }, { "id": "rfid-lonely", "...": "..." } ]
  },
  "summary": {
    "candidate_edges": 2,
    "matched_pairs": 1,
    "unmatched_left": 1,
    "unmatched_right": 2,
    "total_abs_diff_us": 50000,
    "source_combos": [
      {
        "left_source": "camera-1",
        "right_source": "rfid-gate",
        "capacity": 1,
        "candidate_edges": 2,
        "matched_pairs": 1,
        "rejected_by_capacity": 1
      }
    ]
  }
}
```

容量相关字段说明：

- `capacity`：该组合声明的上限；未声明为 `null`（不限额）。
- `candidate_edges`：该组合落在容差窗口内的候选边总数。
- `matched_pairs`：最终匹配中属于该组合的实际配对数（≤ `capacity`）。
- `rejected_by_capacity`：因容量被拒的候选边数——即**无约束全局最优**在该组合的配对数与容量约束结果实际配对数的净差；容量未绑定时恒为 `0`。

注意：容量约束为精确全局优化，单请求候选边数上限为 60；超限返回 HTTP `422`，请收紧 `tolerance_us` 或按 `source_key` 拆分为多个请求。重复声明同一来源组合、`max_matches < 0` 同样返回 `422`。

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

校验失败（重复 ID、负容差、无法解析的时间戳、重复或非法的来源容量声明等）返回 HTTP `422`；容量约束请求候选边超过 60 条也返回 `422`。未知 `job_id` 返回 `404`。

### 请求体字段：`source_capacities`（可选）

| 字段           | 类型    | 说明                                                                 |
| -------------- | ------- | -------------------------------------------------------------------- |
| `left_source`  | string  | 左侧事件的 `source_key`，非空                                        |
| `right_source` | string  | 右侧事件的 `source_key`，非空                                        |
| `max_matches`  | integer | 该来源组合允许的最大配对数，≥ 0；`0` 表示禁止配对                    |

- 数组内同一 `(left_source, right_source)` 只能声明一次，否则 `422`。
- 只对列出的组合生效；任何未列出的来源组合（包括来源键未在容量里出现的事件）都不限额。
- 容量是对**最终匹配**中属于该组合的配对数计数，而不是对候选边数计数。

### 配置项（环境变量）

| 变量名              | 默认值               | 作用                                   |
| ------------------- | -------------------- | -------------------------------------- |
| `ALIGNMENT_DB_PATH` | `/data/alignment.db` | SQLite 数据库文件路径（容器内挂卷）    |
| `ALIGNMENT_PORT`    | `8000`               | 宿主机映射端口（仅 Compose 使用）      |

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

- `tests/test_alignment.py`：核心算法单测（仅标准库），含边界、矩形规模、ID 决胜、顺序无关性、无约束 400 组随机用例对暴力枚举最优解的对照，以及容量约束下的 300 组随机暴力对照、零容量/未声明组合/summary 归因/规模上限等用例。
- `tests/test_api.py`：基于 TestClient 的端到端测试（建任务、持久化、校验错误、平局决胜、容量约束与来源组合 summary、重复/负容量拒绝）。

## 目录结构

```
.
├── app/
│   ├── __init__.py
│   ├── alignment.py     # 纯标准库：候选边 + 匈牙利最优 + 容量约束精确分支限界 + ID 决胜
│   ├── schemas.py       # Pydantic 请求/响应模型（含 source_capacities）、ISO-8601 时间解析
│   ├── db.py            # SQLite 连接、建表脚本（容量与来源组合统计持久化）
│   └── main.py          # FastAPI 路由与结果组装
├── tests/               # 单元测试与 API 集成测试
├── examples/
│   ├── request.json                 # 基础请求示例
│   └── request_with_capacity.json   # 带来源配对容量的请求示例
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
