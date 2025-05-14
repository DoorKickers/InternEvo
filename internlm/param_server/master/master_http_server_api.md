# master http server接口文档

## 1. `/global_status`

**方法**：GET  
**描述**：获取 master 节点当前状态及历史状态记录。

### 请求参数

| 参数名       | 类型   | 是否必传 | 默认值 | 描述                       |
|------------|--------|--------|------|--------------------------|
| last_of_n  | int    | 否     | 10   | 返回最近 last_of_n 条状态记录（按时间倒序） |

### 响应字段


| 字段名                     | 类型            | 描述             |
| -------------------------- | --------------- | ---------------- |
| current_global_status      | string          | 当前 master 状态 |
| global_status_history_list | list of objects | 状态变更历史     |

### global_status_history_list 子字段：

| 字段名          | 类型   | 描述     |
| --------------- | ------ | -------- |
| status_name     | string | 状态名称 |
| begin_timestamp | float  | 起始时间 |
| end_timestamp   | float  | 结束时间 |

---

## 2. `/alive_ps`

**方法**：GET  
**描述**：返回存活的 PS 节点编号列表。

### 响应字段

| 字段名   | 类型       | 描述               |
| -------- | ---------- | ------------------ |
| alive_ps | array[int] | 存活的 PS 节点编号 |

---

## 3. `/alive_group`

**方法**：GET  
**描述**：返回当前存活的 group 编号列表。

### 响应字段

| 字段名      | 类型      | 描述              |
| ----------- | --------- | ----------------- |
| alive_group | List[int] | 存活的 group 编号 |

---

## 4. `/group_status`

**方法**：GET  
**描述**：返回所有 group 的状态、事件历史和训练指标。

### 请求参数

| 参数名       | 类型   | 是否必传 | 默认值 | 描述                       |
|------------|--------|--------|------|--------------------------|
| last_of_n  | int    | 否     | 10   | 返回最近 last_of_n 条状态记录（按时间倒序） |

### 响应字段

每个 group 为一个 object，字段如下：

| 字段名               | 类型            | 描述                   |
| -------------------- | --------------- | ---------------------- |
| history_status_list  | List of objects | 历史事件记录           |
| current_status       | string          | 当前状态               |
| normal_ratio         | float           | 正常状态比例           |
| last_training_metric | string          | 最近训练指标摘要字符串 |

### history_status_list 子字段：

| 字段名          | 类型   | 描述         |
| --------------- | ------ | ------------ |
| event_name      | string | 事件名称     |
| start_timestamp | float  | 事件起始时间 |
| end_timestamp   | float  | 事件结束时间 |

