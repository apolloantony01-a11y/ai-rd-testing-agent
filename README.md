# W2 ingestion 骨架(无 Docker 可跑)

选型已定(2026-10-30, 按 W2 初稿推荐默认值):
1. Embedding: 本地 sentence-transformers, 模型默认 BAAI/bge-small-zh-v1.5; API 只作 W5 对照组。
2. Docker: 本机没有, W2 先做 ingestion + 本地向量评测, pgvector 接入放 W4; 架构上向量库是适配器, 换 pgvector 只换 Store 类。
3. 语料: 全量 M1-1~M1-7 文档+代码+测试, 外加 week1 需求画布(KListView/AlarmKit 例子作 L1/L2 补充)。week3 缺陷样本是留出测试集, 不入库(不含 week2 设计文档本身, 避免评测循环引用); 第一版误喂 week3 已纠正重跑。
4. 切分与 rerank: 文档 400-600 字、代码按函数/类、用例按条, overlap 100 字; rerank 默认本地交叉编码器, 本机未装模型时自动降级规则重排, 成绩必须标注版本。
5. L1 映射: 先用一张表(l1_mapping.json)起步, 即用例—模块—改动映射, W3 再看要不要进库。

无 Docker 的连锁影响(诚实标注):
- pgvector -> 本地 numpy 向量库(store/): 无服务端, 无需 Docker, 数据在本地文件, 换 pgvector 时接口不变。
- Redis 队列 -> W4 再定, W2 不需要队列。
- Dify 本地部署同样要 Docker, 所以 W4 的 Dify 版改为 Dify Cloud/托管试用或工作流设计导出, 不在本机装 Docker 版; 若之后装了 Docker Desktop, 可 revert 回本地 Dify + pgvector + Redis 的 Compose 方案, 架构图不变。

跑法(本会话已跑通, 见 evaluation_report.json):
  python3 ingest.py --source <模板引擎目录> --source <week1文档> --out ./store
  python3 search.py --store ./store --query "文字含单引号渲染失败"
  python3 evaluate.py --store ./store --queries queries.json --topk 5

Mac 上跑: --source 改成 /Users/thamelsu/Documents/Code/template-engine/ 即可, 代码是纯文本, 可按模板引擎的老办法同步。
store/ 是可再生成的中间产物, 不用同步到 Mac。

GitHub 同步(2026-10-30 定: Linux 验算、Mac 最终演示):
- 这个目录就是仓库根, 只推文本文件: *.py、*.json、*.md、.gitignore; store/、__pycache__、*.npz 不进仓库。
- 模板引擎语料不进这个仓库: Mac 上它已在 /Users/thamelsu/Documents/Code/template-engine/, 跑 ingestion 时用 --source 指过去即可, 避免覆盖 Mac 上 render.py 的 Hiragino 字体行、test_render.py 的 Desktop 出片段等本地改动。
- Mac 拉取后跑法:
  git pull
  python3 ingest.py --source /Users/thamelsu/Documents/Code/template-engine/ --source corpus/week1-demand-canvas-selection.md --out ./store
  注意: corpus/ 里的 week1 文档是 W2 时的语料快照, 只供 ingestion 用, 不是 live 设计文档; 设计文档本身仍在目标工作区, 改了设计要重新拷快照才生效。
  python3 search.py --store ./store --query "文字含单引号渲染失败"
  python3 evaluate.py --store ./store --queries queries.json --topk 5