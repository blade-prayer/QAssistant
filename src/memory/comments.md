## 共享记忆不仅仅只实现“对象去重”，而是进一步实现为“金融证据管理”

理论上每条record可以追溯到：agent，task，tool，当时的query，URL，source和写入时间


## 优化DataCollector和DeepSearchAgent的相关prompt

- src/agents/search_agent/prompts/general_prompts.yaml： 明确金融研究来源优先级：公告/交易所/监管/公司 IR > 权威数据源 > 主流财经媒体/研究机构 > 搜索摘要。

- src/agents/data_collector/prompts/prompts.yaml： 明确工具选择优先级：结构化金融/宏观/行业 API 优先，DeepSearch 用于非结构化证据、事件、竞争格局、政策、管理层等。