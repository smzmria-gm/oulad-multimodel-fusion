# 数据说明 / Data

OULAD（Open University Learning Analytics Dataset）数据文件**不包含**在本仓库中，
请从官方页面下载并解压到本目录：

- 官方页面：https://analyse.kmi.open.ac.uk/open_dataset
- 数据论文：Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics Dataset //
  Scientific Data. 2017. Vol. 4. 170171. DOI: 10.1038/sdata.2017.103

## 实验所需文件（解压后应位于本目录）

| 文件 | 用途 | 说明 |
|---|---|---|
| `studentInfo.csv` | 标签与静态特征 | 32 593 条选课注册记录，含 `final_result` |
| `studentRegistration.csv` | 注册日期特征 | 含 `date_registration` |
| `assessments.csv` | 测评元数据 | 与 `studentAssessment.csv` 关联 |
| `studentAssessment.csv` | 测评成绩特征 | 含 `score`、`date_submitted` |
| `studentVle.csv` | VLE 行为特征 | 约 454 MB / 1 065 万行，脚本按块聚合 |
| `vle.csv` | （备用）VLE 资源元数据 | 当前管线未直接使用 |
| `courses.csv` | （备用）课程元数据 | 当前管线未直接使用 |

解压后总大小约 460 MB。本目录下的 `*.csv` / `*.zip` 已在 `.gitignore` 中排除，不会被提交。
