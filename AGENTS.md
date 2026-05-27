# AGENTS.md

## Code Style

- 生成 Python 代码时，确保括号/引号/f-string 闭合正确；避免嵌套同类型引号；复杂 f-string 先拆分变量，避免内联条件表达式导致语法错误

- Python 导入遵循 PEP8：
  stdlib → third-party → local；
  分组空行；组内按字母排序

- 前端导入顺序：
  third-party → local → styles；
  样式导入放最后

- 当前项目部署在远程服务器, 本地不具备完整运行环境

## Pitfalls

- 暂无

## Best Practices

- 长文件生成时，建议分块写入，避免输出超出Token限制而导致内容截断
- PowerShell 不支持`&&`, 需要使用 PowerShell 语法。
