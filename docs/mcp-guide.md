# MCP 工具配置完整指南

## 目录
- [1. MCP 工具概述](#1-mcp-工具概述)
- [2. MCP 服务器开发](#2-mcp-服务器开发)
- [3. 配置文件详解](#3-配置文件详解)
- [4. 启动与连接](#4-启动与连接)
- [5. Agent 识别机制](#5-agent-识别机制)
- [6. 完整实践示例](#6-完整实践示例)

---

## 1. MCP 工具概述

**MCP (Model Context Protocol)** 是一个开放协议，允许 AI 助手（如 Claude）与外部工具和数据源进行标准化交互。

### 1.1 核心组件
- **MCP Server**: 提供工具和资源的服务器程序
- **MCP Client**: 连接到服务器的客户端（如 Claude Desktop）
- **Tools**: 服务器暴露给 AI 的功能接口
- **Resources**: 服务器提供的数据资源

### 1.2 工作流程
```
User Request -> Claude -> MCP Client -> MCP Server -> External API/Service
                  ^                           |
                  |                           |
                  +------ Response -----------+
```

---

## 2. MCP 服务器开发

### 2.1 技术栈选择
MCP 服务器可以使用多种语言开发：
- **Node.js/TypeScript** (官方 SDK)
- **Python** (社区 SDK)
- **Go, Rust** (社区实现)

### 2.2 Node.js 示例源代码

#### 2.2.1 项目初始化
```bash
mkdir mcp-github-server
cd mcp-github-server
npm init -y
npm install @modelcontextprotocol/sdk zod
npm install -D @types/node typescript
```

#### 2.2.2 package.json
```json
{
  "name": "mcp-github-server",
  "version": "1.0.0",
  "type": "module",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "@octokit/rest": "^20.0.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0"
  }
}
```

#### 2.2.3 tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "Node16",
    "moduleResolution": "Node16",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

#### 2.2.4 核心服务器代码 (src/index.ts)
```typescript
#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { Octokit } from "@octokit/rest";
import { z } from "zod";

// 初始化 GitHub 客户端
const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN,
});

// 定义工具
const TOOLS: Tool[] = [
  {
    name: "github_create_issue",
    description: "在 GitHub 仓库中创建新的 Issue",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "仓库所有者",
        },
        repo: {
          type: "string",
          description: "仓库名称",
        },
        title: {
          type: "string",
          description: "Issue 标题",
        },
        body: {
          type: "string",
          description: "Issue 内容",
        },
      },
      required: ["owner", "repo", "title"],
    },
  },
  {
    name: "github_list_repos",
    description: "列出用户的所有仓库",
    inputSchema: {
      type: "object",
      properties: {
        username: {
          type: "string",
          description: "GitHub 用户名",
        },
      },
      required: ["username"],
    },
  },
];

// 工具执行逻辑
async function handleToolCall(name: string, args: any) {
  switch (name) {
    case "github_create_issue": {
      const { owner, repo, title, body } = args;
      const response = await octokit.issues.create({
        owner,
        repo,
        title,
        body: body || "",
      });
      return {
        content: [
          {
            type: "text",
            text: `Issue 创建成功: ${response.data.html_url}`,
          },
        ],
      };
    }

    case "github_list_repos": {
      const { username } = args;
      const response = await octokit.repos.listForUser({
        username,
        per_page: 10,
      });
      const repos = response.data.map((repo) => ({
        name: repo.name,
        url: repo.html_url,
        description: repo.description,
      }));
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(repos, null, 2),
          },
        ],
      };
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// 初始化服务器
async function main() {
  const server = new Server(
    {
      name: "mcp-github-server",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // 处理工具列表请求
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return { tools: TOOLS };
  });

  // 处理工具调用请求
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    return await handleToolCall(name, args || {});
  });

  // 启动 stdio 传输
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("MCP GitHub Server running on stdio");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
```

#### 2.2.5 构建
```bash
npm run build
```

---

## 3. 配置文件详解

### 3.1 配置文件位置

MCP 服务器配置存储在 Claude Desktop 的配置文件中：

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Linux**: `~/.config/Claude/claude_desktop_config.json`

### 3.2 配置文件结构

```json
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": [
        "/absolute/path/to/mcp-github-server/dist/index.js"
      ],
      "env": {
        "GITHUB_TOKEN": "ghp_your_github_token_here"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/projects"
      ]
    },
    "custom-python-server": {
      "command": "python",
      "args": [
        "/path/to/your/mcp_server.py"
      ],
      "env": {
        "API_KEY": "your_api_key"
      }
    }
  }
}
```

### 3.3 配置字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `mcpServers` | object | 是 | 所有 MCP 服务器的配置容器 |
| `<server-name>` | object | 是 | 服务器唯一标识符（自定义名称） |
| `command` | string | 是 | 启动服务器的命令（如 node, python, npx） |
| `args` | array | 是 | 传递给命令的参数列表 |
| `env` | object | 否 | 环境变量配置（API keys, tokens 等） |

### 3.4 完整配置示例

```json
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": [
        "/Users/username/mcp-servers/github/dist/index.js"
      ],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxx"
      }
    },
    "database": {
      "command": "node",
      "args": [
        "/Users/username/mcp-servers/database/dist/index.js"
      ],
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_USER": "admin",
        "DB_PASSWORD": "secret"
      }
    },
    "weather": {
      "command": "python3",
      "args": [
        "/Users/username/mcp-servers/weather/server.py"
      ],
      "env": {
        "WEATHER_API_KEY": "your_api_key"
      }
    }
  }
}
```

---

## 4. 启动与连接

### 4.1 启动流程

1. **启动 Claude Desktop**
   - Claude Desktop 读取 `claude_desktop_config.json`
   - 解析 `mcpServers` 配置

2. **自动启动 MCP 服务器**
   - 对每个配置的服务器执行 `command` + `args`
   - 注入 `env` 环境变量
   - 通过 stdio (标准输入/输出) 建立通信

3. **协议握手**
   ```
   Claude Desktop (Client) -> MCP Server: Initialize Request
   MCP Server -> Claude Desktop: Initialize Response (capabilities)
   Claude Desktop -> MCP Server: List Tools Request
   MCP Server -> Claude Desktop: Tools List
   ```

4. **就绪状态**
   - 所有工具已加载到 Claude 的上下文
   - 用户可以开始使用

### 4.2 验证连接

在 Claude Desktop 中：
- 打开设置 (Settings)
- 查看 Developer 标签
- 检查 MCP Servers 状态
- 确认工具列表是否显示

### 4.3 调试方法

#### 查看日志
**macOS/Linux**:
```bash
tail -f ~/Library/Logs/Claude/mcp*.log
```

**Windows**:
```powershell
Get-Content $env:APPDATA\Claude\logs\mcp*.log -Wait
```

#### 手动测试服务器
```bash
cd /path/to/mcp-server
node dist/index.js
# 服务器应该启动并等待 stdio 输入
```

#### 常见问题
- **服务器未启动**: 检查 `command` 路径是否正确
- **工具未显示**: 检查服务器日志，确认 `ListToolsRequestSchema` 处理正确
- **环境变量错误**: 确认 `env` 配置和服务器代码中的变量名匹配

---

## 5. Agent 识别机制

### 5.1 工具发现过程

Claude 通过以下机制识别可用的 MCP 工具：

```
1. 启动时加载
   Claude Desktop -> 读取配置 -> 启动所有 MCP 服务器

2. 工具注册
   Claude -> 向每个服务器发送 ListToolsRequest
   Server -> 返回工具列表（名称、描述、参数 schema）

3. 上下文构建
   Claude -> 将所有工具添加到内部函数列表
   Claude -> 在系统提示中包含工具说明
```

### 5.2 显式调用

用户明确指定要使用某个 MCP 工具：

**示例 1: 直接命名**
```
用户: "使用 github_create_issue 工具在我的 repo 创建一个 issue"
```

Claude 识别流程：
1. 解析用户意图，识别工具名称 `github_create_issue`
2. 检查该工具是否在可用列表中
3. 提取或询问必需参数 (owner, repo, title)
4. 构造工具调用请求
5. 发送到对应的 MCP 服务器

**示例 2: 功能描述**
```
用户: "帮我在 GitHub 上创建一个 issue，标题是'修复登录 bug'"
```

Claude 识别流程：
1. 理解用户意图："创建 GitHub issue"
2. 搜索工具列表，匹配描述关键词
3. 找到 `github_create_issue` 工具
4. 执行调用

### 5.3 隐式调用

用户描述任务，Claude 自动选择合适的 MCP 工具：

**示例 1: 任务导向**
```
用户: "列出我的所有 GitHub 仓库"
```

Claude 推理过程：
```
1. 任务分析: 需要获取 GitHub 仓库列表
2. 工具搜索: 
   - 关键词: "list", "repositories", "GitHub"
   - 匹配工具: github_list_repos
3. 参数推断:
   - 需要 username
   - 从上下文或询问用户获取
4. 执行调用
```

**示例 2: 多步骤任务**
```
用户: "帮我检查 myproject 仓库的 issues，然后创建一个新的 issue"
```

Claude 执行流程：
```
1. 分解任务:
   - 子任务 1: 获取 issues 列表
   - 子任务 2: 创建新 issue

2. 工具映射:
   - 子任务 1 -> github_list_issues
   - 子任务 2 -> github_create_issue

3. 顺序执行:
   - 调用 github_list_issues
   - 展示结果给用户
   - 调用 github_create_issue
```

### 5.4 工具选择算法

Claude 使用以下因素选择 MCP 工具：

1. **语义匹配**
   - 工具名称与用户请求的相关性
   - 工具描述与任务的匹配度

2. **参数可用性**
   - 必需参数是否可从上下文获取
   - 是否需要询问用户

3. **工具能力**
   - 工具的 inputSchema 是否满足需求
   - 工具的输出是否符合预期

4. **优先级规则**
   - 专用工具优先于通用工具
   - MCP 工具优先于内置功能
   - 最近成功使用的工具权重更高

### 5.5 工具调用示例

#### 内部调用流程
```json
{
  "tool_name": "github_create_issue",
  "arguments": {
    "owner": "pzzzzzzy",
    "repo": "interview",
    "title": "添加 MCP 配置文档",
    "body": "需要创建一个完整的 MCP 工具配置指南"
  }
}
```

#### 服务器响应
```json
{
  "content": [
    {
      "type": "text",
      "text": "Issue 创建成功: https://github.com/pzzzzzzy/interview/issues/1"
    }
  ]
}
```

---

## 6. 完整实践示例

### 6.1 场景: 构建 Notion MCP 服务器

#### 步骤 1: 创建项目
```bash
mkdir mcp-notion-server
cd mcp-notion-server
npm init -y
npm install @modelcontextprotocol/sdk @notionhq/client zod
npm install -D typescript @types/node
npx tsc --init
```

#### 步骤 2: 编写服务器 (src/index.ts)
```typescript
#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { Client } from "@notionhq/client";

const notion = new Client({ auth: process.env.NOTION_API_KEY });

const server = new Server(
  { name: "mcp-notion-server", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "notion_create_page",
      description: "在 Notion 数据库中创建新页面",
      inputSchema: {
        type: "object",
        properties: {
          database_id: { type: "string" },
          title: { type: "string" },
          content: { type: "string" },
        },
        required: ["database_id", "title"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  if (name === "notion_create_page") {
    const { database_id, title, content } = args as any;
    const response = await notion.pages.create({
      parent: { database_id },
      properties: {
        Name: { title: [{ text: { content: title } }] },
      },
      children: content ? [
        { object: "block", type: "paragraph", paragraph: { rich_text: [{ text: { content } }] } }
      ] : [],
    });
    
    return {
      content: [{ type: "text", text: `页面创建成功: ${response.url}` }],
    };
  }
  
  throw new Error(`Unknown tool: ${name}`);
});

const transport = new StdioServerTransport();
server.connect(transport);
```

#### 步骤 3: 构建
```bash
npm run build
```

#### 步骤 4: 配置 Claude Desktop
编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "notion": {
      "command": "node",
      "args": [
        "/Users/username/mcp-notion-server/dist/index.js"
      ],
      "env": {
        "NOTION_API_KEY": "secret_xxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

#### 步骤 5: 重启 Claude Desktop
- 完全退出 Claude Desktop (Cmd+Q on macOS)
- 重新启动应用

#### 步骤 6: 测试使用

**显式调用**:
```
用户: "使用 notion_create_page 工具在数据库 abc123 中创建标题为'会议记录'的页面"
```

**隐式调用**:
```
用户: "帮我在 Notion 创建一个新页面，标题是'项目计划'，内容是'Q1 目标和里程碑'"
```

Claude 会自动：
1. 识别需要使用 `notion_create_page` 工具
2. 提取参数或询问缺失的 `database_id`
3. 执行调用并返回结果

---

## 7. 最佳实践

### 7.1 安全性
- **永远不要在配置文件中硬编码密钥**
- 使用环境变量管理敏感信息
- 定期轮换 API tokens
- 限制 MCP 服务器的权限范围

### 7.2 错误处理
```typescript
async function handleToolCall(name: string, args: any) {
  try {
    // 工具逻辑
    return { content: [{ type: "text", text: "成功" }] };
  } catch (error) {
    return {
      content: [{
        type: "text",
        text: `错误: ${error.message}`
      }],
      isError: true,
    };
  }
}
```

### 7.3 参数验证
使用 Zod 进行运行时验证：
```typescript
import { z } from "zod";

const CreateIssueSchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
  title: z.string().min(1),
  body: z.string().optional(),
});

// 在工具处理中
const validated = CreateIssueSchema.parse(args);
```

### 7.4 日志记录
```typescript
console.error(`[MCP Server] Tool called: ${name}`);
console.error(`[MCP Server] Arguments:`, JSON.stringify(args));
```

---

## 8. 常见问题

### Q1: 工具修改后如何重新加载?
**A**: 重启 Claude Desktop 应用，服务器会自动重新启动。

### Q2: 可以同时运行多个 MCP 服务器吗?
**A**: 可以，在配置文件中添加多个服务器配置即可。

### Q3: 如何调试 MCP 服务器?
**A**: 
- 查看 `~/Library/Logs/Claude/mcp*.log`
- 在服务器代码中使用 `console.error()` 输出日志
- 手动运行服务器进行测试

### Q4: MCP 工具调用失败怎么办?
**A**: 
1. 检查服务器是否正常启动
2. 验证配置文件语法
3. 确认环境变量正确设置
4. 查看错误日志

---

## 9. 参考资源

- **官方文档**: https://modelcontextprotocol.io
- **SDK 仓库**: https://github.com/modelcontextprotocol/sdk
- **社区服务器**: https://github.com/modelcontextprotocol/servers
- **示例项目**: https://github.com/modelcontextprotocol/examples

---

**文档版本**: 1.0.0  
**最后更新**: 2026-09-19  
**作者**: pzzzzzzy  
**许可**: MIT License