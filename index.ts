import 'dotenv/config'
import { ChatOpenAICompletions } from '@langchain/openai'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { traceable } from 'langsmith/traceable'
import { z } from 'zod'

const llm = new ChatOpenAICompletions({
  model: 'gpt-5.3-codex',
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL, apiKey: process.env.OPENAI_API_KEY },
})

// --- Tool definitions ---

interface Tool {
  name: string
  description: string
  schema: z.ZodObject<any>
  fn: (input: any) => string | Promise<string>
}

const tools: Tool[] = [
  {
    name: 'get_weather',
    description: 'Get the weather for a given city',
    schema: z.object({
      city: z.string().describe('The city to get the weather for'),
    }),
    fn: (input: { city: string }) => `It's always sunny in ${input.city}!`,
  },
]

function buildRouterPrompt(tools: Tool[]): string {
  const toolList = tools.map(t => {
    const params = Object.entries(t.schema.shape)
      .map(([k, v]: [string, any]) => `${k}: string`)
      .join(', ')
    return `- ${t.name}(${params}): ${t.description}`
  }).join('\n')

  return (
    'You are a tool router. Given a user request and available tools, decide which tool to call.\n' +
    'Respond with ONLY a JSON object. No explanation, no markdown, ONLY valid JSON.\n\n' +
    `Available tools:\n${toolList}\n\n` +
    'If a tool matches, respond: {"name": "<tool_name>", "arguments": {<args>}}\n' +
    'If NO tool matches, respond: {"name": "none"}\n' +
    'Output ONLY JSON.'
  )
}

// --- Agent loop (traced for LangSmith) ---

const selectTool = traceable(async (userMessage: string) => {
  const response = await llm.invoke([
    new SystemMessage(buildRouterPrompt(tools)),
    new HumanMessage(userMessage),
  ])
  const content = (typeof response.content === 'string' ? response.content : '').trim()
  return JSON.parse(content)
}, { name: 'select-tool' })

const executeTool = traceable(async (toolCall: { name: string; arguments: Record<string, any> }) => {
  const tool = tools.find(t => t.name === toolCall.name)
  if (!tool) throw new Error(`Tool "${toolCall.name}" not found`)

  const parsed = tool.schema.parse(toolCall.arguments)
  return await tool.fn(parsed)
}, { name: 'execute-tool' })

const synthesizeAnswer = traceable(async (userMessage: string, toolName: string, toolResult: string) => {
  const response = await llm.invoke([
    new SystemMessage('Answer the user based on the tool result. Be concise and helpful.'),
    new HumanMessage(
      `User asked: ${userMessage}\nTool "${toolName}" returned: ${toolResult}\n\nGive a helpful answer.`
    ),
  ])
  return typeof response.content === 'string' ? response.content : ''
}, { name: 'synthesize-answer' })

const runAgent = traceable(async (userMessage: string) => {
  // Step 1: Route to tool
  const toolCall = await selectTool(userMessage)
  console.log(`  [router] selected: ${toolCall.name}(${JSON.stringify(toolCall.arguments ?? {})})`)

  if (toolCall.name === 'none') {
    const response = await llm.invoke([new HumanMessage(userMessage)])
    return typeof response.content === 'string' ? response.content : ''
  }

  // Step 2: Execute tool
  const toolResult = await executeTool(toolCall)
  console.log(`  [tool]   ${toolCall.name} => ${toolResult}`)

  // Step 3: Synthesize final answer
  return await synthesizeAnswer(userMessage, toolCall.name, toolResult)
}, { name: 'weather-agent' })

// --- Run ---

console.log('Running agent...')
const result = await runAgent("What's the weather in Tokyo?")
console.log('\nFinal answer:', result)
