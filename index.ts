import 'dotenv/config'
import { ChatOpenAICompletions } from '@langchain/openai'
import { createAgent, tool } from 'langchain'
import { z } from 'zod'
import { PromptToolsChatModel } from './PromptToolsChatModel.ts'

const llm = new ChatOpenAICompletions({
  model: 'gpt-5',
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL },
})

const llmToolCall = new ChatOpenAICompletions({
  model: 'gpt-5.3-codex',
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL },
})

const model = new PromptToolsChatModel({ 
    llm,
    llmToolCall
});

const getWeather = tool((input) => `It's sunny in ${input.city}`, {
  name: 'get_weather',
  description: 'Get the weather for a given city',
  schema: z.object({
    city: z.string().describe('The city to get the weather for'),
  }),
})

const math = tool((input) => {
  const { a, b, operation } = input
  switch (operation) {
    case 'add': return String(a + b)
    case 'subtract': return String(a - b)
    case 'multiply': return String(a * b)
    case 'divide': return b === 0 ? 'Error: division by zero' : String(a / b)
  }
}, {
  name: 'math',
  description: 'Perform a math operation on two numbers',
  schema: z.object({
    a: z.number().describe('The first number'),
    b: z.number().describe('The second number'),
    operation: z.enum(['add', 'subtract', 'multiply', 'divide']).describe('The operation to perform'),
  }),
})

const agent = createAgent({
  name: 'test-agent',
  model,
  tools: [getWeather, math],
})

console.log('Running agent...')
const result = await agent.invoke({
  messages: [{ role: 'user', content: "What is 10 + 10? and what's the weather in San Francisco?" }],
})

console.log('detail conversation', result.messages)

console.log('\nFinal answer:', result.messages.at(-1)?.content)