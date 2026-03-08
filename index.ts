import 'dotenv/config'
import { ChatOpenAICompletions } from '@langchain/openai'
import { createAgent, tool } from 'langchain'
import { z } from 'zod'
import { PromptToolsChatModel } from './PromptToolsChatModel.ts'

const innerLlm = new ChatOpenAICompletions({
  model: 'gpt-5.3-codex',
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL, apiKey: process.env.OPENAI_API_KEY },
})

const model = new PromptToolsChatModel({ llm: innerLlm })

const getWeather = tool((input) => `It's always sunny in ${input.city}!`, {
  name: 'get_weather',
  description: 'Get the weather for a given city',
  schema: z.object({
    city: z.string().describe('The city to get the weather for'),
  }),
})

const agent = createAgent({
  model,
  tools: [getWeather],
})

console.log('Running agent...')
const result = await agent.invoke({
  messages: [{ role: 'user', content: "What's the weather in San Francisco?" }],
})

console.log('\nFinal answer:', result.messages.at(-1)?.content)
