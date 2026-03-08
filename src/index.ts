import 'dotenv/config'
import { ChatOpenAICompletions } from '@langchain/openai'
import { createAgent } from 'langchain'
import { OpenAIFullPromptModel } from './models/prompt-tools-chat-model.ts'
import { getWeather } from './tools/get-weather.ts'
import { math } from './tools/math.ts'

const fields = {
  model: 'gpt-5',
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL },
}

const llm = new ChatOpenAICompletions(fields)

const llmToolCall = new ChatOpenAICompletions({
  ...fields,
  model: 'gpt-5.3-codex',
})

const agent = createAgent({
  name: 'test-agent',
  model: new OpenAIFullPromptModel({ 
    llm,
    llmToolCall
  }),
  tools: [getWeather, math],
})

console.log('Running agent...');

const result = await agent.invoke({
  messages: [
    { role: 'user', content: "what's the weather in San Francisco?, and please calculate 3 + x, x come from weather in San Francisco, if sunny x means 3 unless x will be 1" }],
})

console.log('detail conversation', result.messages)

console.log('\nFinal answer:', result.messages.at(-1)?.content)