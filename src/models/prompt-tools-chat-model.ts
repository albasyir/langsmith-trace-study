import { BaseChatModel } from '@langchain/core/language_models/chat_models'
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import type { ChatOpenAICompletions } from '@langchain/openai'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'

interface ToolMeta {
  name: string
  description: string
  parameters: Record<string, any>
}

/**
 * A ChatModel wrapper that implements tool calling via prompt engineering.
 * Compatible with LangChain's `createAgent` / ReAct agent loop.
 *
 * Use this when the underlying LLM server doesn't support native
 * OpenAI tool_calls in the response.
 */
export class OpenAIFullPromptModel extends BaseChatModel {
  private llm: ChatOpenAICompletions
  private boundTools: ToolMeta[] = []
  private llmToolCall: ChatOpenAICompletions;

  constructor(fields: { llm: ChatOpenAICompletions, llmToolCall?: ChatOpenAICompletions }) {
    super({})
    this.llm = fields.llm
    this.llmToolCall = fields.llmToolCall || this.llm;
  }

  _llmType() {
    return 'prompt-tools-chat-model'
  }

  bindTools(tools: any[], kwargs?: any): this {
    const bound = new OpenAIFullPromptModel({ llm: this.llm, llmToolCall: this.llmToolCall })
    bound.boundTools = tools.map((t) => {
      // OpenAI format: {type: "function", function: {name, description, parameters}}
      if (t.function) {
        return {
          name: t.function.name,
          description: t.function.description ?? '',
          parameters: t.function.parameters ?? {},
        }
      }
      // LangChain tool (DynamicStructuredTool) with Zod schema
      let parameters: Record<string, any> = {}
      if (t.schema) {
        if (typeof t.schema.toJSONSchema === 'function') {
          parameters = t.schema.toJSONSchema()
        } else if (t.schema.shape) {
          // Fallback: manually extract from Zod shape
          const props: Record<string, any> = {}
          const required: string[] = []
          for (const [key, val] of Object.entries(t.schema.shape)) {
            const v = val as any
            props[key] = {
              type: v.type ?? 'string',
              ...(v.description ? { description: v.description } : {}),
            }
            if (!v.isOptional?.()) required.push(key)
          }
          parameters = { type: 'object', properties: props, required }
        }
      }
      return {
        name: t.name,
        description: t.description ?? '',
        parameters,
      }
    })
    return bound as this
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ) {
    if (this.boundTools.length === 0) {
      const result = await this.llm.invoke(messages)
      return {
        generations: [{ message: result, text: typeof result.content === 'string' ? result.content : '' }],
      }
    }

    const flat = this.flattenMessages(messages)

    const routerResult = await this.callRouter(flat)
    const calls = Array.isArray(routerResult) ? routerResult : (routerResult ? [routerResult] : [])
    const validCalls = calls.filter((c) => c.name && c.name !== 'none')

    if (validCalls.length > 0) {
      const msg = new AIMessage({
        content: '',
        tool_calls: validCalls.map((c) => ({
          id: `call_${crypto.randomUUID()}`,
          name: c.name,
          args: c.arguments ?? {},
        })),
      })
      return { generations: [{ message: msg, text: '' }] }
    }

    const answer = await this.llm.invoke(flat)
    return {
      generations: [{
        message: answer,
        text: typeof answer.content === 'string' ? answer.content : '',
      }],
    }
  }

  /**
   * Convert message history into a format the server understands.
   * Replaces AIMessage(tool_calls) + ToolMessage pairs with plain messages.
   */
  private flattenMessages(messages: BaseMessage[]): BaseMessage[] {
    const result: BaseMessage[] = []

    for (const msg of messages) {
      if (AIMessage.isInstance(msg) && msg.tool_calls && msg.tool_calls.length > 0) {
        const callSummary = msg.tool_calls
          .map((tc: any) => `Called tool "${tc.name}" with ${JSON.stringify(tc.args)}`)
          .join('; ')
        result.push(new AIMessage(callSummary))
        continue
      }

      if (ToolMessage.isInstance(msg)) {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
        result.push(new HumanMessage(`[Tool result]: ${content}`))
        continue
      }

      result.push(msg)
    }

    return result
  }

  /**
   * Separate LLM call framed as a "tool router" that MUST output JSON.
   * This bypasses the model's refusal to use tools in normal conversation.
   */
  private async callRouter(
    conversationMessages: BaseMessage[],
  ): Promise<{ name: string; arguments?: Record<string, any> }[]> {
    const toolDefs = this.boundTools.map((t) => {
      const props = t.parameters?.properties ?? {}
      const required = t.parameters?.required ?? []
      const paramLines = Object.entries(props)
        .map(([k, v]: [string, any]) => {
          const req = required.includes(k) ? ' (required)' : ''
          const desc = v.description ? ` - ${v.description}` : ''
          const enumVals = v.enum ? ` [values: ${v.enum.map((e: any) => `"${e}"`).join(', ')}]` : ''
          return `    "${k}": ${v.type ?? 'string'}${enumVals}${req}${desc}`
        })
        .join('\n')
      const exampleArgs = Object.fromEntries(
        Object.entries(props).map(([k]) => [k, `<${k}>`])
      )
      return (
        `Tool: ${t.name}\n` +
        `Description: ${t.description}\n` +
        `Parameters:\n${paramLines}\n` +
        `Example: {"name": "${t.name}", "arguments": ${JSON.stringify(exampleArgs)}}`
      )
    }).join('\n\n')

    const conversationSummary = conversationMessages
      .filter((m) => !SystemMessage.isInstance(m))
      .map((m) => {
        const role = HumanMessage.isInstance(m) ? 'User' : 'Assistant'
        return `${role}: ${typeof m.content === 'string' ? m.content : JSON.stringify(m.content)}`
      })
      .join('\n')

    const routerSystemPrompt =
      'You are a tool router. Given a conversation and available tools, decide which tools to call.\n' +
      'Respond with ONLY a JSON array. No explanation, no markdown, ONLY valid JSON.\n' +
      'You MUST use the EXACT parameter names shown below.\n\n' +
      `${toolDefs}\n\n` +
      'RULES:\n' +
      '- If ONE tool is needed, respond: [{"name": "<tool>", "arguments": {<params>}}]\n' +
      '- If MULTIPLE tools are needed, return ALL of them: [{"name": "<tool1>", "arguments": {...}}, {"name": "<tool2>", "arguments": {...}}]\n' +
      '- If the conversation is already fully answered or no tool is needed, respond: [{"name": "none"}]\n' +
      '- If some parts are answered (by previous tool results) but other parts still need a tool, return ONLY the unanswered tool calls.\n' +
      'Output ONLY a JSON array.'

    const response = await this.llmToolCall.invoke([
      new SystemMessage(routerSystemPrompt),
      new HumanMessage(conversationSummary || 'No conversation yet'),
    ])

    const content = (typeof response.content === 'string' ? response.content : '').trim()
    try {
      const parsed = JSON.parse(content)
      return Array.isArray(parsed) ? parsed : [parsed]
    } catch {
      return []
    }
  }
}
