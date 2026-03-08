import { tool } from 'langchain'
import { z } from 'zod'

export const math = tool((input) => {
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
