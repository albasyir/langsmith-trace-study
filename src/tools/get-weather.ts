import { tool } from 'langchain'
import { z } from 'zod'

export const getWeather = tool((input) => `It's sunny in ${input.city}`, {
  name: 'get_weather',
  description: 'Get the weather for a given city',
  schema: z.object({
    city: z.string().describe('The city to get the weather for'),
  }),
})
