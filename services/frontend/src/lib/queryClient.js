import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // This is a personal single-user tool against a local backend -
      // refetching on every window focus is noise, not freshness. The one
      // genuinely live view (Queue) opts into polling explicitly.
      refetchOnWindowFocus: false,
      staleTime: 30_000,
      retry: 1,
    },
  },
})
