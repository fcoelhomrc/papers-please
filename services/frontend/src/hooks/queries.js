/* Every server interaction in one place, as TanStack Query hooks.
 *
 * Replaces the useState/useEffect/.then/.catch/.finally block that each page
 * used to hand-roll: loading and error are derived, results are cached by
 * key (so paging back to a previous offset or re-running an earlier search
 * is instant), in-flight duplicates are deduped, and Queue's polling is a
 * declarative refetchInterval instead of a hand-managed setInterval plus a
 * `cancelled` flag.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import * as api from '../api'

export const keys = {
  search: (mode, q, opts) => ['search', mode, q, opts],
  documents: (params) => ['documents', params],
  status: () => ['status'],
}

export function useSearch({ mode, query, topK, rerank, enabled }) {
  return useQuery({
    queryKey: keys.search(mode, query, { topK, rerank }),
    queryFn: () =>
      mode === 'semantic'
        ? api.search(query, { topK, rerank, rerankTopK: 5 })
        : api.searchKeyword(query, { topK }),
    enabled: Boolean(enabled && query.trim()),
    // A search result is a snapshot of an index that only changes when the
    // pipeline ingests more - not worth auto-invalidating, but cheap to keep.
    staleTime: 5 * 60_000,
  })
}

export function useDocuments(params) {
  return useQuery({
    queryKey: keys.documents(params),
    queryFn: () => api.listDocuments(params),
    // Keeps the previous page's rows on screen while the next page loads,
    // so paging doesn't flash an empty table.
    placeholderData: (prev) => prev,
  })
}

export function useStatus({ refetchInterval } = {}) {
  return useQuery({
    queryKey: keys.status(),
    queryFn: api.getStatus,
    refetchInterval,
    staleTime: 0,
  })
}

export function useFetchPapers() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.fetchPapers,
    onSuccess: () => {
      // New papers change both the document list and the pipeline backlog -
      // the old UI left stale counts on screen until a manual reload.
      qc.invalidateQueries({ queryKey: ['documents'] })
      qc.invalidateQueries({ queryKey: keys.status() })
    },
  })
}

export function useChat() {
  return useMutation({
    mutationFn: ({ message, threadId }) => api.chat(message, threadId),
  })
}
