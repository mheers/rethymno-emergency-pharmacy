package adjudicate

import (
	"testing"

	jev "github.com/mheers/typesafeai-systemone-jev-go"
)

// newTestClient returns a client that talks to baseURL (usually an
// httptest.Server) and names model, so the request bodies are easy to
// assert on.
func newTestClient(t *testing.T, baseURL, model string) *Client {
	t.Helper()
	client, err := NewClient("test-key", jev.WithBaseURL(baseURL), jev.WithModel(model))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return client
}

// truncate shortens s to at most n bytes, for the fixed-width experiment
// tables.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
