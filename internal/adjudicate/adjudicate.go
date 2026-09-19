// Package adjudicate wraps TypeSafe System One judgments for this project.
//
// The System One client itself is the community jev-go SDK
// (github.com/mheers/typesafeai-systemone-jev-go): it builds the typed
// questions (Choice, Noul, Score), posts them to the TypeSafe HTTP API
// together with a state, and returns the typed answers, with retries and
// typed errors. This package adds the judgments the pipeline runs on top:
// catalog-identity adjudication, golden-catalog merge routing and
// warnings-only plausibility verification. A judgment never writes into the
// schedule or the catalog — callers decide what an answer means, and every
// caller keeps a deterministic fallback for when the service is unavailable.
//
// The API key is read from TYPESAFE_API_KEY and is never persisted.
//
// See TYPESAFE_EVALUATION.md for why these judgments exist and where they are
// allowed to act.
package adjudicate

import (
	"context"
	"fmt"
	"os"
	"time"

	jev "github.com/mheers/typesafeai-systemone-jev-go"
)

const (
	// DefaultBaseURL is the TypeSafe API root.
	DefaultBaseURL = jev.DefaultBaseURL
	// DefaultModel is the model alias used when none is configured. Pin a
	// versioned ID instead when thresholds have been tuned against it.
	DefaultModel = jev.ModelJevLatest
	// APIKeyEnv is the environment variable carrying the API key.
	APIKeyEnv = jev.EnvAPIKey
)

// requestTimeout bounds one request attempt. It is generous because the
// batching experiments send requests with hundreds of questions.
const requestTimeout = 120 * time.Second

// Type aliases re-export the SDK's question, answer and response types, so
// the judgments in this package read in their own vocabulary while the wire
// format, decoding and validation live in one place.
type (
	// Question is one typed System One question: Choice, Noul or Score.
	Question = jev.Question
	// Noul answers a yes/no question with the probability that it is true.
	Noul = jev.Noul
	// NoulCriteria clarifies what yes and no mean.
	NoulCriteria = jev.NoulCriteria
	// Choice picks one option from a fixed set.
	Choice = jev.Choice
	// Score rates the state along ordered levels.
	Score = jev.Score
	// Response is a System One evaluation result.
	Response = jev.SystemOneResponse
	// Usage reports the tokens a request consumed.
	Usage = jev.Usage
)

// Client is the project's System One client: a jev-go SDK client configured
// with the project's timeout and retry policy. It is safe for concurrent use
// by multiple goroutines; create one and reuse it.
type Client struct {
	sdk *jev.Client
}

// NewClient returns a client for apiKey with the project's transport
// settings. Options configure the underlying SDK client (for example
// jev.WithModel or jev.WithBaseURL) and are applied last, so they win over
// the project defaults.
func NewClient(apiKey string, opts ...jev.Option) (*Client, error) {
	base := []jev.Option{
		jev.WithAPIKey(apiKey),
		jev.WithTimeout(requestTimeout),
		jev.WithRetryPolicy(retryPolicy()),
	}
	sdk, err := jev.NewClient(append(base, opts...)...)
	if err != nil {
		return nil, fmt.Errorf("adjudicate: %w", err)
	}
	return &Client{sdk: sdk}, nil
}

// NewClientFromEnv builds a client from TYPESAFE_API_KEY.
func NewClientFromEnv(opts ...jev.Option) (*Client, error) {
	key := os.Getenv(APIKeyEnv)
	if key == "" {
		return nil, fmt.Errorf("adjudicate: %s is not set", APIKeyEnv)
	}
	return NewClient(key, opts...)
}

// Model reports the model the client calls unless a request names another.
func (c *Client) Model() string { return c.sdk.Model() }

// retryPolicy preserves the retry behavior of the client this package used
// before adopting the SDK: three retries after the first attempt, one to
// eight seconds of backoff, honoring Retry-After, with no overall budget (the
// caller's context bounds a call).
func retryPolicy() jev.RetryPolicy {
	p := jev.DefaultRetryPolicy()
	p.MaxRetries = 3
	p.BackoffInitial = time.Second
	p.BackoffMax = 8 * time.Second
	p.TotalTimeout = 0
	return p
}

// SystemOne evaluates the questions against the state. All questions are
// answered in one request; independent questions of every type can be mixed.
func (c *Client) SystemOne(ctx context.Context, state any, questions map[string]Question) (*Response, error) {
	return c.sdk.SystemOne(ctx, jev.SystemOneRequest{
		State:     state,
		Questions: jev.Questions(questions),
	})
}
