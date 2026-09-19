// Package adjudicate wraps TypeSafe System One judgments for this project.
//
// The package is deliberately small: it builds typed questions (Choice, Noul,
// Score), posts them to the TypeSafe HTTP API together with a state, and
// returns the answers for ordinary Go code to consume. A judgment never writes
// into the schedule or the catalog — callers decide what an answer means, and
// every caller keeps a deterministic fallback for when the service is
// unavailable.
//
// The API key is read from TYPESAFE_API_KEY and is never persisted.
//
// See TYPESAFE_EVALUATION.md for why these judgments exist and where they are
// allowed to act.
package adjudicate

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"
)

const (
	// DefaultBaseURL is the TypeSafe API root.
	DefaultBaseURL = "https://api.typesafe.ai"
	// DefaultModel is the model alias used when none is configured. Pin a
	// versioned ID instead when thresholds have been tuned against it.
	DefaultModel = "jev-latest"
	// APIKeyEnv is the environment variable carrying the API key.
	APIKeyEnv = "TYPESAFE_API_KEY"
)

// Question is one typed System One question. The concrete types are Choice,
// Noul and Score.
type Question interface {
	question()
}

// Choice picks one option from a fixed set. The response carries the chosen
// option, the full probability distribution and a confidence.
type Choice struct {
	Instructions any
	Criteria     map[string]any // option -> description (string, object or nil)
}

func (Choice) question() {}

// Noul answers a yes/no question with the probability that the answer is yes.
type Noul struct {
	Instructions any
	Criteria     *NoulCriteria // optional true/false descriptions
}

// NoulCriteria clarifies what yes and no mean.
type NoulCriteria struct {
	True  string
	False string
}

func (Noul) question() {}

// Score rates the state along ordered levels. The response carries a
// probability-weighted score, the distribution and a confidence.
type Score struct {
	Instructions any
	Criteria     []string // ordered level descriptions, at least two
}

func (Score) question() {}

func marshalQuestion(q Question) (map[string]any, error) {
	switch v := q.(type) {
	case Choice:
		return map[string]any{
			"type":         "choice",
			"instructions": v.Instructions,
			"criteria":     v.Criteria,
		}, nil
	case Noul:
		m := map[string]any{
			"type":         "noul",
			"instructions": v.Instructions,
		}
		if v.Criteria != nil {
			m["criteria"] = map[string]string{"true": v.Criteria.True, "false": v.Criteria.False}
		}
		return m, nil
	case Score:
		return map[string]any{
			"type":         "score",
			"instructions": v.Instructions,
			"criteria":     v.Criteria,
		}, nil
	default:
		return nil, fmt.Errorf("adjudicate: unknown question type %T", q)
	}
}

// Usage reports the tokens a request consumed.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Answer is one typed answer. Fields not belonging to the answer's Type are
// zero.
type Answer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence,omitempty"`
	Noul          float64            `json:"noul,omitempty"`
	Score         float64            `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
}

// Response is a System One evaluation result.
type Response struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
}

// Client calls the TypeSafe System One endpoint.
type Client struct {
	APIKey  string
	BaseURL string
	Model   string
	HTTP    *http.Client
}

// NewClient returns a client with defaults filled in.
func NewClient(apiKey string) *Client {
	return &Client{
		APIKey:  apiKey,
		BaseURL: DefaultBaseURL,
		Model:   DefaultModel,
		HTTP:    &http.Client{Timeout: 120 * time.Second},
	}
}

// NewClientFromEnv builds a client from TYPESAFE_API_KEY.
func NewClientFromEnv() (*Client, error) {
	key := os.Getenv(APIKeyEnv)
	if key == "" {
		return nil, fmt.Errorf("adjudicate: %s is not set", APIKeyEnv)
	}
	return NewClient(key), nil
}

// SystemOne evaluates the questions against the state. All questions are
// answered in one request; independent questions of every type can be mixed.
func (c *Client) SystemOne(ctx context.Context, state any, questions map[string]Question) (*Response, error) {
	if len(questions) == 0 {
		return nil, errors.New("adjudicate: no questions")
	}
	marshaled := make(map[string]any, len(questions))
	for id, q := range questions {
		m, err := marshalQuestion(q)
		if err != nil {
			return nil, err
		}
		marshaled[id] = m
	}
	body, err := json.Marshal(map[string]any{
		"state":     state,
		"model":     c.model(),
		"questions": marshaled,
	})
	if err != nil {
		return nil, fmt.Errorf("adjudicate: encode request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		if attempt > 0 {
			time.Sleep(backoff(attempt))
		}
		resp, retryAfter, err := c.post(ctx, body)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		var httpErr *HTTPError
		if !errors.As(err, &httpErr) || !httpErr.Retryable() {
			return nil, err
		}
		if retryAfter > 0 && retryAfter < 15*time.Second {
			time.Sleep(retryAfter)
		}
	}
	return nil, fmt.Errorf("adjudicate: giving up after retries: %w", lastErr)
}

func (c *Client) model() string {
	if c.Model != "" {
		return c.Model
	}
	return DefaultModel
}

func (c *Client) post(ctx context.Context, body []byte) (*Response, time.Duration, error) {
	base := c.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/systemone", bytes.NewReader(body))
	if err != nil {
		return nil, 0, fmt.Errorf("adjudicate: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	httpClient := c.HTTP
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * time.Second}
	}
	res, err := httpClient.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("adjudicate: call: %w", err)
	}
	defer res.Body.Close()
	data, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return nil, 0, fmt.Errorf("adjudicate: read response: %w", err)
	}
	if res.StatusCode != http.StatusOK {
		return nil, parseRetryAfter(res.Header.Get("Retry-After")), &HTTPError{
			StatusCode: res.StatusCode,
			Body:       string(data),
		}
	}
	var out Response
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, 0, fmt.Errorf("adjudicate: decode response: %w", err)
	}
	return &out, 0, nil
}

// HTTPError is a non-200 API response.
type HTTPError struct {
	StatusCode int
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("adjudicate: API status %d: %s", e.StatusCode, truncate(e.Body, 300))
}

// Retryable reports whether the request may succeed later.
func (e *HTTPError) Retryable() bool {
	switch e.StatusCode {
	case http.StatusTooManyRequests, 529, http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	}
	return false
}

func parseRetryAfter(v string) time.Duration {
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

func backoff(attempt int) time.Duration {
	d := time.Duration(1<<uint(attempt-1)) * time.Second
	if d > 8*time.Second {
		d = 8 * time.Second
	}
	return d
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
