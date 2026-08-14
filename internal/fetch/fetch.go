// Package fetch downloads the FSKriti schedule page, its images, and the
// municipality reference pages over HTTP.
package fetch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client is a small HTTP client with retries.
type Client struct {
	http    *http.Client
	ua      string
	timeout time.Duration
	retries int
}

// New creates a Client.
func New(timeout time.Duration) *Client {
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	return &Client{
		http:    &http.Client{Timeout: timeout},
		ua:      "rethymno-emergency-pharmacy/1.0 (local OCR pipeline)",
		retries: 2,
	}
}

// Get fetches a URL and returns the body bytes.
func (c *Client) Get(ctx context.Context, url string) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt <= c.retries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt) * 500 * time.Millisecond):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", c.ua)
		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("GET %s: status %d", url, resp.StatusCode)
			continue
		}
		return body, nil
	}
	return nil, errors.New("fetch: " + lastErr.Error())
}
