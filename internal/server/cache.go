// Package server implements the optional HTTP API for the pipeline: a
// small internal JSON endpoint with an in-memory TTL cache and a daily
// midnight refresh of the current schedule week.
package server

import (
	"sync"
	"time"

	rethymnoemergency "github.com/mheers/rethymno-emergency-pharmacy"
)

// entry is one cached pipeline result.
type entry struct {
	result    *rethymnoemergency.Result
	json      []byte
	etag      string // image SHA-256, raw (quoted when sent)
	fetchedAt time.Time
}

// cache is a small in-memory TTL cache keyed by ISO week, plus a bounded
// replay index keyed by image SHA-256.
type cache struct {
	now func() time.Time

	mu       sync.Mutex
	weeks    map[string]*entry
	sha      map[string]*entry
	shaOrder []string
	maxSHA   int
}

func newCache(now func() time.Time) *cache {
	return &cache{
		now:    now,
		weeks:  map[string]*entry{},
		sha:    map[string]*entry{},
		maxSHA: 20,
	}
}

func (c *cache) get(key string) *entry {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.weeks[key]
}

// put stores the entry under its week key and registers it in the SHA-256
// replay index, evicting the oldest index entries beyond maxSHA.
func (c *cache) put(key string, e *entry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.weeks[key] = e
	if e.etag == "" {
		return
	}
	if _, ok := c.sha[e.etag]; !ok {
		c.shaOrder = append(c.shaOrder, e.etag)
	}
	c.sha[e.etag] = e
	for len(c.shaOrder) > c.maxSHA {
		old := c.shaOrder[0]
		c.shaOrder = c.shaOrder[1:]
		delete(c.sha, old)
	}
}

func (c *cache) getSHA(etag string) *entry {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sha[etag]
}

func (c *cache) fresh(e *entry, ttl time.Duration) bool {
	return c.now().Sub(e.fetchedAt) < ttl
}
