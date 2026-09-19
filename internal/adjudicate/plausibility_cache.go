package adjudicate

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// CachedPlausibilityJudge wraps a Client with a DecisionCache. It is the
// runtime-facing judge for the warnings-only plausibility verification
// (TYPESAFE_EVALUATION.md §3D): every decision is persisted before it can
// influence a warning, and cached decisions are reused, so repeated parses
// stay deterministic. It shares the cache file with the identity judge (same
// record shape, namespaced keys), so callers configure one cache path for
// both.
//
// Cache misses are evaluated one request per entry — the measured granularity
// (§4.2): merging entries into one request changes answers beyond run-to-run
// noise. Gates are applied by the caller, so a gate change does not
// invalidate recorded probabilities.
type CachedPlausibilityJudge struct {
	client *Client
	cache  *DecisionCache
}

// NewCachedPlausibilityJudge returns a judge backed by the cache.
func NewCachedPlausibilityJudge(client *Client, cache *DecisionCache) *CachedPlausibilityJudge {
	return &CachedPlausibilityJudge{client: client, cache: cache}
}

// Model reports the pinned model the judge calls.
func (j *CachedPlausibilityJudge) Model() string {
	return j.client.Model()
}

// AdjudicatePlausibility serves cached verdicts and evaluates misses, one
// request each. Verdicts are in entry order; each probability is -1 when the
// field was empty and no question was asked.
func (j *CachedPlausibilityJudge) AdjudicatePlausibility(ctx context.Context, entries []PlausibilityEntry) ([]PlausibilityVerdict, error) {
	if len(entries) == 0 {
		return nil, errors.New("adjudicate: no plausibility entries")
	}
	model := j.client.Model()
	verdicts := make([]PlausibilityVerdict, len(entries))
	for i, e := range entries {
		unasked := PlausibilityVerdict{Entry: i, NamePlausible: -1, AddressPlausible: -1}
		if e.Name == "" && e.Address == "" {
			verdicts[i] = unasked
			continue
		}
		key, err := plausibilityCacheKey(model, e)
		if err != nil {
			return nil, err
		}
		if ce, ok := j.cache.Get(key); ok {
			v := unasked
			if e.Name != "" && ce.NamePlausible != nil {
				v.NamePlausible = *ce.NamePlausible
			}
			if e.Address != "" && ce.AddressPlausible != nil {
				v.AddressPlausible = *ce.AddressPlausible
			}
			if v.NamePlausible >= 0 || v.AddressPlausible >= 0 {
				verdicts[i] = v
				continue
			}
		}
		got, _, err := j.client.AdjudicatePlausibility(ctx, []PlausibilityEntry{e})
		if err != nil {
			return nil, err
		}
		if len(got) != 1 {
			return nil, fmt.Errorf("adjudicate: %d verdicts for one plausibility entry", len(got))
		}
		v := got[0]
		v.Entry = i
		verdicts[i] = v
		entry, err := entryFromPlausibility(v, model)
		if err != nil {
			return nil, err
		}
		if err := j.cache.put(key, entry); err != nil {
			return nil, err
		}
	}
	return verdicts, nil
}

// plausibilityCacheKey hashes the decision's inputs: the model and the entry.
// The key is namespaced so it cannot collide with an identity decision in the
// shared cache file.
func plausibilityCacheKey(model string, e PlausibilityEntry) (string, error) {
	payload, err := json.Marshal(struct {
		Model string            `json:"model"`
		Entry PlausibilityEntry `json:"entry"`
	}{Model: model, Entry: e})
	if err != nil {
		return "", fmt.Errorf("adjudicate: encode plausibility cache key: %w", err)
	}
	sum := sha256.Sum256(payload)
	return "plaus:" + hex.EncodeToString(sum[:]), nil
}

// entryFromPlausibility records the answered fields of a verdict. An entry
// with no answered field is a programming error: it would make the cache
// replay indistinguishable from a miss.
func entryFromPlausibility(v PlausibilityVerdict, model string) (cacheEntry, error) {
	e := cacheEntry{Model: model, DecidedAt: time.Now().UTC().Format(time.RFC3339)}
	if v.NamePlausible >= 0 {
		name := v.NamePlausible
		e.NamePlausible = &name
	}
	if v.AddressPlausible >= 0 {
		address := v.AddressPlausible
		e.AddressPlausible = &address
	}
	if e.NamePlausible == nil && e.AddressPlausible == nil {
		return cacheEntry{}, errors.New("adjudicate: plausibility verdict has no answered field")
	}
	return e, nil
}
