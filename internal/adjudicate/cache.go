package adjudicate

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	jev "github.com/mheers/typesafeai-systemone-jev-go"
)

// DefaultJudgeModel is the pinned System One model the runtime identity
// adjudicator was measured with (TYPESAFE_EVALUATION.md §4.1). Re-measure the
// acceptance gate before changing it.
const DefaultJudgeModel = jev.ModelJev1130

const cacheVersion = 1

// DecisionCache is the on-disk record of identity decisions, keyed by a hash
// of the model, the OCR reading and the candidate list. Either every decision
// is recorded and reused, or a group is re-evaluated — an uncached live answer
// never enters the output silently (TYPESAFE_EVALUATION.md §5).
//
// The file is a JSON document with a version, a model frame and the entries;
// a missing file starts empty. Writes are atomic (temporary file + rename),
// so a crash cannot leave a half-written cache.
type DecisionCache struct {
	path    string
	mu      sync.Mutex
	entries map[string]cacheEntry
}

// cacheEntry is one recorded decision. ChoiceIndex is the candidate's position
// in the request (-1 for IdentityNone), so the record does not depend on a
// caller's candidate ID scheme. The plausibility fields are nil for identity
// decisions and vice versa; identity and plausibility decisions share the file
// under namespaced keys.
type cacheEntry struct {
	ChoiceIndex   int                `json:"choice_index"`
	Confidence    float64            `json:"confidence"`
	SamePharmacy  float64            `json:"same_pharmacy"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Model         string             `json:"model,omitempty"`
	DecidedAt     string             `json:"decided_at,omitempty"`

	NamePlausible    *float64 `json:"name_plausible,omitempty"`
	AddressPlausible *float64 `json:"address_plausible,omitempty"`
}

type cacheFile struct {
	Version int                   `json:"version"`
	Entries map[string]cacheEntry `json:"entries"`
}

// OpenDecisionCache loads the cache at path, creating an empty one when the
// file does not exist. An unreadable or incompatible file is an error: a
// corrupt cache must stop the run, not silently reset it.
func OpenDecisionCache(path string) (*DecisionCache, error) {
	if strings.TrimSpace(path) == "" {
		return nil, errors.New("adjudicate: decision cache path is empty")
	}
	c := &DecisionCache{path: path, entries: map[string]cacheEntry{}}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return nil, fmt.Errorf("adjudicate: read decision cache: %w", err)
	}
	if len(data) == 0 {
		return c, nil
	}
	var f cacheFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("adjudicate: parse decision cache %s: %w", path, err)
	}
	if f.Version != cacheVersion {
		return nil, fmt.Errorf("adjudicate: decision cache %s has version %d, want %d", path, f.Version, cacheVersion)
	}
	if f.Entries != nil {
		c.entries = f.Entries
	}
	return c, nil
}

// Len reports how many decisions the cache holds.
func (c *DecisionCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries)
}

// Get returns the recorded decision for a key.
func (c *DecisionCache) Get(key string) (cacheEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	return e, ok
}

// put records a decision and persists the cache.
func (c *DecisionCache) put(key string, e cacheEntry) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[key] = e
	return c.saveLocked()
}

func (c *DecisionCache) saveLocked() error {
	data, err := json.MarshalIndent(cacheFile{Version: cacheVersion, Entries: c.entries}, "", "  ")
	if err != nil {
		return fmt.Errorf("adjudicate: encode decision cache: %w", err)
	}
	dir := filepath.Dir(c.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("adjudicate: decision cache directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".identity-cache-*")
	if err != nil {
		return fmt.Errorf("adjudicate: decision cache temp file: %w", err)
	}
	name := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(name)
		return fmt.Errorf("adjudicate: write decision cache: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(name)
		return fmt.Errorf("adjudicate: close decision cache: %w", err)
	}
	if err := os.Rename(name, c.path); err != nil {
		os.Remove(name)
		return fmt.Errorf("adjudicate: replace decision cache: %w", err)
	}
	return nil
}

// CachedJudge wraps a Client with a DecisionCache. Cached groups are served
// from the cache; misses go to the model in one request and are persisted.
// It is the runtime-facing judge: the pipeline accepts this type instead of a
// bare Client, so a decision can never reach the output without being
// recorded.
type CachedJudge struct {
	client *Client
	cache  *DecisionCache
}

// NewCachedJudge returns a judge backed by the cache.
func NewCachedJudge(client *Client, cache *DecisionCache) *CachedJudge {
	return &CachedJudge{client: client, cache: cache}
}

// Model reports the pinned model the judge calls.
func (j *CachedJudge) Model() string {
	return j.client.Model()
}

// AdjudicateIdentity serves cached verdicts and evaluates misses, in one
// request. Accepted is recomputed from cfg for cached and fresh verdicts
// alike, so both paths gate identically.
func (j *CachedJudge) AdjudicateIdentity(ctx context.Context, groups []IdentityGroup, cfg IdentityConfig) ([]IdentityVerdict, error) {
	if len(groups) == 0 {
		return nil, errors.New("adjudicate: no identity groups")
	}
	model := j.client.Model()
	verdicts := make([]IdentityVerdict, len(groups))
	var missIdx []int
	var missGroups []IdentityGroup
	for i, g := range groups {
		key, err := cacheKey(model, g)
		if err == nil {
			if e, ok := j.cache.Get(key); ok {
				v, err := verdictFromEntry(i, g, e)
				if err == nil {
					verdicts[i] = v
					continue
				}
			}
		}
		missIdx = append(missIdx, i)
		missGroups = append(missGroups, g)
	}
	if len(missGroups) > 0 {
		fresh, _, err := j.client.AdjudicateIdentity(ctx, missGroups, IdentityConfig{})
		if err != nil {
			return nil, err
		}
		if len(fresh) != len(missGroups) {
			return nil, fmt.Errorf("adjudicate: %d verdicts for %d groups", len(fresh), len(missGroups))
		}
		for k, idx := range missIdx {
			verdicts[idx] = fresh[k]
			key, err := cacheKey(model, missGroups[k])
			if err != nil {
				continue
			}
			if err := j.cache.put(key, entryFromVerdict(missGroups[k], fresh[k], model)); err != nil {
				return nil, err
			}
		}
	}
	for i := range verdicts {
		verdicts[i].Accepted = AcceptIdentity(verdicts[i].Choice, verdicts[i].Confidence, verdicts[i].SamePharmacy, cfg)
	}
	return verdicts, nil
}

// cacheKey hashes the decision's inputs: the model, the reading and the
// candidate names and addresses in order. Candidate IDs are excluded so that
// the key survives a caller changing its ID scheme.
func cacheKey(model string, g IdentityGroup) (string, error) {
	payload, err := json.Marshal(struct {
		Model string        `json:"model"`
		Group IdentityGroup `json:"group"`
	}{Model: model, Group: groupWithoutIDs(g)})
	if err != nil {
		return "", fmt.Errorf("adjudicate: encode cache key: %w", err)
	}
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:]), nil
}

func groupWithoutIDs(g IdentityGroup) IdentityGroup {
	out := IdentityGroup{Reading: g.Reading, Candidates: make([]IdentityCandidate, len(g.Candidates))}
	for i, c := range g.Candidates {
		out.Candidates[i] = IdentityCandidate{Name: c.Name, Address: c.Address}
	}
	return out
}

func entryFromVerdict(g IdentityGroup, v IdentityVerdict, model string) cacheEntry {
	index := -1
	if v.Choice != IdentityNone {
		for i, c := range g.Candidates {
			if c.ID == v.Choice {
				index = i
				break
			}
		}
	}
	return cacheEntry{
		ChoiceIndex:   index,
		Confidence:    v.Confidence,
		SamePharmacy:  v.SamePharmacy,
		Probabilities: v.Probabilities,
		Model:         model,
		DecidedAt:     time.Now().UTC().Format(time.RFC3339),
	}
}

func verdictFromEntry(group int, g IdentityGroup, e cacheEntry) (IdentityVerdict, error) {
	v := IdentityVerdict{
		Group:         group,
		Choice:        IdentityNone,
		Confidence:    e.Confidence,
		Probabilities: e.Probabilities,
		SamePharmacy:  e.SamePharmacy,
	}
	switch {
	case e.ChoiceIndex < 0:
		// IdentityNone.
	case e.ChoiceIndex >= len(g.Candidates):
		return IdentityVerdict{}, fmt.Errorf("adjudicate: cached choice index %d out of range for %d candidates", e.ChoiceIndex, len(g.Candidates))
	default:
		v.Choice = g.Candidates[e.ChoiceIndex].ID
	}
	return v, nil
}
