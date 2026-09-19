package adjudicate

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func testIdentityGroup(reading string) IdentityGroup {
	return IdentityGroup{
		Reading: IdentityReading{Name: reading, Address: "Δημητρακάκη 21", Phone: "2831055212"},
		Candidates: []IdentityCandidate{
			{ID: "cand-a", Name: "Βαρούχα - Αναγνωστάκης", Address: "Δημητρακάκη 21"},
			{ID: "cand-b", Name: "Καλαφατάκη Ελένη - Γεωργία", Address: "Δημητρακάκη 21"},
		},
	}
}

func TestDecisionCacheRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "decisions.json")
	cache, err := OpenDecisionCache(path)
	if err != nil {
		t.Fatalf("OpenDecisionCache: %v", err)
	}
	if cache.Len() != 0 {
		t.Fatalf("new cache has %d entries", cache.Len())
	}
	key, err := cacheKey(DefaultJudgeModel, testIdentityGroup("KAAYΦATAKH EΛENH"))
	if err != nil {
		t.Fatalf("cacheKey: %v", err)
	}
	if _, ok := cache.Get(key); ok {
		t.Fatal("empty cache returned an entry")
	}
	if err := cache.put(key, cacheEntry{ChoiceIndex: 1, Confidence: 0.9, SamePharmacy: 0.95, Model: DefaultJudgeModel}); err != nil {
		t.Fatalf("put: %v", err)
	}

	reloaded, err := OpenDecisionCache(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	e, ok := reloaded.Get(key)
	if !ok {
		t.Fatal("reloaded cache lost the entry")
	}
	if e.ChoiceIndex != 1 || e.Confidence != 0.9 || e.SamePharmacy != 0.95 {
		t.Errorf("entry = %+v", e)
	}

	// corrupt file and wrong version must fail loudly, not reset silently
	if err := os.WriteFile(path, []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenDecisionCache(path); err == nil {
		t.Error("corrupt cache did not error")
	}
	if err := os.WriteFile(path, []byte(`{"version":99,"entries":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenDecisionCache(path); err == nil {
		t.Error("wrong cache version did not error")
	}
}

func TestCacheKeyContentSensitivity(t *testing.T) {
	group := testIdentityGroup("A")
	base, err := cacheKey("m1", group)
	if err != nil {
		t.Fatalf("cacheKey: %v", err)
	}
	if k, _ := cacheKey("m2", group); k == base {
		t.Error("model must affect the key")
	}
	renamed := group
	renamed.Candidates = append([]IdentityCandidate(nil), group.Candidates...)
	renamed.Candidates[0].ID = "different-id"
	if k, _ := cacheKey("m1", renamed); k != base {
		t.Error("candidate IDs must not affect the key")
	}
	changed := group
	changed.Candidates = append([]IdentityCandidate(nil), group.Candidates...)
	changed.Candidates[0].Name = "Άλλο Φαρμακείο"
	if k, _ := cacheKey("m1", changed); k == base {
		t.Error("candidate names must affect the key")
	}
	other := group
	other.Reading.Name = "B"
	if k, _ := cacheKey("m1", other); k == base {
		t.Error("the reading must affect the key")
	}
}

func TestVerdictFromEntry(t *testing.T) {
	group := testIdentityGroup("x")
	v, err := verdictFromEntry(0, group, cacheEntry{ChoiceIndex: -1, Confidence: 0.7, SamePharmacy: -1})
	if err != nil {
		t.Fatalf("none entry: %v", err)
	}
	if v.Choice != IdentityNone || v.Confidence != 0.7 {
		t.Errorf("none entry = %+v", v)
	}
	v, err = verdictFromEntry(0, group, cacheEntry{ChoiceIndex: 1, Confidence: 0.8, SamePharmacy: 0.9})
	if err != nil {
		t.Fatalf("candidate entry: %v", err)
	}
	if v.Choice != "cand-b" {
		t.Errorf("choice = %q, want cand-b", v.Choice)
	}
	if _, err := verdictFromEntry(0, group, cacheEntry{ChoiceIndex: 5}); err == nil {
		t.Error("out-of-range index did not error")
	}
}

// TestCachedJudgeLive exercises the runtime judge path against the real API
// (one request) and verifies the decision is persisted and reused. It is
// skipped unless TYPESAFE_EXPERIMENT=1 and TYPESAFE_API_KEY are set.
func TestCachedJudgeLive(t *testing.T) {
	if os.Getenv("TYPESAFE_EXPERIMENT") == "" {
		t.Skip("set TYPESAFE_EXPERIMENT=1 to run the live TypeSafe experiment")
	}
	client, err := NewClientFromEnv()
	if err != nil {
		t.Skipf("live experiment unavailable: %v", err)
	}
	client.Model = DefaultJudgeModel
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	cache, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatalf("cache: %v", err)
	}
	judge := NewCachedJudge(client, cache)
	group := testIdentityGroup("KAAYΦATAKH EΛENH")
	group.Reading.Address = "ΔHMHTPAKAKH 21"
	cfg := IdentityConfig{MinConfidence: 0.8, MinSamePharmacy: 0.8}

	verdicts, err := judge.AdjudicateIdentity(context.Background(), []IdentityGroup{group}, cfg)
	if err != nil {
		t.Fatalf("live call: %v", err)
	}
	if len(verdicts) != 1 {
		t.Fatalf("verdicts = %d", len(verdicts))
	}
	v := verdicts[0]
	if v.Choice != IdentityNone && v.Choice != "cand-a" && v.Choice != "cand-b" {
		t.Fatalf("unexpected choice %q", v.Choice)
	}
	if cache.Len() != 1 {
		t.Fatalf("cache entries = %d, want 1", cache.Len())
	}
	reloaded, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	replay, err := NewCachedJudge(client, reloaded).AdjudicateIdentity(context.Background(), []IdentityGroup{group}, cfg)
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if replay[0].Choice != v.Choice || replay[0].Confidence != v.Confidence {
		t.Errorf("replay = %+v, want the cached %+v", replay[0], v)
	}
	t.Logf("choice=%s confidence=%.2f same_pharmacy=%.2f accepted=%v", v.Choice, v.Confidence, v.SamePharmacy, v.Accepted)
}

func TestCachedJudgeServesDecisions(t *testing.T) {
	var requests int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&requests, 1)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "jev-test",
			"answers": map[string]any{
				"g0_identity":         map[string]any{"type": "choice", "choice": "c1", "confidence": 0.9},
				"g0_c0_same_pharmacy": map[string]any{"type": "noul", "noul": 0.1},
				"g0_c1_same_pharmacy": map[string]any{"type": "noul", "noul": 0.95},
			},
			"usage": map[string]int{"input_tokens": 10, "output_tokens": 5},
		})
	}))
	defer srv.Close()

	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	cache, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatalf("cache: %v", err)
	}
	client := NewClient("test-key")
	client.BaseURL = srv.URL
	client.Model = "jev-test"
	judge := NewCachedJudge(client, cache)
	cfg := IdentityConfig{MinConfidence: 0.8, MinSamePharmacy: 0.8}
	ctx := context.Background()

	verdicts, err := judge.AdjudicateIdentity(ctx, []IdentityGroup{testIdentityGroup("KAAYΦATAKH EΛENH")}, cfg)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	if len(verdicts) != 1 || verdicts[0].Choice != "cand-b" || !verdicts[0].Accepted || verdicts[0].SamePharmacy != 0.95 {
		t.Fatalf("first verdict = %+v", verdicts)
	}
	if got := atomic.LoadInt64(&requests); got != 1 {
		t.Fatalf("requests = %d, want 1", got)
	}
	if judge.Model() != "jev-test" {
		t.Errorf("model = %q", judge.Model())
	}

	// same process, same cache: no new request
	if _, err := judge.AdjudicateIdentity(ctx, []IdentityGroup{testIdentityGroup("KAAYΦATAKH EΛENH")}, cfg); err != nil {
		t.Fatalf("second call: %v", err)
	}
	if got := atomic.LoadInt64(&requests); got != 1 {
		t.Fatalf("cached call made a request: %d", got)
	}

	// a new judge from the same file: still no request
	reloaded, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	judge2 := NewCachedJudge(client, reloaded)
	verdicts, err = judge2.AdjudicateIdentity(ctx, []IdentityGroup{testIdentityGroup("KAAYΦATAKH EΛENH")}, cfg)
	if err != nil {
		t.Fatalf("reloaded call: %v", err)
	}
	if verdicts[0].Choice != "cand-b" || !verdicts[0].Accepted {
		t.Errorf("reloaded verdict = %+v", verdicts[0])
	}
	if got := atomic.LoadInt64(&requests); got != 1 {
		t.Fatalf("reloaded cache made a request: %d", got)
	}

	// a different reading is a cache miss and is persisted
	if _, err := judge2.AdjudicateIdentity(ctx, []IdentityGroup{testIdentityGroup("ΑΛΛΟ ΚΕΙΜΕΝΟ")}, cfg); err != nil {
		t.Fatalf("miss call: %v", err)
	}
	if got := atomic.LoadInt64(&requests); got != 2 {
		t.Fatalf("miss did not request: %d", got)
	}
	if reloaded.Len() != 2 {
		t.Errorf("cache entries = %d, want 2", reloaded.Len())
	}
}
