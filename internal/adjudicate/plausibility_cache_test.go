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

	jev "github.com/mheers/typesafeai-systemone-jev-go"
)

// TestCachedPlausibilityJudgeServesDecisions verifies the runtime path:
// misses are evaluated one request per entry, every decision is persisted,
// and a reloaded cache replays without touching the API.
func TestCachedPlausibilityJudgeServesDecisions(t *testing.T) {
	var requests int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&requests, 1)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "jev-test",
			"answers": map[string]any{
				"e0_name_plausible":    map[string]any{"type": "noul", "noul": 0.31},
				"e0_address_plausible": map[string]any{"type": "noul", "noul": 0.82},
			},
			"usage": map[string]int{"input_tokens": 100, "output_tokens": 5},
		})
	}))
	defer srv.Close()

	client := newTestClient(t, srv.URL, "jev-test")
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	cache, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatalf("cache: %v", err)
	}
	judge := NewCachedPlausibilityJudge(client, cache)
	entries := []PlausibilityEntry{
		{Name: "X", Address: "Y", Phone: "1"},
		{Name: "Z", Phone: "2"},
		{Address: "W", Phone: "3"},
	}

	got, err := judge.AdjudicatePlausibility(context.Background(), entries)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	if n := atomic.LoadInt64(&requests); n != 3 {
		t.Fatalf("requests = %d, want one per entry (3)", n)
	}
	if cache.Len() != 3 {
		t.Fatalf("cache entries = %d, want 3", cache.Len())
	}
	if v := got[0]; v.NamePlausible != 0.31 || v.AddressPlausible != 0.82 || v.Entry != 0 {
		t.Errorf("verdict 0 = %+v", v)
	}
	if v := got[1]; v.NamePlausible != 0.31 || v.AddressPlausible != -1 {
		t.Errorf("verdict 1 = %+v, want the unasked address at -1", v)
	}
	if v := got[2]; v.NamePlausible != -1 || v.AddressPlausible != 0.82 {
		t.Errorf("verdict 2 = %+v, want the unasked name at -1", v)
	}

	// In-memory replay: no new requests.
	replay, err := judge.AdjudicatePlausibility(context.Background(), entries)
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if n := atomic.LoadInt64(&requests); n != 3 {
		t.Fatalf("requests after replay = %d, want 3", n)
	}
	for i := range got {
		if replay[i] != got[i] {
			t.Errorf("replay[%d] = %+v, want %+v", i, replay[i], got[i])
		}
	}

	// Disk reload: the decisions are durable, determinism survives a restart.
	reloaded, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	fromDisk, err := NewCachedPlausibilityJudge(client, reloaded).AdjudicatePlausibility(context.Background(), entries)
	if err != nil {
		t.Fatalf("disk replay: %v", err)
	}
	if n := atomic.LoadInt64(&requests); n != 3 {
		t.Fatalf("requests after disk replay = %d, want 3", n)
	}
	for i := range got {
		if fromDisk[i] != got[i] {
			t.Errorf("fromDisk[%d] = %+v, want %+v", i, fromDisk[i], got[i])
		}
	}
}

// TestCachedPlausibilityJudgeEmptyFields covers entries nothing can be asked
// about and mixed empty fields.
func TestCachedPlausibilityJudgeEmptyFields(t *testing.T) {
	var requests int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&requests, 1)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "jev-test",
			"answers": map[string]any{
				"e0_name_plausible": map[string]any{"type": "noul", "noul": 0.7},
			},
		})
	}))
	defer srv.Close()

	client := newTestClient(t, srv.URL, "jev-test")
	cache, err := OpenDecisionCache(filepath.Join(t.TempDir(), "decisions.json"))
	if err != nil {
		t.Fatal(err)
	}
	judge := NewCachedPlausibilityJudge(client, cache)
	got, err := judge.AdjudicatePlausibility(context.Background(), []PlausibilityEntry{
		{},          // nothing to ask
		{Name: "X"}, // name only
	})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if n := atomic.LoadInt64(&requests); n != 1 {
		t.Fatalf("requests = %d, want 1 (the empty entry is not asked)", n)
	}
	if v := got[0]; v.NamePlausible != -1 || v.AddressPlausible != -1 {
		t.Errorf("empty entry = %+v, want -1 on both", v)
	}
	if v := got[1]; v.NamePlausible != 0.7 || v.AddressPlausible != -1 {
		t.Errorf("name-only entry = %+v", v)
	}
	if cache.Len() != 1 {
		t.Errorf("cache entries = %d, want 1 (the empty entry is not recorded)", cache.Len())
	}
}

// TestPlausibilityAndIdentityDecisionsShareTheCache pins the CLI's shared
// cache file: both decision kinds survive a save/load of one file.
func TestPlausibilityAndIdentityDecisionsShareTheCache(t *testing.T) {
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	cache, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatal(err)
	}
	entry := PlausibilityEntry{Name: "X", Address: "Y"}
	pKey, err := plausibilityCacheKey("jev-test", entry)
	if err != nil {
		t.Fatal(err)
	}
	pEntry, err := entryFromPlausibility(PlausibilityVerdict{NamePlausible: 0.4, AddressPlausible: 0.8}, "jev-test")
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.put(pKey, pEntry); err != nil {
		t.Fatal(err)
	}
	group := testIdentityGroup("X")
	iKey, err := cacheKey("jev-test", group)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.put(iKey, entryFromVerdict(group, IdentityVerdict{Choice: "cand-a"}, "jev-test")); err != nil {
		t.Fatal(err)
	}

	reloaded, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.Len() != 2 {
		t.Fatalf("cache entries = %d, want 2", reloaded.Len())
	}
	pe, ok := reloaded.Get(pKey)
	if !ok || pe.NamePlausible == nil || *pe.NamePlausible != 0.4 || pe.AddressPlausible == nil || *pe.AddressPlausible != 0.8 {
		t.Errorf("plausibility record = %+v (hit=%v)", pe, ok)
	}
	ie, ok := reloaded.Get(iKey)
	if !ok || ie.ChoiceIndex != 0 {
		t.Errorf("identity record = %+v (hit=%v)", ie, ok)
	}
}

func TestPlausibilityCacheKeyIsNamespaced(t *testing.T) {
	entry := PlausibilityEntry{Name: "X", Address: "Y"}
	key, err := plausibilityCacheKey("jev-test", entry)
	if err != nil {
		t.Fatal(err)
	}
	if len(key) < len("plaus:") || key[:len("plaus:")] != "plaus:" {
		t.Errorf("key = %q, want plaus: prefix", key)
	}
	// The same model and text must never produce another key: the cache is
	// only deterministic if the lookup is.
	again, err := plausibilityCacheKey("jev-test", entry)
	if err != nil {
		t.Fatal(err)
	}
	if key != again {
		t.Errorf("key not stable: %q vs %q", key, again)
	}
}

func TestEntryFromPlausibilityRejectsEmptyVerdict(t *testing.T) {
	if _, err := entryFromPlausibility(PlausibilityVerdict{NamePlausible: -1, AddressPlausible: -1}, "jev-test"); err == nil {
		t.Error("a verdict with no answered field must not be recorded")
	}
}

// TestCachedPlausibilityJudgeLive exercises the runtime verifier path against
// the real API: one request per entry, decisions persisted and replayed. It is
// skipped unless TYPESAFE_EXPERIMENT=1 and TYPESAFE_API_KEY are set.
func TestCachedPlausibilityJudgeLive(t *testing.T) {
	if os.Getenv("TYPESAFE_EXPERIMENT") == "" {
		t.Skip("set TYPESAFE_EXPERIMENT=1 to run the live TypeSafe experiment")
	}
	client, err := NewClientFromEnv(jev.WithModel(DefaultJudgeModel))
	if err != nil {
		t.Skipf("live experiment unavailable: %v", err)
	}
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	cache, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatalf("cache: %v", err)
	}
	judge := NewCachedPlausibilityJudge(client, cache)
	entries := []PlausibilityEntry{
		// the corpus's polluted-name case (§4.2): a name with a location
		// description, an address merged with its transliteration
		{Name: "KEPAMIANAKH ENANTIΣABOIΔAKH", Address: "MAPKOYOPTAAIOY20 20MARKOUPOTALIOUSTR", Phone: "2831051113"},
		// verbatim raw_ocr garbage: a phone line as the name, a shift as the address
		{Name: "THΛ.2831023347", Address: "08:00-21:00"},
	}
	got, err := judge.AdjudicatePlausibility(context.Background(), entries)
	if err != nil {
		t.Fatalf("live call: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("verdicts = %d, want 2", len(got))
	}
	for i, v := range got {
		if v.NamePlausible < 0 || v.NamePlausible > 1 || v.AddressPlausible < 0 || v.AddressPlausible > 1 {
			t.Errorf("verdict %d out of range: %+v", i, v)
		}
	}
	if cache.Len() != 2 {
		t.Fatalf("cache entries = %d, want 2", cache.Len())
	}
	reloaded, err := OpenDecisionCache(cachePath)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	replay, err := NewCachedPlausibilityJudge(client, reloaded).AdjudicatePlausibility(context.Background(), entries)
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	for i := range got {
		if replay[i] != got[i] {
			t.Errorf("replay[%d] = %+v, want the cached %+v", i, replay[i], got[i])
		}
	}
	t.Logf("polluted: name %.2f address %.2f; garbage: name %.2f address %.2f",
		got[0].NamePlausible, got[0].AddressPlausible, got[1].NamePlausible, got[1].AddressPlausible)
}
