package rethymnoemergency

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
)

// testCaches returns a fresh cache registry, as New builds for one client.
func testCaches() *judgeCaches {
	return &judgeCaches{byPath: map[string]*adjudicate.DecisionCache{}}
}

// TestBuildIdentityJudgeDefaults verifies the opt-in judge wiring: defaults,
// the pinned model, and that the returned judge caches through a file.
func TestBuildIdentityJudgeDefaults(t *testing.T) {
	t.Setenv(adjudicate.APIKeyEnv, "test-key")
	caches := testCaches()
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	judge, gate, err := buildIdentityJudge(&IdentityJudgeConfig{CachePath: cachePath}, caches.open)
	if err != nil {
		t.Fatalf("buildIdentityJudge: %v", err)
	}
	if judge == nil {
		t.Fatal("nil judge")
	}
	if gate.MinConfidence != defaultIdentityGate || gate.MinSamePharmacy != defaultIdentityGate {
		t.Errorf("gates = %+v, want %v on both", gate, defaultIdentityGate)
	}
	cached, ok := judge.(*adjudicate.CachedJudge)
	if !ok {
		t.Fatalf("judge = %T, want *adjudicate.CachedJudge", judge)
	}
	if cached.Model() != adjudicate.DefaultJudgeModel {
		t.Errorf("model = %q, want %q", cached.Model(), adjudicate.DefaultJudgeModel)
	}

	// explicit gates win over the defaults
	_, gate, err = buildIdentityJudge(&IdentityJudgeConfig{CachePath: cachePath, MinConfidence: 0.9, MinSamePharmacy: 0.7}, caches.open)
	if err != nil {
		t.Fatalf("buildIdentityJudge: %v", err)
	}
	if gate.MinConfidence != 0.9 || gate.MinSamePharmacy != 0.7 {
		t.Errorf("explicit gates = %+v", gate)
	}

	// no config means no judge: the deterministic path is untouched
	judge, gate, err = buildIdentityJudge(nil, caches.open)
	if err != nil || judge != nil || gate != (adjudicate.IdentityConfig{}) {
		t.Errorf("nil config: judge = %v, gate = %+v, err = %v", judge, gate, err)
	}
}

func TestBuildIdentityJudgeValidation(t *testing.T) {
	caches := testCaches()
	if _, _, err := buildIdentityJudge(&IdentityJudgeConfig{}, caches.open); err == nil || !strings.Contains(err.Error(), "CachePath") {
		t.Errorf("missing cache path: err = %v", err)
	}

	t.Setenv(adjudicate.APIKeyEnv, "")
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	if _, _, err := buildIdentityJudge(&IdentityJudgeConfig{CachePath: cachePath}, caches.open); err == nil || !strings.Contains(err.Error(), adjudicate.APIKeyEnv) {
		t.Errorf("missing API key: err = %v", err)
	}

	t.Setenv(adjudicate.APIKeyEnv, "test-key")
	if err := os.WriteFile(cachePath, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := buildIdentityJudge(&IdentityJudgeConfig{CachePath: cachePath}, caches.open); err == nil || !strings.Contains(err.Error(), "cache") {
		t.Errorf("corrupt cache: err = %v", err)
	}
}

// TestBuildPlausibilityJudgeDefaults verifies the opt-in verifier wiring: the
// measured gates (name 0.5, address 0.7), the pinned model, and the nil path.
func TestBuildPlausibilityJudgeDefaults(t *testing.T) {
	t.Setenv(adjudicate.APIKeyEnv, "test-key")
	caches := testCaches()
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	judge, gate, err := buildPlausibilityJudge(&PlausibilityJudgeConfig{CachePath: cachePath}, caches.open)
	if err != nil {
		t.Fatalf("buildPlausibilityJudge: %v", err)
	}
	if judge == nil {
		t.Fatal("nil judge")
	}
	if gate.MinName != defaultPlausibilityNameGate || gate.MinAddress != defaultPlausibilityAddressGate {
		t.Errorf("gates = %+v, want name %v address %v", gate, defaultPlausibilityNameGate, defaultPlausibilityAddressGate)
	}
	cached, ok := judge.(*adjudicate.CachedPlausibilityJudge)
	if !ok {
		t.Fatalf("judge = %T, want *adjudicate.CachedPlausibilityJudge", judge)
	}
	if cached.Model() != adjudicate.DefaultJudgeModel {
		t.Errorf("model = %q, want %q", cached.Model(), adjudicate.DefaultJudgeModel)
	}

	// explicit gates win over the defaults
	_, gate, err = buildPlausibilityJudge(&PlausibilityJudgeConfig{CachePath: cachePath, MinName: 0.6, MinAddress: 0.8}, caches.open)
	if err != nil {
		t.Fatalf("buildPlausibilityJudge: %v", err)
	}
	if gate.MinName != 0.6 || gate.MinAddress != 0.8 {
		t.Errorf("explicit gates = %+v", gate)
	}

	// no config means no judge: the deterministic path is untouched
	judge, gate, err = buildPlausibilityJudge(nil, caches.open)
	if err != nil || judge != nil || gate != (adjudicate.PlausibilityConfig{}) {
		t.Errorf("nil config: judge = %v, gate = %+v, err = %v", judge, gate, err)
	}
}

func TestBuildPlausibilityJudgeValidation(t *testing.T) {
	caches := testCaches()
	if _, _, err := buildPlausibilityJudge(&PlausibilityJudgeConfig{}, caches.open); err == nil || !strings.Contains(err.Error(), "CachePath") {
		t.Errorf("missing cache path: err = %v", err)
	}

	t.Setenv(adjudicate.APIKeyEnv, "")
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	if _, _, err := buildPlausibilityJudge(&PlausibilityJudgeConfig{CachePath: cachePath}, caches.open); err == nil || !strings.Contains(err.Error(), adjudicate.APIKeyEnv) {
		t.Errorf("missing API key: err = %v", err)
	}

	t.Setenv(adjudicate.APIKeyEnv, "test-key")
	if err := os.WriteFile(cachePath, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := buildPlausibilityJudge(&PlausibilityJudgeConfig{CachePath: cachePath}, caches.open); err == nil || !strings.Contains(err.Error(), "cache") {
		t.Errorf("corrupt cache: err = %v", err)
	}
}

// TestJudgeCachesShareInstance guards the shared-file case: the CLI enables
// both judges with one cache path, and they must not keep two in-memory maps
// that overwrite each other on save.
func TestJudgeCachesShareInstance(t *testing.T) {
	caches := testCaches()
	dir := t.TempDir()
	path := filepath.Join(dir, "decisions.json")
	first, err := caches.open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	second, err := caches.open(path)
	if err != nil {
		t.Fatalf("open again: %v", err)
	}
	if first != second {
		t.Error("the same cache path returned two instances")
	}
	other, err := caches.open(filepath.Join(dir, "other.json"))
	if err != nil {
		t.Fatalf("open other: %v", err)
	}
	if other == first {
		t.Error("different cache paths returned one instance")
	}
}
