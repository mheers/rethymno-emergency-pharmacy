package rethymnoemergency

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
)

// TestBuildIdentityJudgeDefaults verifies the opt-in judge wiring: defaults,
// the pinned model, and that the returned judge caches through a file.
func TestBuildIdentityJudgeDefaults(t *testing.T) {
	t.Setenv(adjudicate.APIKeyEnv, "test-key")
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	judge, gate, err := buildIdentityJudge(&IdentityJudgeConfig{CachePath: cachePath})
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
	_, gate, err = buildIdentityJudge(&IdentityJudgeConfig{CachePath: cachePath, MinConfidence: 0.9, MinSamePharmacy: 0.7})
	if err != nil {
		t.Fatalf("buildIdentityJudge: %v", err)
	}
	if gate.MinConfidence != 0.9 || gate.MinSamePharmacy != 0.7 {
		t.Errorf("explicit gates = %+v", gate)
	}

	// no config means no judge: the deterministic path is untouched
	judge, gate, err = buildIdentityJudge(nil)
	if err != nil || judge != nil || gate != (adjudicate.IdentityConfig{}) {
		t.Errorf("nil config: judge = %v, gate = %+v, err = %v", judge, gate, err)
	}
}

func TestBuildIdentityJudgeValidation(t *testing.T) {
	if _, _, err := buildIdentityJudge(&IdentityJudgeConfig{}); err == nil || !strings.Contains(err.Error(), "CachePath") {
		t.Errorf("missing cache path: err = %v", err)
	}

	t.Setenv(adjudicate.APIKeyEnv, "")
	cachePath := filepath.Join(t.TempDir(), "decisions.json")
	if _, _, err := buildIdentityJudge(&IdentityJudgeConfig{CachePath: cachePath}); err == nil || !strings.Contains(err.Error(), adjudicate.APIKeyEnv) {
		t.Errorf("missing API key: err = %v", err)
	}

	t.Setenv(adjudicate.APIKeyEnv, "test-key")
	if err := os.WriteFile(cachePath, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := buildIdentityJudge(&IdentityJudgeConfig{CachePath: cachePath}); err == nil || !strings.Contains(err.Error(), "cache") {
		t.Errorf("corrupt cache: err = %v", err)
	}
}
