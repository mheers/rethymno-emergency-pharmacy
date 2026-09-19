package adjudicate

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// TestIdentityCaseDataset guards the live experiment's dataset builder: every
// expected candidate exists in its group, ambiguous groups carry both
// entries, and the no-pick readings are present. It runs hermetically.
func TestIdentityCaseDataset(t *testing.T) {
	refs, err := validate.LoadReference(validate.ReferenceJSON)
	if err != nil {
		t.Fatalf("load reference catalog: %v", err)
	}
	cases := buildIdentityCases(t, refs)
	if len(cases) < 40 {
		t.Fatalf("dataset has %d readings; want at least 40", len(cases))
	}
	kinds := map[string]int{}
	for _, c := range cases {
		kinds[c.kind]++
		if got := len(c.group.Candidates); got != 2 {
			t.Errorf("case %q: %d candidates, want 2", c.name, got)
		}
		if c.group.Reading.Phone == "" {
			t.Errorf("case %q: missing phone", c.name)
		}
		if c.expect == "" {
			continue
		}
		found := false
		for _, cand := range c.group.Candidates {
			if cand.ID == c.expect {
				found = true
			}
		}
		if !found {
			t.Errorf("case %q: expect %q is not a candidate", c.name, c.expect)
		}
		if c.group.Reading.Name == "" {
			t.Errorf("case %q: decisive reading without a name", c.name)
		}
	}
	for _, kind := range []string{"exact", "short", "latin", "lookalike", "typo", "address-only", "street-name", "foreign-name", "unrelated"} {
		if kinds[kind] == 0 {
			t.Errorf("dataset lacks kind %q", kind)
		}
	}
	t.Logf("%d readings: %v", len(cases), kinds)
}

func TestAcceptIdentity(t *testing.T) {
	cases := []struct {
		choice       string
		conf, noul   float64
		cfg          IdentityConfig
		wantAccepted bool
	}{
		{"cand-a", 0.80, 0.90, IdentityConfig{MinConfidence: 0.5, MinSamePharmacy: 0.5}, true},
		{IdentityNone, 0.99, 0.99, IdentityConfig{MinConfidence: 0.5, MinSamePharmacy: 0.5}, false},
		{"cand-a", 0.40, 0.90, IdentityConfig{MinConfidence: 0.5, MinSamePharmacy: 0.5}, false},
		{"cand-a", 0.80, 0.40, IdentityConfig{MinConfidence: 0.5, MinSamePharmacy: 0.5}, false},
		{"cand-a", 0.80, 0.90, IdentityConfig{}, true}, // gates disabled
		{"cand-a", 0.00, -1, IdentityConfig{}, false},  // missing noul never accepts
		{IdentityNone, 0.99, -1, IdentityConfig{}, false},
	}
	for _, c := range cases {
		if got := AcceptIdentity(c.choice, c.conf, c.noul, c.cfg); got != c.wantAccepted {
			t.Errorf("AcceptIdentity(%q, %.2f, %.2f, %+v) = %v, want %v",
				c.choice, c.conf, c.noul, c.cfg, got, c.wantAccepted)
		}
	}
}

func TestIdentityOptionIndex(t *testing.T) {
	cases := []struct {
		in     string
		want   int
		wantOK bool
	}{
		{"c0", 0, true},
		{"C12", 12, true},
		{" none ", -1, true},
		{"café", 0, false},
		{"candidate-a", 0, false},
		{"", 0, false},
	}
	for _, c := range cases {
		got, ok := identityOptionIndex(c.in)
		if got != c.want || ok != c.wantOK {
			t.Errorf("identityOptionIndex(%q) = (%d, %v), want (%d, %v)", c.in, got, ok, c.want, c.wantOK)
		}
	}
}

func TestAdjudicateIdentityAgainstFakeServer(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/systemone" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		bodies = append(bodies, body)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "jev-test",
			"answers": map[string]any{
				"g0_identity": map[string]any{
					"type": "choice", "choice": "c1", "confidence": 0.83,
					"probabilities": map[string]float64{"c0": 0.1, "c1": 0.85, "none": 0.05},
				},
				"g0_c0_same_pharmacy": map[string]any{"type": "noul", "noul": 0.05},
				"g0_c1_same_pharmacy": map[string]any{"type": "noul", "noul": 0.96},
				// case-insensitive answers
				"g1_identity": map[string]any{
					"type": "choice", "choice": "NONE", "confidence": 0.7,
					"probabilities": map[string]float64{"none": 0.7, "c0": 0.2, "c1": 0.1},
				},
				"g1_c0_same_pharmacy": map[string]any{"type": "noul", "noul": 0.2},
				"g1_c1_same_pharmacy": map[string]any{"type": "noul", "noul": 0.1},
			},
			"usage": map[string]int{"input_tokens": 900, "output_tokens": 40},
		})
	}))
	defer srv.Close()

	client := newTestClient(t, srv.URL, "jev-test")

	groups := []IdentityGroup{
		{
			Reading: IdentityReading{Name: "KAAYΦATAKH EΛENH", Address: "ΔHMHTPAKAKH 21", Phone: "2831055212"},
			Candidates: []IdentityCandidate{
				{ID: "varoucha", Name: "Βαρούχα - Αναγνωστάκης", Address: "Δημητρακάκη 21"},
				{ID: "kalafataki", Name: "Καλαφατάκη Ελένη - Γεωργία", Address: "Δημητρακάκη 21"},
			},
		},
		{
			Reading: IdentityReading{Name: "", Address: "Δημητρακάκη 21", Phone: "2831055212"},
			Candidates: []IdentityCandidate{
				{ID: "varoucha", Name: "Βαρούχα - Αναγνωστάκης", Address: "Δημητρακάκη 21"},
				{ID: "kalafataki", Name: "Καλαφατάκη Ελένη - Γεωργία", Address: "Δημητρακάκη 21"},
			},
		},
	}

	verdicts, resp, err := client.AdjudicateIdentity(context.Background(), groups, IdentityConfig{MinConfidence: 0.5, MinSamePharmacy: 0.5})
	if err != nil {
		t.Fatalf("AdjudicateIdentity: %v", err)
	}
	if len(verdicts) != 2 {
		t.Fatalf("verdicts = %d, want 2", len(verdicts))
	}
	v0 := verdicts[0]
	if v0.Choice != "kalafataki" || !v0.Accepted {
		t.Errorf("v0 = %+v, want choice kalafataki accepted", v0)
	}
	if v0.Confidence != 0.83 || v0.SamePharmacy != 0.96 {
		t.Errorf("v0 confidence/noul = %.2f/%.2f, want 0.83/0.96", v0.Confidence, v0.SamePharmacy)
	}
	if v0.Probabilities["c1"] != 0.85 {
		t.Errorf("v0 probabilities not carried: %+v", v0.Probabilities)
	}
	if v1 := verdicts[1]; v1.Choice != IdentityNone || v1.Accepted {
		t.Errorf("v1 = %+v, want none and not accepted", v1)
	}
	if resp.Model != "jev-test" || resp.Usage.InputTokens != 900 {
		t.Errorf("response metadata: %+v", resp)
	}

	// ---- request shape ----
	body := bodies[0]
	if got := body["model"]; got != "jev-test" {
		t.Errorf("model = %v", got)
	}
	questions := body["questions"].(map[string]any)
	for _, id := range []string{"g0_identity", "g0_c0_same_pharmacy", "g0_c1_same_pharmacy", "g1_identity"} {
		if _, ok := questions[id]; !ok {
			t.Errorf("question %s missing", id)
		}
	}
	if got := len(questions); got != 6 {
		t.Errorf("questions = %d, want 6", got)
	}
	choice := questions["g0_identity"].(map[string]any)
	if choice["type"] != "choice" {
		t.Errorf("g0_identity type = %v", choice["type"])
	}
	criteria := choice["criteria"].(map[string]any)
	for _, key := range []string{"c0", "c1", IdentityNone} {
		if _, ok := criteria[key]; !ok {
			t.Errorf("criteria %q missing: %v", key, criteria)
		}
	}

	state := body["state"].(map[string]any)
	stateGroups := state["groups"].([]any)
	stateGroup0 := stateGroups[0].(map[string]any)
	cands := stateGroup0["catalog_candidates"].([]any)
	if len(cands) != 2 {
		t.Fatalf("state candidates = %d, want 2", len(cands))
	}
	if _, leaked := cands[0].(map[string]any)["id"]; leaked {
		t.Errorf("state leaks candidate IDs: %v", cands[0])
	}
	reading := stateGroup0["ocr_reading"].(map[string]any)
	if reading["name"] != "KAAYΦATAKH EΛENH" || reading["phone"] != "2831055212" {
		t.Errorf("state reading = %v", reading)
	}
}

func TestAdjudicateIdentityRejectsBadAnswers(t *testing.T) {
	cases := []struct {
		name    string
		answers map[string]any
	}{
		{
			name: "unknown choice",
			answers: map[string]any{
				"g0_identity": map[string]any{
					"type": "choice", "choice": "candidate-a", "confidence": 0.9,
					"probabilities": map[string]float64{"candidate-a": 0.9},
				},
				"g0_c0_same_pharmacy": map[string]any{"type": "noul", "noul": 0.9},
			},
		},
		{
			name: "choice out of range",
			answers: map[string]any{
				"g0_identity": map[string]any{
					"type": "choice", "choice": "c7", "confidence": 0.9,
					"probabilities": map[string]float64{"c7": 0.9},
				},
				"g0_c0_same_pharmacy": map[string]any{"type": "noul", "noul": 0.9},
			},
		},
		{
			name:    "missing choice",
			answers: map[string]any{},
		},
		{
			name: "wrong answer type",
			answers: map[string]any{
				"g0_identity": map[string]any{"type": "noul", "noul": 0.9},
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"model":   "jev-test",
					"answers": c.answers,
					"usage":   map[string]int{"input_tokens": 1, "output_tokens": 1},
				})
			}))
			defer srv.Close()

			client := newTestClient(t, srv.URL, "jev-test")
			groups := []IdentityGroup{{
				Reading:    IdentityReading{Name: "X", Phone: "2831055212"},
				Candidates: []IdentityCandidate{{ID: "a", Name: "A"}, {ID: "b", Name: "B"}},
			}}
			if _, _, err := client.AdjudicateIdentity(context.Background(), groups, IdentityConfig{}); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}
