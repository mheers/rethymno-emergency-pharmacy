package adjudicate

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestPlausibilityCaseDataset guards the live experiment's dataset: the
// fixture loads, every entry carries both fields and a known quality label,
// and the reported classes are present. It runs hermetically.
func TestPlausibilityCaseDataset(t *testing.T) {
	cases, err := loadPlausibilityCases()
	if err != nil {
		t.Fatalf("load dataset: %v", err)
	}
	if len(cases) < 70 {
		t.Fatalf("dataset has %d entries; want at least 70", len(cases))
	}
	sources := map[string]int{}
	names := map[string]int{}
	addrs := map[string]int{}
	for _, c := range cases {
		sources[c.Source]++
		names[c.NameQuality]++
		addrs[c.AddressQuality]++
		if c.Entry.Name == "" || c.Entry.Address == "" {
			t.Errorf("entry %s: empty field", c.ID)
		}
	}
	if sources["corpus"] == 0 || sources["synthetic"] == 0 {
		t.Errorf("sources = %v, want corpus and synthetic entries", sources)
	}
	for _, q := range []string{qualityClean, qualityGarbage, qualityPolluted} {
		if names[q] == 0 {
			t.Errorf("no %s name fields", q)
		}
	}
	if addrs[qualityClean] == 0 || addrs[qualityGarbage] == 0 {
		t.Errorf("address qualities = %v, want clean and garbage fields", addrs)
	}
	t.Logf("%d entries: names %v, addresses %v", len(cases), names, addrs)
}

func TestImplausibleFields(t *testing.T) {
	cfg := PlausibilityConfig{MinName: 0.5, MinAddress: 0.5}
	cases := []struct {
		name string
		v    PlausibilityVerdict
		want []string
	}{
		{"both plausible", PlausibilityVerdict{NamePlausible: 0.91, AddressPlausible: 0.88}, nil},
		{"name garbage", PlausibilityVerdict{NamePlausible: 0.12, AddressPlausible: 0.7}, []string{"name"}},
		{"address garbage", PlausibilityVerdict{NamePlausible: 0.9, AddressPlausible: 0.05}, []string{"address"}},
		{"both garbage", PlausibilityVerdict{NamePlausible: 0.1, AddressPlausible: 0.2}, []string{"name", "address"}},
		{"unasked fields never flag", PlausibilityVerdict{NamePlausible: -1, AddressPlausible: -1}, nil},
		{"gates disabled", PlausibilityVerdict{NamePlausible: 0.01, AddressPlausible: 0.01}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			use := cfg
			if c.name == "gates disabled" {
				use = PlausibilityConfig{}
			}
			got := c.v.ImplausibleFields(use)
			if len(got) != len(c.want) {
				t.Fatalf("ImplausibleFields = %v, want %v", got, c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Fatalf("ImplausibleFields = %v, want %v", got, c.want)
				}
			}
		})
	}
}

func TestAdjudicatePlausibilityAgainstFakeServer(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/systemone" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		bodies = append(bodies, body)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "jev-test",
			"answers": map[string]any{
				"e0_name_plausible":    map[string]any{"type": "noul", "noul": 0.94},
				"e0_address_plausible": map[string]any{"type": "noul", "noul": 0.88},
				"e1_name_plausible":    map[string]any{"type": "noul", "noul": 0.08},
			},
			"usage": map[string]int{"input_tokens": 210, "output_tokens": 12},
		})
	}))
	defer srv.Close()

	client := newTestClient(t, srv.URL, "jev-test")

	entries := []PlausibilityEntry{
		{Name: "ΠΑΠΑΤΖΑΝΗ ΜΑΡΙΑ", Address: "Γερακάρη 96", Phone: "2831023347"},
		{Name: "TEPAKAPH96 96GERAKARISTR", Phone: "2831023347"}, // address empty: no address question
	}
	verdicts, resp, err := client.AdjudicatePlausibility(context.Background(), entries)
	if err != nil {
		t.Fatalf("AdjudicatePlausibility: %v", err)
	}
	if len(verdicts) != 2 {
		t.Fatalf("verdicts = %d, want 2", len(verdicts))
	}
	if v := verdicts[0]; v.NamePlausible != 0.94 || v.AddressPlausible != 0.88 {
		t.Errorf("verdict 0 = %+v, want 0.94/0.88", v)
	}
	if v := verdicts[1]; v.NamePlausible != 0.08 || v.AddressPlausible != -1 {
		t.Errorf("verdict 1 = %+v, want 0.08/-1 (unasked)", v)
	}
	if resp.Model != "jev-test" || resp.Usage.InputTokens != 210 {
		t.Errorf("response metadata: %+v", resp)
	}

	// ---- request shape: one Noul per non-empty field, no question for the
	// empty address ----
	if len(bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(bodies))
	}
	questions := bodies[0]["questions"].(map[string]any)
	if len(questions) != 3 {
		t.Fatalf("questions = %d, want 3: %v", len(questions), questions)
	}
	for _, id := range []string{"e0_name_plausible", "e0_address_plausible", "e1_name_plausible"} {
		q, ok := questions[id].(map[string]any)
		if !ok {
			t.Fatalf("question %s missing", id)
		}
		if q["type"] != "noul" {
			t.Errorf("question %s type = %v, want noul", id, q["type"])
		}
	}
	if _, ok := questions["e1_address_plausible"]; ok {
		t.Error("asked for an empty address")
	}
	state := bodies[0]["state"].(map[string]any)
	stateEntries := state["entries"].([]any)
	if len(stateEntries) != 2 {
		t.Fatalf("state entries = %d, want 2", len(stateEntries))
	}
	if got := stateEntries[0].(map[string]any)["name"]; got != "ΠΑΠΑΤΖΑΝΗ ΜΑΡΙΑ" {
		t.Errorf("state name = %v", got)
	}
}

func TestAdjudicatePlausibilityRejectsBadAnswers(t *testing.T) {
	cases := []struct {
		name    string
		answers map[string]any
	}{
		{
			name:    "missing name noul",
			answers: map[string]any{},
		},
		{
			name: "wrong answer type",
			answers: map[string]any{
				"e0_name_plausible": map[string]any{
					"type": "choice", "choice": "c0", "confidence": 0.9,
					"probabilities": map[string]float64{"c0": 0.9},
				},
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
			if _, _, err := client.AdjudicatePlausibility(context.Background(), []PlausibilityEntry{{Name: "X"}}); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

func TestAdjudicatePlausibilityValidation(t *testing.T) {
	client, err := NewClient("test-key")
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, _, err := client.AdjudicatePlausibility(context.Background(), nil); err == nil {
		t.Error("no entries: expected an error")
	}
	if _, _, err := client.AdjudicatePlausibility(context.Background(), []PlausibilityEntry{{}}); err == nil {
		t.Error("no fields: expected an error")
	}
}
