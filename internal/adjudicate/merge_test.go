package adjudicate

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRouteMerge(t *testing.T) {
	cases := []struct {
		score, conf, min float64
		want             MergeOutcome
	}{
		{2.0, 0.9, 0.5, OutcomeSame},
		{1.6, 0.6, 0.5, OutcomeSame},
		{1.9, 0.4, 0.5, OutcomeCurator}, // confidence gate
		{2.0, 0.4, 0, OutcomeSame},      // gate disabled
		{1.4, 0.9, 0.5, OutcomeCurator},
		{0.6, 0.9, 0.5, OutcomeCurator},
		{0.4, 0.9, 0.5, OutcomeDifferent},
		{0.0, 0.9, 0.5, OutcomeDifferent},
	}
	for _, c := range cases {
		if got := RouteMerge(c.score, c.conf, c.min); got != c.want {
			t.Errorf("RouteMerge(%.1f, %.1f, %.1f) = %v, want %v", c.score, c.conf, c.min, got, c.want)
		}
	}
}

func TestAdjudicateMergeAgainstFakeServer(t *testing.T) {
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

		scoreOnly := !hasQuestion(body, "g0_r0_c0_same_name")
		legend := map[string]any{"0": MergeLevels[0], "1": MergeLevels[1], "2": MergeLevels[2]}
		answers := map[string]any{
			"g0_r0_c0_link_state": map[string]any{"type": "score", "score": 1.92, "confidence": 0.9,
				"legend": legend, "probabilities": map[string]float64{"0": 0.01, "1": 0.06, "2": 0.93}},
			"g0_r0_c1_link_state": map[string]any{"type": "score", "score": 0.2, "confidence": 0.9,
				"legend": legend, "probabilities": map[string]float64{"0": 0.85, "1": 0.1, "2": 0.05}},
		}
		if !scoreOnly {
			answers["g0_r0_c0_same_name"] = map[string]any{"type": "noul", "noul": 0.96}
			answers["g0_r0_c0_same_address"] = map[string]any{"type": "noul", "noul": 0.95}
			answers["g0_r0_c1_same_name"] = map[string]any{"type": "noul", "noul": 0.05}
			answers["g0_r0_c1_same_address"] = map[string]any{"type": "noul", "noul": 0.95}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "jev-test",
			"answers": answers,
			"usage":   map[string]int{"input_tokens": 100, "output_tokens": 10},
		})
	}))
	defer srv.Close()

	client := newTestClient(t, srv.URL, "jev-test")
	groups := []MergeGroup{{
		References: []MergeRef{{Name: "Βαρούχα - Αναγνωστάκης", Address: "Δημητρακάκη 21", Phone: "2831055212"}},
		Candidates: []MergeCandidate{
			{ID: "cand-0", Name: "Φαρμακείο Βαρούχα", Address: "Δημητρακάκη 21"},
			{ID: "cand-1", Name: "Φαρμακείο Καλαφατάκη", Address: "Δημητρακάκη 21"},
		},
	}}

	verdicts, resp, err := client.AdjudicateMerge(context.Background(), groups, MergeConfig{MinConfidence: 0.5})
	if err != nil {
		t.Fatalf("AdjudicateMerge: %v", err)
	}
	if len(verdicts) != 2 {
		t.Fatalf("verdicts = %d, want 2", len(verdicts))
	}
	if verdicts[0].Outcome != OutcomeSame || verdicts[1].Outcome != OutcomeDifferent {
		t.Errorf("outcomes = %v, %v; want same, different", verdicts[0].Outcome, verdicts[1].Outcome)
	}
	if verdicts[0].SameName != 0.96 || verdicts[0].SameAddress != 0.95 {
		t.Errorf("noul detail not carried: %+v", verdicts[0])
	}
	if verdicts[1].SameName != 0.05 {
		t.Errorf("second candidate noul = %v", verdicts[1].SameName)
	}
	if resp.Model != "jev-test" || resp.Usage.InputTokens != 100 {
		t.Errorf("response metadata: %+v", resp)
	}
	if got := len(bodies[0]["questions"].(map[string]any)); got != 6 {
		t.Errorf("questions = %d, want 6", got)
	}
	if got := bodies[0]["model"]; got != "jev-test" {
		t.Errorf("model = %v", got)
	}

	// ScoreOnly drops the curator-detail Nouls.
	if _, _, err := client.AdjudicateMerge(context.Background(), groups, MergeConfig{ScoreOnly: true}); err != nil {
		t.Fatalf("AdjudicateMerge score-only: %v", err)
	}
	if got := len(bodies[1]["questions"].(map[string]any)); got != 2 {
		t.Errorf("score-only questions = %d, want 2", got)
	}
}

func hasQuestion(body map[string]any, id string) bool {
	questions, ok := body["questions"].(map[string]any)
	if !ok {
		return false
	}
	_, ok = questions[id]
	return ok
}
