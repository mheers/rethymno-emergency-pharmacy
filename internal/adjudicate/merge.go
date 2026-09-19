package adjudicate

import (
	"context"
	"encoding/json"
	"fmt"
)

// Merge adjudication implements the entity-alignment pattern for
// cmd/merge-golden: given a reference catalog entry and the Google-enriched
// golden records that share its phone number, decide whether each pair
// describes one and the same pharmacy.
//
// One Score question per pair carries the decision, because its three levels
// are the three things that can happen to a pair: leave it unlinked, send it
// to a curator, or merge it. A Noul question per compared field rides along to
// give the curator detail when the score lands in the middle. The caller
// routes on the score; there is no similarity threshold in the judgment path.

// MergeRef is one reference-catalog entry as the judgment sees it.
type MergeRef struct {
	Name    string `json:"name"`
	Address string `json:"address,omitempty"`
	Phone   string `json:"phone,omitempty"`
}

// MergeCandidate is one golden (Google-enriched) record.
type MergeCandidate struct {
	ID         string `json:"id"`
	Name       string `json:"name,omitempty"`        // name as enriched (Google)
	SourceName string `json:"source_name,omitempty"` // name from the upstream source
	Address    string `json:"address,omitempty"`     // formatted address, when available
	Phone      string `json:"phone,omitempty"`
}

// MergeGroup is one phone group: the reference entries and the golden
// candidates that share the number. Groups are batched into a single request,
// so one call can settle several ambiguous numbers at once.
type MergeGroup struct {
	References []MergeRef       `json:"reference_entries"`
	Candidates []MergeCandidate `json:"golden_candidates"`
}

// MergeOutcome is the verdict for one pair, one-to-one with the Score levels.
type MergeOutcome int

const (
	// OutcomeDifferent: the two records describe different pharmacies.
	OutcomeDifferent MergeOutcome = iota
	// OutcomeCurator: not safe to merge or drop; a person should decide.
	OutcomeCurator
	// OutcomeSame: the two records describe the same pharmacy.
	OutcomeSame
)

func (o MergeOutcome) String() string {
	switch o {
	case OutcomeDifferent:
		return "different"
	case OutcomeCurator:
		return "curator"
	case OutcomeSame:
		return "same"
	}
	return fmt.Sprintf("outcome(%d)", int(o))
}

// MarshalJSON renders the outcome as its name in reports.
func (o MergeOutcome) MarshalJSON() ([]byte, error) {
	return json.Marshal(o.String())
}

// MergeLevels are the Score levels, in order. They are the entire decision;
// code only rounds the probability-weighted score to the nearest level.
var MergeLevels = []string{
	"They describe two different pharmacies.",
	"They may describe the same pharmacy, but the evidence is incomplete or conflicting, or the names differ enough that they could also refer to a different business; a person should decide.",
	"They describe one and the same pharmacy.",
}

// MergeConfig tunes how a score becomes an outcome.
type MergeConfig struct {
	// MinConfidence is the confidence required for OutcomeSame. Below it, a
	// same-score is demoted to OutcomeCurator. Zero disables the gate.
	MinConfidence float64
	// ScoreOnly skips the curator-detail Noul questions, leaving one Score
	// question per pair. Used when batching many pairs into one request.
	ScoreOnly bool
}

// RouteMerge rounds the Score answer to the nearest level and applies the
// confidence gate.
func RouteMerge(score, confidence, minConfidence float64) MergeOutcome {
	level := int(score + 0.5)
	if level < 0 {
		level = 0
	}
	if level > len(MergeLevels)-1 {
		level = len(MergeLevels) - 1
	}
	out := MergeOutcome(level)
	if out == OutcomeSame && confidence < minConfidence {
		return OutcomeCurator
	}
	return out
}

// PairVerdict is the judgment for one (reference, golden) pair. Field-match
// Nouls are -1 when the question was not asked (for example, no address on
// one side).
type PairVerdict struct {
	Group       int          `json:"group"`
	RefIndex    int          `json:"ref_index"`
	Candidate   string       `json:"candidate"`
	Outcome     MergeOutcome `json:"outcome"`
	Score       float64      `json:"score"`
	Confidence  float64      `json:"confidence"`
	SameName    float64      `json:"same_name"`
	SameAddress float64      `json:"same_address"`
}

// AdjudicateMerge evaluates all groups in one request. The returned Response
// carries the model version and token usage for the caller's log.
func (c *Client) AdjudicateMerge(ctx context.Context, groups []MergeGroup, cfg MergeConfig) ([]PairVerdict, *Response, error) {
	if len(groups) == 0 {
		return nil, nil, fmt.Errorf("adjudicate: no merge groups")
	}

	state := map[string]any{"groups": mergeState(groups)}
	questions := map[string]Question{}
	for gi, g := range groups {
		for ri, ref := range g.References {
			for ci, cand := range g.Candidates {
				base := fmt.Sprintf("g%d_r%d_c%d", gi, ri, ci)
				refPath := fmt.Sprintf("`groups[%d].reference_entries[%d]`", gi, ri)
				candPath := fmt.Sprintf("`groups[%d].golden_candidates[%d]`", gi, ci)
				questions[base+"_link_state"] = Score{
					Instructions: fmt.Sprintf(
						"How do %s and %s relate as pharmacy records?", refPath, candPath),
					Criteria: MergeLevels,
				}
				if cfg.ScoreOnly {
					continue
				}
				questions[base+"_same_name"] = Noul{
					Instructions: fmt.Sprintf(
						"Do %s and %s state the same pharmacy name, or a recognizable variant of it (short form, word order, transliteration, or minor spelling noise)?",
						refPath, candPath),
					Criteria: &NoulCriteria{
						True:  "The names are the same or a plausible variant of one another.",
						False: "The names point to different businesses.",
					},
				}
				if ref.Address != "" && cand.Address != "" {
					questions[base+"_same_address"] = Noul{
						Instructions: fmt.Sprintf(
							"Do %s and %s give the same street address, allowing formatting differences?",
							refPath, candPath),
					}
				}
			}
		}
	}

	resp, err := c.SystemOne(ctx, state, questions)
	if err != nil {
		return nil, nil, err
	}

	var verdicts []PairVerdict
	for gi, g := range groups {
		for ri := range g.References {
			for ci, cand := range g.Candidates {
				base := fmt.Sprintf("g%d_r%d_c%d", gi, ri, ci)
				link, ok := resp.Answers[base+"_link_state"]
				if !ok || link.Type != "score" {
					return nil, resp, fmt.Errorf("adjudicate: missing score answer %s_link_state", base)
				}
				v := PairVerdict{
					Group:       gi,
					RefIndex:    ri,
					Candidate:   cand.ID,
					Outcome:     RouteMerge(link.Score, link.Confidence, cfg.MinConfidence),
					Score:       link.Score,
					Confidence:  link.Confidence,
					SameName:    -1,
					SameAddress: -1,
				}
				if a, ok := resp.Answers[base+"_same_name"]; ok && a.Type == "noul" {
					v.SameName = a.Noul
				}
				if a, ok := resp.Answers[base+"_same_address"]; ok && a.Type == "noul" {
					v.SameAddress = a.Noul
				}
				verdicts = append(verdicts, v)
			}
		}
	}
	return verdicts, resp, nil
}

// mergeState renders the groups for the model without the code-side candidate
// IDs: the model sees names, addresses and phones only.
func mergeState(groups []MergeGroup) []map[string]any {
	out := make([]map[string]any, len(groups))
	for gi, g := range groups {
		refs := make([]map[string]any, len(g.References))
		for ri, ref := range g.References {
			refs[ri] = map[string]any{"name": ref.Name, "address": ref.Address, "phone": ref.Phone}
		}
		cands := make([]map[string]any, len(g.Candidates))
		for ci, cand := range g.Candidates {
			cands[ci] = map[string]any{
				"name":        cand.Name,
				"source_name": cand.SourceName,
				"address":     cand.Address,
				"phone":       cand.Phone,
			}
		}
		out[gi] = map[string]any{"reference_entries": refs, "golden_candidates": cands}
	}
	return out
}
