package adjudicate

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// Identity adjudication implements the rerank pattern for the runtime
// pipeline (TYPESAFE_EVALUATION.md §3A): given one OCR entry and the catalog
// entries that share its phone number, decide which catalog entry the reading
// describes, or that none does. Code retrieves the candidates and consumes
// the answer; the judgment never writes a name, address or phone into the
// output — every accepted answer is an existing catalog entry or an explicit
// "none", and a caller keeps the deterministic pick as its fallback.
//
// One Choice question per entry carries the selection; one Noul per candidate
// reports whether the reading is a plausible rendering of that candidate and
// corroborates the selection. Groups are batched into a single request, so
// one call can settle every ambiguous entry of a schedule. Ask for entries
// that resolve to more than one catalog candidate; a single candidate needs
// no selection.

// IdentityNone is the Choice answer when no candidate fits the reading.
const IdentityNone = "none"

// IdentityReading is one OCR entry as the judgment sees it: the fields the
// parser reconstructed, before any catalog fill.
type IdentityReading struct {
	Name    string `json:"name,omitempty"`
	Address string `json:"address,omitempty"`
	Phone   string `json:"phone,omitempty"`
}

// IdentityCandidate is one catalog entry supplied by code. The ID is
// code-side only: the model sees names and addresses, never the ID, so a
// judgment cannot invent or leak a candidate identifier.
type IdentityCandidate struct {
	ID      string `json:"id"`
	Name    string `json:"name,omitempty"`
	Address string `json:"address,omitempty"`
}

// IdentityGroup is one OCR entry and the catalog candidates that share its
// phone number.
type IdentityGroup struct {
	Reading    IdentityReading     `json:"ocr_reading"`
	Candidates []IdentityCandidate `json:"catalog_candidates"`
}

// IdentityConfig gates when a Choice answer may replace the deterministic
// pick. Zero values disable the respective gate.
type IdentityConfig struct {
	// MinConfidence is the Choice confidence required to accept a candidate.
	MinConfidence float64
	// MinSamePharmacy is the same-pharmacy Noul required on the accepted
	// candidate.
	MinSamePharmacy float64
}

// IdentityVerdict is the judgment for one group. Choice carries the raw
// answer (a candidate ID or IdentityNone); Accepted applies the config's
// gates. SamePharmacy is -1 when the Noul was not answered.
type IdentityVerdict struct {
	Group         int                `json:"group"`
	Choice        string             `json:"choice"`
	Accepted      bool               `json:"accepted"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	SamePharmacy  float64            `json:"same_pharmacy"`
}

// AcceptIdentity reports whether a Choice answer may replace the deterministic
// pick. "none", a confidence below MinConfidence, or a same-pharmacy Noul
// below MinSamePharmacy all keep the caller's fallback path.
func AcceptIdentity(choice string, confidence, samePharmacy float64, cfg IdentityConfig) bool {
	if choice == IdentityNone {
		return false
	}
	if confidence < cfg.MinConfidence {
		return false
	}
	if samePharmacy < cfg.MinSamePharmacy {
		return false
	}
	return true
}

// AdjudicateIdentity evaluates all groups in one request. The returned
// verdicts are in group order.
func (c *Client) AdjudicateIdentity(ctx context.Context, groups []IdentityGroup, cfg IdentityConfig) ([]IdentityVerdict, *Response, error) {
	if len(groups) == 0 {
		return nil, nil, fmt.Errorf("adjudicate: no identity groups")
	}
	for gi, g := range groups {
		if len(g.Candidates) == 0 {
			return nil, nil, fmt.Errorf("adjudicate: group %d has no candidates", gi)
		}
	}

	state := map[string]any{"groups": identityState(groups)}
	questions := map[string]Question{}
	for gi, g := range groups {
		readPath := fmt.Sprintf("`groups[%d].ocr_reading`", gi)
		candPath := fmt.Sprintf("`groups[%d].catalog_candidates`", gi)

		criteria := make(map[string]any, len(g.Candidates)+1)
		for ci, cand := range g.Candidates {
			key := identityOptionKey(ci)
			criteria[key] = identityCandidateDescription(cand)
			qualifier := fmt.Sprintf("`groups[%d].catalog_candidates[%d]`", gi, ci)
			questions[fmt.Sprintf("g%d_%s_same_pharmacy", gi, key)] = Noul{
				Instructions: fmt.Sprintf(
					"Do %s (the pharmacy entry read by OCR) and %s describe one and the same pharmacy?",
					readPath, qualifier),
				Criteria: &NoulCriteria{
					True:  "The readings are a plausible noisy rendering of the candidate (mixed Latin/Greek script, missing accents, transliteration, abbreviation or minor spelling noise).",
					False: "The readings point to a different business, or are too garbled or incomplete to tell.",
				},
			}
		}
		criteria[IdentityNone] = "The readings do not match any candidate: a different business, or too garbled and incomplete to decide."
		questions[fmt.Sprintf("g%d_identity", gi)] = Choice{
			Instructions: fmt.Sprintf(
				"%s is one pharmacy entry read from a photographed duty schedule by OCR; the readings may mix Greek and Latin letters and contain recognition noise. Which entry of %s does the entry describe?",
				readPath, candPath),
			Criteria: criteria,
		}
	}

	resp, err := c.SystemOne(ctx, state, questions)
	if err != nil {
		return nil, nil, err
	}

	verdicts := make([]IdentityVerdict, 0, len(groups))
	for gi, g := range groups {
		answer, ok := resp.Choice(fmt.Sprintf("g%d_identity", gi))
		if !ok {
			return nil, resp, fmt.Errorf("adjudicate: missing choice answer g%d_identity", gi)
		}
		v := IdentityVerdict{
			Group:         gi,
			Choice:        IdentityNone,
			Confidence:    answer.Confidence,
			Probabilities: answer.Probabilities,
			SamePharmacy:  -1,
		}
		index, ok := identityOptionIndex(answer.Choice)
		if !ok {
			return nil, resp, fmt.Errorf("adjudicate: unexpected choice %q in g%d_identity", answer.Choice, gi)
		}
		switch {
		case index < 0:
			// IdentityNone: no candidate selected.
		case index >= len(g.Candidates):
			return nil, resp, fmt.Errorf("adjudicate: choice %q out of range in g%d_identity", answer.Choice, gi)
		default:
			v.Choice = g.Candidates[index].ID
			key := identityOptionKey(index)
			if noul, ok := resp.Noul(fmt.Sprintf("g%d_%s_same_pharmacy", gi, key)); ok {
				v.SamePharmacy = noul.Noul
			}
		}
		v.Accepted = AcceptIdentity(v.Choice, v.Confidence, v.SamePharmacy, cfg)
		verdicts = append(verdicts, v)
	}
	return verdicts, resp, nil
}

func identityOptionKey(index int) string {
	return fmt.Sprintf("c%d", index)
}

// identityOptionIndex parses a Choice answer. It accepts the option keys
// case-insensitively (c0, C1, none). ok is false for anything else; a
// negative index is IdentityNone.
func identityOptionIndex(choice string) (int, bool) {
	s := strings.ToLower(strings.TrimSpace(choice))
	if s == IdentityNone {
		return -1, true
	}
	if len(s) < 2 || s[0] != 'c' {
		return 0, false
	}
	n, err := strconv.Atoi(s[1:])
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

func identityCandidateDescription(c IdentityCandidate) string {
	switch {
	case c.Name != "" && c.Address != "":
		return fmt.Sprintf("The readings refer to %s (%s).", c.Name, c.Address)
	case c.Name != "":
		return fmt.Sprintf("The readings refer to %s.", c.Name)
	default:
		return "The readings refer to this candidate."
	}
}

// identityState renders the groups for the model without the code-side
// candidate IDs: the model sees the reading and the candidate names and
// addresses only.
func identityState(groups []IdentityGroup) []map[string]any {
	out := make([]map[string]any, len(groups))
	for gi, g := range groups {
		cands := make([]map[string]any, len(g.Candidates))
		for ci, cand := range g.Candidates {
			cands[ci] = map[string]any{"name": cand.Name, "address": cand.Address}
		}
		out[gi] = map[string]any{
			"ocr_reading": map[string]any{
				"name":    g.Reading.Name,
				"address": g.Reading.Address,
				"phone":   g.Reading.Phone,
			},
			"catalog_candidates": cands,
		}
	}
	return out
}
