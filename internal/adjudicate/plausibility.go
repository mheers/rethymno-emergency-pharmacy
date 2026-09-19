package adjudicate

import (
	"context"
	"errors"
	"fmt"
)

// Plausibility verification implements the warnings-only judgment for entries
// the catalog cannot rescue (TYPESAFE_EVALUATION.md §3D): given one OCR entry
// that did not match the reference catalog, ask whether its name is a
// plausible Greek pharmacy name and whether its address is a plausible street
// address in or near Rethymno. The judgment reports probabilities; code turns
// them into warnings. It never rewrites a field, never generates a name or
// address, and the deterministic confidence formula stays in code.
//
// Two Noul questions per entry (one per non-empty field) are asked in one
// request per entry: the earlier experiments showed batching many independent
// units into one request degrades the answers (§4, §4.1).

// PlausibilityEntry is one OCR entry as the judgment sees it: the fields the
// parser produced, for a pharmacy whose phone number did not match the
// reference catalog.
type PlausibilityEntry struct {
	Name    string `json:"name,omitempty"`
	Address string `json:"address,omitempty"`
	Phone   string `json:"phone,omitempty"`
}

// PlausibilityVerdict is the judgment for one entry. Each probability is -1
// when the field was empty and the question was not asked; higher means more
// plausible.
type PlausibilityVerdict struct {
	Entry            int     `json:"entry"`
	NamePlausible    float64 `json:"name_plausible"`
	AddressPlausible float64 `json:"address_plausible"`
}

// PlausibilityConfig gates the Noul answers when a caller turns them into
// flags. Zero disables the respective gate.
type PlausibilityConfig struct {
	// MinName is the name-plausibility probability at or above which the
	// name is accepted as plausible.
	MinName float64
	// MinAddress is the address-plausibility probability at or above which
	// the address is accepted as plausible.
	MinAddress float64
}

// ImplausibleFields applies the gates and returns the field names ("name",
// "address") whose probability is below their gate. Fields that were not
// asked (empty) never come back: the caller raises warnings, never rewrites.
func (v PlausibilityVerdict) ImplausibleFields(cfg PlausibilityConfig) []string {
	var out []string
	if v.NamePlausible >= 0 && v.NamePlausible < cfg.MinName {
		out = append(out, "name")
	}
	if v.AddressPlausible >= 0 && v.AddressPlausible < cfg.MinAddress {
		out = append(out, "address")
	}
	return out
}

// AdjudicatePlausibility evaluates all entries in one request. The returned
// verdicts are in entry order.
func (c *Client) AdjudicatePlausibility(ctx context.Context, entries []PlausibilityEntry) ([]PlausibilityVerdict, *Response, error) {
	if len(entries) == 0 {
		return nil, nil, errors.New("adjudicate: no plausibility entries")
	}
	questions := 0
	for _, e := range entries {
		if e.Name != "" {
			questions++
		}
		if e.Address != "" {
			questions++
		}
	}
	if questions == 0 {
		return nil, nil, errors.New("adjudicate: no plausibility fields to judge")
	}

	state := map[string]any{"entries": plausibilityState(entries)}
	qs := map[string]Question{}
	for ei, e := range entries {
		path := fmt.Sprintf("`entries[%d]`", ei)
		if e.Name != "" {
			qs[fmt.Sprintf("e%d_name_plausible", ei)] = Noul{
				Instructions: fmt.Sprintf(
					"%s.name is the name field of one pharmacy entry read from a photographed Greek duty schedule by OCR; the text may mix Greek and Latin letters, drop accents, or contain recognition noise. Is it a plausible Greek pharmacy name?",
					path),
				Criteria: &NoulCriteria{
					True:  "The field holds only a Greek pharmacy name (typically one or two person names), even with OCR noise, abbreviation or transliteration.",
					False: "The field is not a name: it is a weekday label, a shift time or date, a document title or table header, a phone number, an address or location description (for example \"opposite X\" or \"below Y\"), a fragment without name content, or unrelated text — even when a name-like word appears inside.",
				},
			}
		}
		if e.Address != "" {
			qs[fmt.Sprintf("e%d_address_plausible", ei)] = Noul{
				Instructions: fmt.Sprintf(
					"%s.address is the address field of one pharmacy entry read from a photographed Greek duty schedule by OCR; the text may mix Greek and Latin letters, drop accents, be transliterated, or contain recognition noise. Is it a plausible street address in or near Rethymno, Greece?",
					path),
				Criteria: &NoulCriteria{
					True:  "It reads like a street address (a street name, usually with a house number), even with OCR noise.",
					False: "It is not a plausible street address: no street name, a bare number, a phone number, a person's name, a time or date, or unrelated text.",
				},
			}
		}
	}

	resp, err := c.SystemOne(ctx, state, qs)
	if err != nil {
		return nil, nil, err
	}

	verdicts := make([]PlausibilityVerdict, len(entries))
	for ei := range entries {
		v := PlausibilityVerdict{
			Entry:            ei,
			NamePlausible:    -1,
			AddressPlausible: -1,
		}
		if entries[ei].Name != "" {
			answer, ok := resp.Answers[fmt.Sprintf("e%d_name_plausible", ei)]
			if !ok || answer.Type != "noul" {
				return nil, resp, fmt.Errorf("adjudicate: missing noul answer e%d_name_plausible", ei)
			}
			v.NamePlausible = answer.Noul
		}
		if entries[ei].Address != "" {
			answer, ok := resp.Answers[fmt.Sprintf("e%d_address_plausible", ei)]
			if !ok || answer.Type != "noul" {
				return nil, resp, fmt.Errorf("adjudicate: missing noul answer e%d_address_plausible", ei)
			}
			v.AddressPlausible = answer.Noul
		}
		verdicts[ei] = v
	}
	return verdicts, resp, nil
}

// plausibilityState renders the entries for the model.
func plausibilityState(entries []PlausibilityEntry) []map[string]any {
	out := make([]map[string]any, len(entries))
	for ei, e := range entries {
		out[ei] = map[string]any{
			"name":    e.Name,
			"address": e.Address,
			"phone":   e.Phone,
		}
	}
	return out
}
