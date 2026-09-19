package pipeline

import (
	"context"
	"fmt"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/parse"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// IdentityJudge selects which catalog entry an OCR entry refers to when its
// phone number matches several. The runtime implementation is
// adjudicate.CachedJudge, which records every decision; a nil judge keeps the
// deterministic similarity pick (TYPESAFE_EVALUATION.md §3A).
type IdentityJudge interface {
	AdjudicateIdentity(ctx context.Context, groups []adjudicate.IdentityGroup, cfg adjudicate.IdentityConfig) ([]adjudicate.IdentityVerdict, error)
}

// CatalogFill is the outcome of a judged catalog fill. Picks records the
// catalog entry used for every pharmacy whose phone number matched several
// entries while judging was enabled, keyed by pharmacyKey; pass it to
// ValidateResultWithPicks so validation checks the same entry.
type CatalogFill struct {
	Filled   int
	Picks    map[string]validate.Reference
	Warnings []string
}

// pharmacyKey identifies one pharmacy within a schedule for the pick map: its
// normalized phone and the name as it stands after the fill.
func pharmacyKey(phone, name string) string {
	return normalize.DigitsOnly(phone) + "\x00" + name
}

// FillFromCatalogJudged is FillFromCatalog with an optional identity judge.
// Only pharmacies whose phone number resolves to more than one catalog entry
// are judged, one request each; an accepted verdict replaces the similarity
// pick, and any other outcome (none, below the gate, service error) keeps the
// similarity pick and adds a warning. Warnings are deterministic and additive:
// the JSON schema does not change.
func FillFromCatalogJudged(ctx context.Context, sched *parse.Schedule, refs []validate.Reference, judge IdentityJudge, cfg adjudicate.IdentityConfig) CatalogFill {
	byPhone := catalogByPhone(refs)
	fill := CatalogFill{Picks: map[string]validate.Reference{}}
	for i := range sched.Days {
		for j := range sched.Days[i].Shifts {
			for k := range sched.Days[i].Shifts[j].Pharmacies {
				ph := &sched.Days[i].Shifts[j].Pharmacies[k]
				candidates := byPhone[normalize.DigitsOnly(ph.Phone)]
				if len(candidates) == 0 {
					continue
				}
				ref := chooseCatalogReference(ph, candidates)
				judged := judge != nil && len(candidates) > 1
				if judged {
					index, warning := judgeCatalogIdentity(ctx, judge, cfg, *ph, candidates)
					if warning != "" {
						fill.Warnings = append(fill.Warnings, warning)
					}
					if index >= 0 {
						if picked := candidates[index]; picked != ref {
							fill.Warnings = append(fill.Warnings, fmt.Sprintf(
								"phone %s matches several catalog entries; adjudicator selected %s instead of the similarity pick %s",
								ph.Phone, picked.Name, ref.Name))
						}
						ref = candidates[index]
					}
				}
				if applyCatalogReference(ph, ref) {
					fill.Filled++
				}
				if judged {
					fill.Picks[pharmacyKey(ph.Phone, ph.Name)] = ref
				}
			}
		}
	}
	return fill
}

// judgeCatalogIdentity evaluates one ambiguous pharmacy. It returns the
// candidate index to use, or -1 to keep the similarity pick, plus an optional
// warning explaining a fallback.
func judgeCatalogIdentity(ctx context.Context, judge IdentityJudge, cfg adjudicate.IdentityConfig, ph parse.Pharmacy, candidates []validate.Reference) (int, string) {
	ids := make([]string, len(candidates))
	group := adjudicate.IdentityGroup{
		Reading: adjudicate.IdentityReading{
			Name:    ph.Name,
			Address: ph.Address,
			Phone:   ph.Phone,
		},
		Candidates: make([]adjudicate.IdentityCandidate, len(candidates)),
	}
	for i, c := range candidates {
		ids[i] = fmt.Sprintf("c%d", i)
		group.Candidates[i] = adjudicate.IdentityCandidate{ID: ids[i], Name: c.Name, Address: c.Address}
	}

	verdicts, err := judge.AdjudicateIdentity(ctx, []adjudicate.IdentityGroup{group}, cfg)
	if err != nil {
		return -1, fmt.Sprintf("phone %s matches several catalog entries and identity adjudication failed (%v); kept the similarity pick", ph.Phone, err)
	}
	if len(verdicts) != 1 {
		return -1, fmt.Sprintf("phone %s matches several catalog entries and identity adjudication returned %d verdicts; kept the similarity pick", ph.Phone, len(verdicts))
	}
	v := verdicts[0]
	if v.Choice == adjudicate.IdentityNone {
		return -1, fmt.Sprintf("phone %s matches several catalog entries; adjudicator found no confident candidate (confidence %.2f, same-pharmacy %.2f); kept the similarity pick", ph.Phone, v.Confidence, v.SamePharmacy)
	}
	index := -1
	for i, id := range ids {
		if id == v.Choice {
			index = i
			break
		}
	}
	if index < 0 {
		return -1, fmt.Sprintf("phone %s matches several catalog entries and identity adjudication chose an unknown candidate %q; kept the similarity pick", ph.Phone, v.Choice)
	}
	if !adjudicate.AcceptIdentity(v.Choice, v.Confidence, v.SamePharmacy, cfg) {
		return -1, fmt.Sprintf("phone %s matches several catalog entries; adjudicator confidence %.2f / same-pharmacy %.2f below the gate; kept the similarity pick", ph.Phone, v.Confidence, v.SamePharmacy)
	}
	return index, ""
}
