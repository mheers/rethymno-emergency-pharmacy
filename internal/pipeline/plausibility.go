package pipeline

import (
	"context"
	"fmt"
	"strings"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/parse"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// PlausibilityJudge answers the warnings-only plausibility questions for
// entries the catalog cannot rescue (TYPESAFE_EVALUATION.md §3D). The runtime
// implementation is adjudicate.CachedPlausibilityJudge, which records every
// decision and evaluates one request per entry; a nil judge keeps the
// deterministic path untouched.
type PlausibilityJudge interface {
	AdjudicatePlausibility(ctx context.Context, entries []adjudicate.PlausibilityEntry) ([]adjudicate.PlausibilityVerdict, error)
}

// VerifyPlausibility asks the judge about every pharmacy whose phone number
// does not match the reference catalog and returns additive warnings for the
// fields whose plausibility falls below their gate. The judgment reports;
// code decides: nothing is rewritten, the confidence formula is untouched, and
// a judge failure produces one warning instead of an error, so the default
// path stays deterministic.
func VerifyPlausibility(ctx context.Context, sched *parse.Schedule, refs []validate.Reference, judge PlausibilityJudge, cfg adjudicate.PlausibilityConfig) []string {
	if judge == nil {
		return nil
	}
	byPhone := catalogByPhone(refs)
	type unmatched struct {
		entry adjudicate.PlausibilityEntry
		label string
		date  string
	}
	var targets []unmatched
	for i := range sched.Days {
		for j := range sched.Days[i].Shifts {
			for k := range sched.Days[i].Shifts[j].Pharmacies {
				ph := &sched.Days[i].Shifts[j].Pharmacies[k]
				if len(byPhone[normalize.DigitsOnly(ph.Phone)]) > 0 {
					continue
				}
				if strings.TrimSpace(ph.Name) == "" && strings.TrimSpace(ph.Address) == "" {
					continue
				}
				targets = append(targets, unmatched{
					entry: adjudicate.PlausibilityEntry{Name: ph.Name, Address: ph.Address, Phone: ph.Phone},
					label: unmatchedLabel(*ph),
					date:  sched.Days[i].Date.Format("02/01/2006"),
				})
			}
		}
	}
	if len(targets) == 0 {
		return nil
	}

	entries := make([]adjudicate.PlausibilityEntry, len(targets))
	for i, t := range targets {
		entries[i] = t.entry
	}
	verdicts, err := judge.AdjudicatePlausibility(ctx, entries)
	if err != nil {
		return []string{fmt.Sprintf("plausibility verification failed for %d unmatched pharmacy entries (%v); the entries are unchanged", len(targets), err)}
	}
	if len(verdicts) != len(targets) {
		return []string{fmt.Sprintf("plausibility verification returned %d verdicts for %d unmatched entries; skipped", len(verdicts), len(targets))}
	}

	var warnings []string
	for i, v := range verdicts {
		for _, field := range v.ImplausibleFields(cfg) {
			probability, gate, value := v.NamePlausible, cfg.MinName, entries[i].Name
			if field == "address" {
				probability, gate, value = v.AddressPlausible, cfg.MinAddress, entries[i].Address
			}
			warnings = append(warnings, fmt.Sprintf(
				"%s is not in the catalog; %s %q reads implausibly (plausibility %.2f below %.2f) on %s",
				targets[i].label, field, value, probability, gate, targets[i].date))
		}
	}
	return warnings
}

// unmatchedLabel identifies an unmatched pharmacy in a warning: by phone when
// the parser read one, otherwise by name.
func unmatchedLabel(ph parse.Pharmacy) string {
	if ph.Phone != "" {
		return "phone " + ph.Phone
	}
	if ph.Name != "" {
		return fmt.Sprintf("entry %q", ph.Name)
	}
	return "entry without phone or name"
}
