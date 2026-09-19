package pipeline_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/parse"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/pipeline"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// fakeIdentityJudge answers every group through answer, recording the calls.
type fakeIdentityJudge struct {
	calls  int
	last   adjudicate.IdentityConfig
	answer func(g adjudicate.IdentityGroup) adjudicate.IdentityVerdict
	err    error
}

func (f *fakeIdentityJudge) AdjudicateIdentity(_ context.Context, groups []adjudicate.IdentityGroup, cfg adjudicate.IdentityConfig) ([]adjudicate.IdentityVerdict, error) {
	f.calls++
	f.last = cfg
	if f.err != nil {
		return nil, f.err
	}
	out := make([]adjudicate.IdentityVerdict, len(groups))
	for i, g := range groups {
		out[i] = f.answer(g)
		out[i].Group = i
	}
	return out, nil
}

// sharedPhoneRefs are the two real catalog entries that share 2831055212.
func sharedPhoneRefs() []validate.Reference {
	return []validate.Reference{
		{Name: "Βαρούχα - Αναγνωστάκης", Address: "Δημητρακάκη 21", Phone: "2831055212",
			NameLat: "Varoucha - Anagnostakis", AddressLat: "Dimitrakaki 21", Lat: 35.1, Lon: 24.1},
		{Name: "Καλαφατάκη Ελένη - Γεωργία", Address: "Δημητρακάκη 21", Phone: "2831055212",
			NameLat: "Kalafataki Eleni - Georgia", AddressLat: "Dimitrakaki 21", Lat: 35.2, Lon: 24.2},
	}
}

func onePharmacySchedule(name, address, phone string) *parse.Schedule {
	return &parse.Schedule{Days: []parse.DaySchedule{{
		Date: time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC), Day: "ΔΕΥΤΕΡΑ",
		Shifts: []parse.Shift{{From: "08:00", To: "21:00", Pharmacies: []parse.Pharmacy{
			{Name: name, Address: address, Phone: phone},
		}}},
	}}}
}

func verdict(choice string, confidence, same float64) func(adjudicate.IdentityGroup) adjudicate.IdentityVerdict {
	return func(adjudicate.IdentityGroup) adjudicate.IdentityVerdict {
		return adjudicate.IdentityVerdict{Choice: choice, Confidence: confidence, SamePharmacy: same}
	}
}

func TestFillFromCatalogJudgedOverride(t *testing.T) {
	// The similarity pick is Βαρούχα: both candidates share the address, so
	// the address similarity (1.0) dominates and the name tie-break decides.
	// The judge instead reads the name and selects Καλαφατάκη (c1).
	sched := onePharmacySchedule("ΚΑΛΑΦΑΤΑΚΗ ΕΛΕΝΗ", "ΔΗΜΗΤΡΑΚΑΚΗ 21", "2831055212")
	judge := &fakeIdentityJudge{answer: verdict("c1", 0.95, 0.9)}
	cfg := adjudicate.IdentityConfig{MinConfidence: 0.8, MinSamePharmacy: 0.8}

	fill := pipeline.FillFromCatalogJudged(context.Background(), sched, sharedPhoneRefs(), judge, cfg)
	if judge.calls != 1 {
		t.Fatalf("judge calls = %d, want 1", judge.calls)
	}
	if judge.last != cfg {
		t.Errorf("judge config = %+v, want %+v", judge.last, cfg)
	}
	if fill.Filled != 1 {
		t.Fatalf("filled = %d, want 1", fill.Filled)
	}
	ph := sched.Days[0].Shifts[0].Pharmacies[0]
	if ph.Name != "ΚΑΛΑΦΑΤΑΚΗ ΕΛΕΝΗ - ΓΕΩΡΓΙΑ" {
		t.Errorf("name = %q, want the judged candidate", ph.Name)
	}
	if ph.NameLat != "Kalafataki Eleni - Georgia" || ph.Lat != 35.2 || ph.Lon != 24.2 {
		t.Errorf("enrichment = %+v, want the judged candidate's", ph)
	}
	if len(fill.Warnings) != 1 || !strings.Contains(fill.Warnings[0], "instead of the similarity pick") {
		t.Errorf("warnings = %v, want an override warning", fill.Warnings)
	}

	// The fill's picks must reach validation: the catalog match agrees with
	// the judged name instead of re-running the similarity pick.
	if len(fill.Picks) != 1 {
		t.Fatalf("picks = %v, want one entry", fill.Picks)
	}
	for _, picked := range fill.Picks {
		if picked.Name != "Καλαφατάκη Ελένη - Γεωργία" {
			t.Fatalf("picked = %q, want the judged reference", picked.Name)
		}
	}
	vals, warnings := pipeline.ValidateResultWithPicks(sched, sharedPhoneRefs(), fill.Picks)
	if len(vals) != 1 || vals[0].Validation.CatalogMatch == nil ||
		vals[0].Validation.CatalogMatch.Name != "Καλαφατάκη Ελένη - Γεωργία" {
		t.Errorf("validation catalog match = %+v, want the judged entry", vals)
	}
	if len(warnings) != 0 {
		t.Errorf("validation warnings = %v, want none against the judged entry", warnings)
	}
	// Without the picks the validator re-picks by address and disagrees.
	_, staleWarnings := pipeline.ValidateResult(sched, sharedPhoneRefs())
	if len(staleWarnings) == 0 || !strings.Contains(staleWarnings[0], "name mismatch") {
		t.Errorf("expected the un-picked validation to flag a mismatch, got %v", staleWarnings)
	}
}

func TestFillFromCatalogJudgedAcceptsAgreement(t *testing.T) {
	// The judge selects c0, which is also the similarity pick: no warning.
	sched := onePharmacySchedule("ΚΑΛΑΦΑΤΑΚΗ ΕΛΕΝΗ", "ΔΗΜΗΤΡΑΚΑΚΗ 21", "2831055212")
	judge := &fakeIdentityJudge{answer: verdict("c0", 0.95, 0.9)}

	fill := pipeline.FillFromCatalogJudged(context.Background(), sched, sharedPhoneRefs(), judge,
		adjudicate.IdentityConfig{MinConfidence: 0.8, MinSamePharmacy: 0.8})
	if fill.Filled != 1 || len(fill.Warnings) != 0 {
		t.Fatalf("filled = %d, warnings = %v; want a silent, correct fill", fill.Filled, fill.Warnings)
	}
	if len(fill.Picks) != 1 {
		t.Errorf("picks = %v, want the judged pick recorded", fill.Picks)
	}
	if got := sched.Days[0].Shifts[0].Pharmacies[0].Name; got != "ΒΑΡΟΥΧΑ - ΑΝΑΓΝΩΣΤΑΚΗΣ" {
		t.Errorf("name = %q", got)
	}
}

func TestFillFromCatalogJudgedFallbacks(t *testing.T) {
	cases := []struct {
		name    string
		judge   *fakeIdentityJudge
		wantSub string
	}{
		{
			name:    "none",
			judge:   &fakeIdentityJudge{answer: verdict(adjudicate.IdentityNone, 0.95, -1)},
			wantSub: "no confident candidate",
		},
		{
			name:    "low confidence",
			judge:   &fakeIdentityJudge{answer: verdict("c0", 0.5, 0.9)},
			wantSub: "below the gate",
		},
		{
			name:    "low same-pharmacy noul",
			judge:   &fakeIdentityJudge{answer: verdict("c0", 0.95, 0.4)},
			wantSub: "below the gate",
		},
		{
			name:    "service error",
			judge:   &fakeIdentityJudge{err: errors.New("connection refused")},
			wantSub: "adjudication failed",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sched := onePharmacySchedule("ΚΑΛΑΦΑΤΑΚΗ ΕΛΕΝΗ", "ΔΗΜΗΤΡΑΚΑΚΗ 21", "2831055212")
			fill := pipeline.FillFromCatalogJudged(context.Background(), sched, sharedPhoneRefs(), c.judge,
				adjudicate.IdentityConfig{MinConfidence: 0.8, MinSamePharmacy: 0.8})
			if got := sched.Days[0].Shifts[0].Pharmacies[0].Name; got != "ΒΑΡΟΥΧΑ - ΑΝΑΓΝΩΣΤΑΚΗΣ" {
				t.Errorf("fallback name = %q, want the similarity pick", got)
			}
			if len(fill.Warnings) != 1 || !strings.Contains(fill.Warnings[0], c.wantSub) {
				t.Errorf("warnings = %v, want one containing %q", fill.Warnings, c.wantSub)
			}
			// the fallback pick is still recorded, so validation checks it
			if len(fill.Picks) != 1 {
				t.Errorf("picks = %v, want the fallback pick recorded", fill.Picks)
			}
		})
	}
}

func TestFillFromCatalogJudgedSkipsSingleCandidates(t *testing.T) {
	refs := []validate.Reference{
		{Name: "Καλογεράκης Ιωάννης", Address: "Μοάτσου 8", Phone: "2831022187", NameLat: "Kalogerakis Ioannis"},
	}
	sched := onePharmacySchedule("", "ΜΟΑΤΣΟΥ 8", "2831022187")
	judge := &fakeIdentityJudge{answer: verdict("c0", 0.99, 0.99)}

	fill := pipeline.FillFromCatalogJudged(context.Background(), sched, refs, judge, adjudicate.IdentityConfig{})
	if judge.calls != 0 {
		t.Errorf("judge called %d times for a single candidate", judge.calls)
	}
	if fill.Filled != 1 || len(fill.Warnings) != 0 {
		t.Fatalf("filled = %d, warnings = %v", fill.Filled, fill.Warnings)
	}
	if len(fill.Picks) != 0 {
		t.Errorf("picks = %v, want none for a single candidate", fill.Picks)
	}
	if got := sched.Days[0].Shifts[0].Pharmacies[0].Name; got != "ΚΑΛΟΓΕΡΑΚΗΣ ΙΩΑΝΝΗΣ" {
		t.Errorf("name = %q", got)
	}
}

func TestFillFromCatalogJudgedNilJudgeMatchesDeterministic(t *testing.T) {
	deterministic := onePharmacySchedule("ΚΑΛΑΦΑΤΑΚΗ ΕΛΕΝΗ", "ΔΗΜΗΤΡΑΚΑΚΗ 21", "2831055212")
	judged := onePharmacySchedule("ΚΑΛΑΦΑΤΑΚΗ ΕΛΕΝΗ", "ΔΗΜΗΤΡΑΚΑΚΗ 21", "2831055212")

	if n := pipeline.FillFromCatalog(deterministic, sharedPhoneRefs()); n != 1 {
		t.Fatalf("deterministic filled = %d", n)
	}
	fill := pipeline.FillFromCatalogJudged(context.Background(), judged, sharedPhoneRefs(), nil, adjudicate.IdentityConfig{})
	if fill.Filled != 1 || len(fill.Warnings) != 0 || len(fill.Picks) != 0 {
		t.Fatalf("nil judge: fill = %+v", fill)
	}
	a := deterministic.Days[0].Shifts[0].Pharmacies[0]
	b := judged.Days[0].Shifts[0].Pharmacies[0]
	if a.Name != b.Name || a.NameLat != b.NameLat || a.Lat != b.Lat || a.Lon != b.Lon {
		t.Errorf("nil judge changed behavior:\n got %+v\nwant %+v", b, a)
	}
}
