package pipeline

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/parse"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// fakePlausibilityJudge records the entries it was asked about and returns
// either fixed verdicts or fully plausible defaults.
type fakePlausibilityJudge struct {
	calls    int
	entries  []adjudicate.PlausibilityEntry
	verdicts []adjudicate.PlausibilityVerdict
	err      error
}

func (f *fakePlausibilityJudge) AdjudicatePlausibility(_ context.Context, entries []adjudicate.PlausibilityEntry) ([]adjudicate.PlausibilityVerdict, error) {
	f.calls++
	f.entries = entries
	if f.err != nil {
		return nil, f.err
	}
	if f.verdicts != nil {
		return f.verdicts, nil
	}
	out := make([]adjudicate.PlausibilityVerdict, len(entries))
	for i, e := range entries {
		out[i] = adjudicate.PlausibilityVerdict{Entry: i, NamePlausible: -1, AddressPlausible: -1}
		if e.Name != "" {
			out[i].NamePlausible = 0.9
		}
		if e.Address != "" {
			out[i].AddressPlausible = 0.9
		}
	}
	return out, nil
}

// scheduleWith builds a one-day schedule with the given name/address/phone
// triples.
func scheduleWith(entries ...[3]string) *parse.Schedule {
	sched := &parse.Schedule{SourceURL: "test", City: "Ρέθυμνο"}
	day := parse.DaySchedule{
		Date:   time.Date(2026, 9, 19, 0, 0, 0, 0, time.UTC),
		Day:    "ΣΑΒΒΑΤΟ",
		Shifts: []parse.Shift{{From: "08:00", To: "21:00"}},
	}
	for _, e := range entries {
		day.Shifts[0].Pharmacies = append(day.Shifts[0].Pharmacies, parse.Pharmacy{Name: e[0], Address: e[1], Phone: e[2]})
	}
	sched.Days = []parse.DaySchedule{day}
	return sched
}

// knownRef is a catalog entry whose phone matches the schedule entries used
// as "rescued" controls.
var knownRef = validate.Reference{Name: "Γνωστή Φαρμακείο", Address: "Οδός 1", Phone: "2831000000"}

func TestVerifyPlausibilityWarnsBelowGate(t *testing.T) {
	judge := &fakePlausibilityJudge{verdicts: []adjudicate.PlausibilityVerdict{
		{Entry: 0, NamePlausible: 0.4, AddressPlausible: 0.8},
		{Entry: 1, NamePlausible: 0.9, AddressPlausible: 0.6},
	}}
	sched := scheduleWith(
		[3]string{"ΔHMHTPAKAKH", "THΛ.2831023347", "2831055212"},
		[3]string{"KEPAMIANAKH", "A.AГNΩΣTОYΣTPATIΩTH31", "2831054706"},
	)
	cfg := adjudicate.PlausibilityConfig{MinName: 0.5, MinAddress: 0.7}
	warnings := VerifyPlausibility(context.Background(), sched, []validate.Reference{knownRef}, judge, cfg)

	if judge.calls != 1 {
		t.Fatalf("judge calls = %d, want 1", judge.calls)
	}
	if len(judge.entries) != 2 {
		t.Fatalf("judged %d entries, want 2", len(judge.entries))
	}
	if judge.entries[0].Phone != "2831055212" || judge.entries[1].Phone != "2831054706" {
		t.Errorf("judged entries out of order: %+v", judge.entries)
	}
	if len(warnings) != 2 {
		t.Fatalf("warnings = %v, want 2", warnings)
	}
	for _, want := range []string{
		`phone 2831055212 is not in the catalog; name "ΔHMHTPAKAKH" reads implausibly (plausibility 0.40 below 0.50) on 19/09/2026`,
		`phone 2831054706 is not in the catalog; address "A.AГNΩΣTОYΣTPATIΩTH31" reads implausibly (plausibility 0.60 below 0.70) on 19/09/2026`,
	} {
		found := false
		for _, w := range warnings {
			if w == want {
				found = true
			}
		}
		if !found {
			t.Errorf("missing warning %q in %v", want, warnings)
		}
	}
}

func TestVerifyPlausibilitySkipsMatchedEntries(t *testing.T) {
	judge := &fakePlausibilityJudge{}
	sched := scheduleWith(
		[3]string{"ΓΝΩΣΤΗ ΦΑΡΜΑΚΕΙΟ", "ΟΔΟΣ 1", "2831000000"},
		[3]string{"ΑΓΝΩΣΤΗ", "ΟΔΟΣ 2", "2831000001"},
	)
	warnings := VerifyPlausibility(context.Background(), sched, []validate.Reference{knownRef}, judge, adjudicate.PlausibilityConfig{MinName: 0.5, MinAddress: 0.7})
	if judge.calls != 1 {
		t.Fatalf("judge calls = %d, want 1", judge.calls)
	}
	if len(judge.entries) != 1 || judge.entries[0].Phone != "2831000001" {
		t.Fatalf("judged entries = %+v, want only the unmatched phone", judge.entries)
	}
	if len(warnings) != 0 {
		t.Errorf("warnings = %v, want none for plausible fields", warnings)
	}
}

func TestVerifyPlausibilityNilJudge(t *testing.T) {
	sched := scheduleWith([3]string{"ΑΓΝΩΣΤΗ", "ΟΔΟΣ 2", "2831000001"})
	if warnings := VerifyPlausibility(context.Background(), sched, []validate.Reference{knownRef}, nil, adjudicate.PlausibilityConfig{MinName: 0.5, MinAddress: 0.7}); warnings != nil {
		t.Errorf("nil judge produced warnings: %v", warnings)
	}
}

func TestVerifyPlausibilityNoUnmatchedEntries(t *testing.T) {
	judge := &fakePlausibilityJudge{}
	sched := scheduleWith(
		[3]string{"ΓΝΩΣΤΗ ΦΑΡΜΑΚΕΙΟ", "ΟΔΟΣ 1", "2831000000"},
		[3]string{"", "", "2831000002"}, // nothing to judge
	)
	if warnings := VerifyPlausibility(context.Background(), sched, []validate.Reference{knownRef}, judge, adjudicate.PlausibilityConfig{MinName: 0.5, MinAddress: 0.7}); warnings != nil {
		t.Errorf("warnings = %v, want none", warnings)
	}
	if judge.calls != 0 {
		t.Errorf("judge calls = %d, want 0", judge.calls)
	}
}

func TestVerifyPlausibilityJudgeErrorIsAWarning(t *testing.T) {
	judge := &fakePlausibilityJudge{err: errors.New("api down")}
	sched := scheduleWith([3]string{"ΑΓΝΩΣΤΗ", "ΟΔΟΣ 2", "2831000001"})
	warnings := VerifyPlausibility(context.Background(), sched, []validate.Reference{knownRef}, judge, adjudicate.PlausibilityConfig{MinName: 0.5, MinAddress: 0.7})
	if len(warnings) != 1 {
		t.Fatalf("warnings = %v, want one", warnings)
	}
	if !strings.Contains(warnings[0], "plausibility verification failed") || !strings.Contains(warnings[0], "api down") {
		t.Errorf("warning = %q", warnings[0])
	}
	if strings.Contains(warnings[0], "reads implausibly") {
		t.Errorf("failure warning claims a judgment: %q", warnings[0])
	}
}

func TestVerifyPlausibilityVerdictMismatchIsAWarning(t *testing.T) {
	judge := &fakePlausibilityJudge{verdicts: []adjudicate.PlausibilityVerdict{}}
	sched := scheduleWith([3]string{"ΑΓΝΩΣΤΗ", "ΟΔΟΣ 2", "2831000001"})
	warnings := VerifyPlausibility(context.Background(), sched, []validate.Reference{knownRef}, judge, adjudicate.PlausibilityConfig{MinName: 0.5, MinAddress: 0.7})
	if len(warnings) != 1 || !strings.Contains(warnings[0], "returned 0 verdicts") {
		t.Errorf("warnings = %v, want one mismatch warning", warnings)
	}
}

func TestVerifyPlausibilityEmptyFieldsAreNotWarned(t *testing.T) {
	judge := &fakePlausibilityJudge{verdicts: []adjudicate.PlausibilityVerdict{
		{Entry: 0, NamePlausible: 0.1, AddressPlausible: -1},
	}}
	sched := scheduleWith([3]string{"ΑΓΝΩΣΤΗ", "", "2831000001"})
	warnings := VerifyPlausibility(context.Background(), sched, []validate.Reference{knownRef}, judge, adjudicate.PlausibilityConfig{MinName: 0.5, MinAddress: 0.7})
	if len(warnings) != 1 || !strings.Contains(warnings[0], `name "ΑΓΝΩΣΤΗ"`) {
		t.Fatalf("warnings = %v, want the name warning only (empty address never flags)", warnings)
	}
}
