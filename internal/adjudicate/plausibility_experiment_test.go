package adjudicate

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// TestPlausibilityAdjudicationExperiment runs the live TypeSafe plausibility
// experiment (TYPESAFE_EVALUATION.md §3D): does a Noul per field separate
// plausible but unmatched OCR readings from garbage, better than the fixed
// validity rules that currently accept both?
//
// It is skipped unless TYPESAFE_EXPERIMENT=1 and TYPESAFE_API_KEY are set, so
// `go test ./...` stays hermetic.
//
// Dataset: testdata/plausibility_corpus.json. The corpus entries are the
// pre-fill parse output of every pharmacy in the three checked-in schedule
// images (dumped with cmd/prefill-dump after the OCR run of 2026-09-19): the
// fields a consumer would see if the phone had been misread and the catalog
// could not fill the entry. Each field is labelled clean / polluted / garbage
// by reviewing the raw OCR lines and the source images. Synthetic entries add
// OCR-noise variants of real catalog entries and the observed garbage classes.
//
// Run it with:
//
//	TYPESAFE_EXPERIMENT=1 go test -run TestPlausibilityAdjudicationExperiment -v ./internal/adjudicate
//
// Set TYPESAFE_EXPERIMENT_OUT to write the raw results as JSON.
func TestPlausibilityAdjudicationExperiment(t *testing.T) {
	if os.Getenv("TYPESAFE_EXPERIMENT") == "" {
		t.Skip("set TYPESAFE_EXPERIMENT=1 to run the live TypeSafe experiment")
	}
	client, err := NewClientFromEnv()
	if err != nil {
		t.Skipf("live experiment unavailable: %v", err)
	}
	if v := os.Getenv("TYPESAFE_EXPERIMENT_MODEL"); v != "" {
		client.Model = v
	} else {
		client.Model = "jev-1.13.0"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	cases, err := loadPlausibilityCases()
	if err != nil {
		t.Fatalf("load dataset: %v", err)
	}
	t.Logf("model %s; %d entries (%d corpus, %d synthetic)",
		client.Model, len(cases), countPlausibilitySource(cases, "corpus"), countPlausibilitySource(cases, "synthetic"))

	// ---- arm A: one request per entry ----
	armA, usageA, wallA := runPlausibilityArm(t, ctx, client, cases)
	// ---- arm A2: repeat, to see how stable the verdicts are ----
	armA2, usageA2, wallA2 := runPlausibilityArm(t, ctx, client, cases)
	// ---- arm B: every entry in one request ----
	entries := make([]PlausibilityEntry, len(cases))
	for i, c := range cases {
		entries[i] = c.Entry
	}
	startB := time.Now()
	verdictsB, respB, errB := client.AdjudicatePlausibility(ctx, entries)
	elapsedB := time.Since(startB)

	report := plausibilityExperimentReport(cases, armA, armA2, verdictsB, errB, usageA, usageA2, respB, wallA, wallA2, elapsedB)
	t.Logf("\n%s", report)

	if dir := os.Getenv("TYPESAFE_EXPERIMENT_OUT"); dir != "" {
		if err := writePlausibilityExperimentJSON(dir, cases, armA, armA2, verdictsB, respB, errB, usageA, usageA2, elapsedB); err != nil {
			t.Logf("write report: %v", err)
		} else {
			t.Logf("wrote %s", filepath.Join(dir, "plausibility-experiment.json"))
		}
	}
}

func runPlausibilityArm(t *testing.T, ctx context.Context, client *Client, cases []plausibilityCase) ([]plausibilityArmResult, Usage, time.Duration) {
	t.Helper()
	results := make([]plausibilityArmResult, len(cases))
	var usage Usage
	var wall time.Duration
	for i, c := range cases {
		start := time.Now()
		verdicts, resp, err := client.AdjudicatePlausibility(ctx, []PlausibilityEntry{c.Entry})
		if err != nil {
			t.Fatalf("entry %d (%s): %v", i, c.ID, err)
		}
		if len(verdicts) != 1 {
			t.Fatalf("entry %d (%s): %d verdicts", i, c.ID, len(verdicts))
		}
		results[i] = plausibilityArmResult{verdict: verdicts[0], resp: resp, elapsed: time.Since(start)}
		usage.InputTokens += resp.Usage.InputTokens
		usage.OutputTokens += resp.Usage.OutputTokens
		wall += results[i].elapsed
	}
	return results, usage, wall
}

// plausibilityArmResult is one per-entry call in an arm.
type plausibilityArmResult struct {
	verdict PlausibilityVerdict
	resp    *Response
	elapsed time.Duration
}

// Field quality labels the ground truth judges the string a consumer sees.
const (
	qualityClean    = "clean"
	qualityPolluted = "polluted"
	qualityGarbage  = "garbage"
)

// plausibilityCase is one dataset entry with per-field ground truth.
type plausibilityCase struct {
	ID             string
	Source         string // "corpus" or "synthetic"
	Provenance     string
	Entry          PlausibilityEntry
	NameQuality    string
	AddressQuality string
}

// plausibilityFixture is the checked-in dataset document.
type plausibilityFixture struct {
	Note    string                    `json:"note"`
	Entries []plausibilityFixtureCase `json:"entries"`
}

type plausibilityFixtureCase struct {
	ID             string `json:"id"`
	Source         string `json:"source"`
	Provenance     string `json:"provenance,omitempty"`
	Name           string `json:"name"`
	Address        string `json:"address"`
	Phone          string `json:"phone,omitempty"`
	NameQuality    string `json:"name_quality"`
	AddressQuality string `json:"address_quality"`
}

// PlausibilityFixturePath is the checked-in experiment dataset.
const PlausibilityFixturePath = "testdata/plausibility_corpus.json"

func loadPlausibilityCases() ([]plausibilityCase, error) {
	data, err := os.ReadFile(PlausibilityFixturePath)
	if err != nil {
		return nil, err
	}
	var f plausibilityFixture
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("parse %s: %w", PlausibilityFixturePath, err)
	}
	if len(f.Entries) == 0 {
		return nil, fmt.Errorf("%s has no entries", PlausibilityFixturePath)
	}
	cases := make([]plausibilityCase, len(f.Entries))
	for i, e := range f.Entries {
		if e.Name == "" || e.Address == "" {
			return nil, fmt.Errorf("entry %d (%s): both name and address are required", i, e.ID)
		}
		if !validPlausibilityQuality(e.NameQuality) || !validPlausibilityQuality(e.AddressQuality) {
			return nil, fmt.Errorf("entry %d (%s): unknown field quality %q/%q", i, e.ID, e.NameQuality, e.AddressQuality)
		}
		cases[i] = plausibilityCase{
			ID:             e.ID,
			Source:         e.Source,
			Provenance:     e.Provenance,
			Entry:          PlausibilityEntry{Name: e.Name, Address: e.Address, Phone: e.Phone},
			NameQuality:    e.NameQuality,
			AddressQuality: e.AddressQuality,
		}
	}
	return cases, nil
}

func validPlausibilityQuality(q string) bool {
	return q == qualityClean || q == qualityPolluted || q == qualityGarbage
}

func countPlausibilitySource(cases []plausibilityCase, source string) int {
	n := 0
	for _, c := range cases {
		if c.Source == source {
			n++
		}
	}
	return n
}

// ---- deterministic baseline ----

// plausibilityBaseline is today's decision for an unmatched entry: the field
// validity rules the validator turns into warnings.
func plausibilityBaseline(c plausibilityCase) (nameOK, addrOK bool) {
	return validate.ValidateNameGreek(c.Entry.Name), validate.ValidateAddress(c.Entry.Address)
}

// ---- analysis ----

// plausibilityRow is one entry's outcome. Each Noul is -1 when unanswered.
type plausibilityRow struct {
	c        plausibilityCase
	nameNoul float64
	addrNoul float64
	baseName bool
	baseAddr bool
}

func plausibilityRows(cases []plausibilityCase, arm []plausibilityArmResult) []plausibilityRow {
	rows := make([]plausibilityRow, len(cases))
	for i, c := range cases {
		nameOK, addrOK := plausibilityBaseline(c)
		rows[i] = plausibilityRow{
			c:        c,
			nameNoul: arm[i].verdict.NamePlausible,
			addrNoul: arm[i].verdict.AddressPlausible,
			baseName: nameOK,
			baseAddr: addrOK,
		}
	}
	return rows
}

// fieldStats counts, per quality class, how many fields an implementation
// accepts.
type fieldStats struct {
	n        int
	accepted int
	signals  []float64
}

// countFieldStats groups fields by their ground-truth quality and asks accept
// for each field's signal.
func countFieldStats(rows []plausibilityRow, name bool, accept func(r plausibilityRow) bool) map[string]*fieldStats {
	m := map[string]*fieldStats{}
	for _, r := range rows {
		quality, signal := r.c.AddressQuality, r.addrNoul
		if name {
			quality, signal = r.c.NameQuality, r.nameNoul
		}
		s := m[quality]
		if s == nil {
			s = &fieldStats{}
			m[quality] = s
		}
		s.n++
		if accept(r) {
			s.accepted++
		}
		s.signals = append(s.signals, signal)
	}
	for _, s := range m {
		sort.Float64s(s.signals)
	}
	return m
}

// plausibilitySignal returns the sorted Noul values of fields with the given
// quality.
func plausibilitySignal(rows []plausibilityRow, name bool, quality string) []float64 {
	var out []float64
	for _, r := range rows {
		q, signal := r.c.AddressQuality, r.addrNoul
		if name {
			q, signal = r.c.NameQuality, r.nameNoul
		}
		if q == quality {
			out = append(out, signal)
		}
	}
	sort.Float64s(out)
	return out
}

func plausibilityExperimentReport(
	cases []plausibilityCase,
	armA, armA2 []plausibilityArmResult,
	verdictsB []PlausibilityVerdict,
	errB error,
	usageA, usageA2 Usage,
	respB *Response,
	wallA, wallA2, wallB time.Duration,
) string {
	rows := plausibilityRows(cases, armA)
	var b strings.Builder

	fmt.Fprintf(&b, "plausibility experiment: %d entries, Noul probability per field (1 = plausible)\n", len(cases))

	// ---- per-field quality: baseline vs system one ----
	for _, field := range []struct {
		name string
		isN  bool
	}{
		{"name", true},
		{"address", false},
	} {
		base := countFieldStats(rows, field.isN, func(r plausibilityRow) bool {
			if field.isN {
				return r.baseName
			}
			return r.baseAddr
		})
		ts := countFieldStats(rows, field.isN, func(r plausibilityRow) bool {
			noul := r.addrNoul
			if field.isN {
				noul = r.nameNoul
			}
			return noul >= 0.5
		})
		fmt.Fprintf(&b, "\n%s fields by ground truth (accepted = signal >= 0.5; a garbage field should be flagged, a clean field accepted)\n", field.name)
		fmt.Fprintf(&b, "  %-9s %4s %14s %14s\n", "quality", "n", "baseline acc", "system one acc")
		for _, q := range []string{qualityClean, qualityPolluted, qualityGarbage} {
			bs, tsr := base[q], ts[q]
			if bs == nil {
				continue
			}
			fmt.Fprintf(&b, "  %-9s %4d %8d (%-5s) %8d (%-5s)\n", q, bs.n,
				bs.accepted, fmt.Sprintf("%.0f%%", 100*float64(bs.accepted)/float64(bs.n)),
				tsr.accepted, fmt.Sprintf("%.0f%%", 100*float64(tsr.accepted)/float64(tsr.n)))
		}
	}

	// ---- signal separation ----
	fmt.Fprintf(&b, "\nsignal separation (Noul values by ground truth):\n")
	for _, field := range []struct {
		name string
		isN  bool
	}{
		{"name", true},
		{"address", false},
	} {
		for _, q := range []string{qualityClean, qualityPolluted, qualityGarbage} {
			fs := plausibilitySignal(rows, field.isN, q)
			if len(fs) == 0 {
				continue
			}
			fmt.Fprintf(&b, "  %-7s %-8s n=%-3d min %.2f, median %.2f, max %.2f\n",
				field.name, q, len(fs), first(fs), median(fs), last(fs))
		}
	}
	cleanName := plausibilitySignal(rows, true, qualityClean)
	garbageName := plausibilitySignal(rows, true, qualityGarbage)
	cleanAddr := plausibilitySignal(rows, false, qualityClean)
	garbageAddr := plausibilitySignal(rows, false, qualityGarbage)
	if len(garbageName) > 0 && len(cleanName) > 0 && last(garbageName) < first(cleanName) {
		fmt.Fprintf(&b, "  names: empty band (%.2f, %.2f]\n", last(garbageName), first(cleanName))
	}
	if len(garbageAddr) > 0 && len(cleanAddr) > 0 && last(garbageAddr) < first(cleanAddr) {
		fmt.Fprintf(&b, "  addresses: empty band (%.2f, %.2f]\n", last(garbageAddr), first(cleanAddr))
	}

	// ---- gate sweep ----
	fmt.Fprintf(&b, "\ngate sweep (flagged when Noul < t; detection = garbage flagged, false alarm = clean flagged):\n")
	fmt.Fprintf(&b, "  %5s %18s %18s %18s %18s\n", "t", "name garbage", "name clean", "address garbage", "address clean")
	for _, t := range []float64{0.3, 0.5, 0.7, 0.8, 0.9} {
		flagged := func(fs []float64) (int, int) {
			n := 0
			for _, v := range fs {
				if v < t {
					n++
				}
			}
			return n, len(fs)
		}
		ng, ngn := flagged(garbageName)
		nc, ncn := flagged(cleanName)
		ag, agn := flagged(garbageAddr)
		ac, acn := flagged(cleanAddr)
		fmt.Fprintf(&b, "  %5.1f %12s %15s %18s %15s\n",
			t, fmt.Sprintf("%d/%d", ng, ngn), fmt.Sprintf("%d/%d", nc, ncn),
			fmt.Sprintf("%d/%d", ag, agn), fmt.Sprintf("%d/%d", ac, acn))
	}

	// ---- entry level ----
	var allClean, anyGarbage int
	var baseCleanFlag, tsCleanFlag, baseGarbFlag, tsGarbFlag int
	for _, r := range rows {
		clean := r.c.NameQuality == qualityClean && r.c.AddressQuality == qualityClean
		garb := r.c.NameQuality == qualityGarbage || r.c.AddressQuality == qualityGarbage
		if clean {
			allClean++
			if !(r.baseName && r.baseAddr) {
				baseCleanFlag++
			}
			if !(r.nameNoul >= 0.5 && r.addrNoul >= 0.5) {
				tsCleanFlag++
			}
		}
		if garb {
			anyGarbage++
			if !(r.baseName && r.baseAddr) {
				baseGarbFlag++
			}
			if !(r.nameNoul >= 0.5 && r.addrNoul >= 0.5) {
				tsGarbFlag++
			}
		}
	}
	fmt.Fprintf(&b, "\nentry level (flagged when any field is below t=0.5):\n")
	fmt.Fprintf(&b, "  all-clean entries (n=%d): baseline flagged %d, system one flagged %d (false alarms)\n",
		allClean, baseCleanFlag, tsCleanFlag)
	fmt.Fprintf(&b, "  entries with a garbage field (n=%d): baseline flagged %d, system one flagged %d (detection)\n",
		anyGarbage, baseGarbFlag, tsGarbFlag)

	// ---- repeatability ----
	nameAgree, addrAgree := 0, 0
	var nameDelta, addrDelta float64
	for i := range rows {
		if d1, d2 := armA[i].verdict.NamePlausible, armA2[i].verdict.NamePlausible; (d1 >= 0.5) == (d2 >= 0.5) {
			nameAgree++
		}
		if d1, d2 := armA[i].verdict.NamePlausible, armA2[i].verdict.NamePlausible; abs(d1-d2) > nameDelta {
			nameDelta = abs(d1 - d2)
		}
		if a1, a2 := armA[i].verdict.AddressPlausible, armA2[i].verdict.AddressPlausible; (a1 >= 0.5) == (a2 >= 0.5) {
			addrAgree++
		}
		if a1, a2 := armA[i].verdict.AddressPlausible, armA2[i].verdict.AddressPlausible; abs(a1-a2) > addrDelta {
			addrDelta = abs(a1 - a2)
		}
	}
	fmt.Fprintf(&b, "\nrepeatability (two identical arm A runs): name t=0.5 decisions agree %d/%d, address %d/%d; max |delta| %.2f / %.2f\n",
		nameAgree, len(rows), addrAgree, len(rows), nameDelta, addrDelta)

	// ---- arm B ----
	fmt.Fprintf(&b, "batched arm (all %d entries in one request): ", len(cases))
	if errB != nil {
		fmt.Fprintf(&b, "failed: %v\n", errB)
	} else {
		nameAgreeB, addrAgreeB := 0, 0
		for i, v := range verdictsB {
			if i < len(rows) && (v.NamePlausible >= 0.5) == (rows[i].nameNoul >= 0.5) {
				nameAgreeB++
			}
			if i < len(rows) && (v.AddressPlausible >= 0.5) == (rows[i].addrNoul >= 0.5) {
				addrAgreeB++
			}
		}
		fmt.Fprintf(&b, "name t=0.5 agrees %d/%d, address %d/%d\n", nameAgreeB, len(verdictsB), addrAgreeB, len(verdictsB))
	}

	// ---- failures ----
	fmt.Fprintf(&b, "\nfields system one got wrong at t=0.5 (raw noul):\n")
	failed := 0
	for _, r := range rows {
		if (r.nameNoul >= 0.5) != (r.c.NameQuality != qualityGarbage) {
			failed++
			fmt.Fprintf(&b, "  %-24s name    %-40s truth=%-8s noul %.2f  %s\n",
				r.c.ID, truncate(fmt.Sprintf("%q", r.c.Entry.Name), 40), r.c.NameQuality, r.nameNoul, truncate(r.c.Provenance, 50))
		}
		if (r.addrNoul >= 0.5) != (r.c.AddressQuality != qualityGarbage) {
			failed++
			fmt.Fprintf(&b, "  %-24s address %-40s truth=%-8s noul %.2f  %s\n",
				r.c.ID, truncate(fmt.Sprintf("%q", r.c.Entry.Address), 40), r.c.AddressQuality, r.addrNoul, truncate(r.c.Provenance, 50))
		}
	}
	if failed == 0 {
		fmt.Fprintf(&b, "  none\n")
	}

	// ---- cost ----
	fmt.Fprintf(&b, "\ncost: arm A %d requests, %d in / %d out tokens, %.1fs wall\n",
		len(rows), usageA.InputTokens, usageA.OutputTokens, wallA.Seconds())
	fmt.Fprintf(&b, "      arm A2 (repeat) %d requests, %d in / %d out tokens, %.1fs wall\n",
		len(rows), usageA2.InputTokens, usageA2.OutputTokens, wallA2.Seconds())
	if respB != nil {
		fmt.Fprintf(&b, "      arm B 1 request, %d in / %d out tokens, %.1fs wall\n",
			respB.Usage.InputTokens, respB.Usage.OutputTokens, wallB.Seconds())
	}
	return b.String()
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

// plausibilityCaseExport is the JSON shape of a case with the truth.
type plausibilityCaseExport struct {
	ID             string            `json:"id"`
	Source         string            `json:"source"`
	Provenance     string            `json:"provenance,omitempty"`
	Entry          PlausibilityEntry `json:"entry"`
	NameQuality    string            `json:"name_quality"`
	AddressQuality string            `json:"address_quality"`
}

// plausibilityArmExport is the JSON shape of one call in an arm.
type plausibilityArmExport struct {
	Verdict PlausibilityVerdict `json:"verdict"`
	Elapsed string              `json:"elapsed"`
}

func writePlausibilityExperimentJSON(
	dir string,
	cases []plausibilityCase,
	armA, armA2 []plausibilityArmResult,
	verdictsB []PlausibilityVerdict,
	respB *Response,
	errB error,
	usageA, usageA2 Usage,
	elapsedB time.Duration,
) error {
	caseExports := make([]plausibilityCaseExport, len(cases))
	for i, c := range cases {
		caseExports[i] = plausibilityCaseExport{
			ID: c.ID, Source: c.Source, Provenance: c.Provenance,
			Entry: c.Entry, NameQuality: c.NameQuality, AddressQuality: c.AddressQuality,
		}
	}
	armExports := make([]plausibilityArmExport, len(armA))
	arm2Exports := make([]plausibilityArmExport, len(armA2))
	for i := range armA {
		armExports[i] = plausibilityArmExport{Verdict: armA[i].verdict, Elapsed: armA[i].elapsed.String()}
		arm2Exports[i] = plausibilityArmExport{Verdict: armA2[i].verdict, Elapsed: armA2[i].elapsed.String()}
	}
	armB := map[string]any{"verdicts": verdictsB, "response": respB, "elapsed": elapsedB.String()}
	if errB != nil {
		armB["error"] = errB.Error()
	}
	out := map[string]any{
		"cases":   caseExports,
		"arm_a":   armExports,
		"arm_a2":  arm2Exports,
		"usage_a": usageA,
		"arm_b":   armB,
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "plausibility-experiment.json"), data, 0o644)
}
