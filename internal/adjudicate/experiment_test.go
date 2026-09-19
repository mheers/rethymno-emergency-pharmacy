package adjudicate

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
	jev "github.com/mheers/typesafeai-systemone-jev-go"
)

// TestMergeAdjudicationExperiment runs the live TypeSafe merge experiment.
//
// It is skipped unless TYPESAFE_EXPERIMENT=1 and TYPESAFE_API_KEY are set, so
// `go test ./...` stays hermetic. It builds a dataset from the real reference
// catalog: the four phone numbers that resolve to two catalog entries, plus
// single-entry cases with distractors. Golden records are synthetic name
// variants of real entries (the real golden catalog is not in this
// repository); the ground truth is which entry a variant derives from.
//
// Run it with:
//
//	TYPESAFE_EXPERIMENT=1 go test -run TestMergeAdjudicationExperiment -v ./internal/adjudicate
func TestMergeAdjudicationExperiment(t *testing.T) {
	if os.Getenv("TYPESAFE_EXPERIMENT") == "" {
		t.Skip("set TYPESAFE_EXPERIMENT=1 to run the live TypeSafe experiment")
	}
	model := os.Getenv("TYPESAFE_EXPERIMENT_MODEL")
	if model == "" {
		model = jev.ModelJev1130
	}
	client, err := NewClientFromEnv(jev.WithModel(model))
	if err != nil {
		t.Skipf("live experiment unavailable: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	refs, err := validate.LoadReference(validate.ReferenceJSON)
	if err != nil {
		t.Fatalf("load reference catalog: %v", err)
	}
	cases := buildExperimentCases(t, refs)
	t.Logf("model %s; %d cases", client.Model(), len(cases))

	// ---- arm A: one request per case (Score + Nouls) ----
	armA := make([]armResult, len(cases))
	var usageA Usage
	var wallA time.Duration
	for i, c := range cases {
		start := time.Now()
		verdicts, resp, err := client.AdjudicateMerge(ctx, []MergeGroup{c.group}, MergeConfig{})
		if err != nil {
			t.Fatalf("case %d (%s): %v", i, c.name, err)
		}
		armA[i] = armResult{verdicts: verdicts, resp: resp, elapsed: time.Since(start)}
		usageA.InputTokens += resp.Usage.InputTokens
		usageA.OutputTokens += resp.Usage.OutputTokens
		wallA += armA[i].elapsed
	}

	// ---- arm B: all groups in one request, Score questions only ----
	groups := make([]MergeGroup, len(cases))
	for i, c := range cases {
		groups[i] = c.group
	}
	startB := time.Now()
	verdictsB, respB, err := client.AdjudicateMerge(ctx, groups, MergeConfig{ScoreOnly: true})
	if err != nil {
		t.Fatalf("batched arm: %v", err)
	}
	elapsedB := time.Since(startB)

	report := experimentReport(cases, armA, verdictsB, usageA, respB, wallA, elapsedB)
	t.Logf("\n%s", report)

	if dir := os.Getenv("TYPESAFE_EXPERIMENT_OUT"); dir != "" {
		if err := writeExperimentJSON(dir, cases, armA, verdictsB, usageA, respB, elapsedB); err != nil {
			t.Logf("write report: %v", err)
		} else {
			t.Logf("wrote %s", filepath.Join(dir, "merge-adjudication-experiment.json"))
		}
	}
}

// armResult is one per-case call in arm A.
type armResult struct {
	verdicts []PairVerdict
	resp     *Response
	elapsed  time.Duration
}

// expCase is one phone group with ground truth.
type expCase struct {
	name  string
	kind  string // "shared", "variant-only", "single"
	group MergeGroup
	// truth maps "<refIndex>|<candidateID>" to the expected pair outcome.
	truth map[string]MergeOutcome
}

func pairKey(refIndex int, candidateID string) string {
	return fmt.Sprintf("%d|%s", refIndex, candidateID)
}

func buildExperimentCases(t *testing.T, refs []validate.Reference) []expCase {
	t.Helper()
	var cases []expCase

	shared := []struct {
		phone   string
		curator []string
	}{
		{"2831055212", []string{"ΦΑΡΜΑΚΕΙΟ ΔΗΜΗΤΡΑΚΑΚΗ"}},
		{"2831054706", nil},
		{"2831025123", nil},
		{"2831027264", nil},
	}
	for _, s := range shared {
		groupRefs := refsWithPhone(t, refs, s.phone)
		if len(groupRefs) != 2 {
			t.Fatalf("phone %s: expected 2 reference entries, got %d", s.phone, len(groupRefs))
		}
		c := expCase{
			kind:  "shared",
			name:  fmt.Sprintf("shared %s: %s / %s", s.phone, groupRefs[0].Name, groupRefs[1].Name),
			truth: map[string]MergeOutcome{},
		}
		for ri, ref := range groupRefs {
			c.group.References = append(c.group.References, MergeRef{
				Name:    ref.Name,
				Address: ref.Address,
				Phone:   s.phone,
			})
			names := append([]string{ref.Name}, variantsFor(ref.Name)...)
			for _, name := range names {
				id := fmt.Sprintf("g%s-r%d-%d", s.phone, ri, len(c.group.Candidates))
				c.group.Candidates = append(c.group.Candidates, MergeCandidate{
					ID: id, Name: name, SourceName: name,
					Address: ref.Address, Phone: s.phone,
				})
				c.truth[pairKey(ri, id)] = OutcomeSame
			}
		}
		for _, name := range s.curator {
			id := fmt.Sprintf("g%s-curator", s.phone)
			c.group.Candidates = append(c.group.Candidates, MergeCandidate{
				ID: id, Name: name, SourceName: name,
				Address: groupRefs[0].Address, Phone: s.phone,
			})
			for ri := range groupRefs {
				c.truth[pairKey(ri, id)] = OutcomeCurator
			}
		}
		fillDefaultTruth(c)
		cases = append(cases, c)

		// Variant-only version: the golden source shows no exact name, only
		// short forms. This is the case the 0.55 gate drops today.
		vc := expCase{
			kind:  "variant-only",
			name:  fmt.Sprintf("variant-only %s: %s / %s", s.phone, groupRefs[0].Name, groupRefs[1].Name),
			truth: map[string]MergeOutcome{},
		}
		for ri, ref := range groupRefs {
			vc.group.References = append(vc.group.References, MergeRef{
				Name:    ref.Name,
				Address: ref.Address,
				Phone:   s.phone,
			})
			for _, name := range variantOnlyNames(ref.Name) {
				id := fmt.Sprintf("v%s-r%d-%d", s.phone, ri, len(vc.group.Candidates))
				vc.group.Candidates = append(vc.group.Candidates, MergeCandidate{
					ID: id, Name: name, SourceName: name,
					Address: ref.Address, Phone: s.phone,
				})
				vc.truth[pairKey(ri, id)] = OutcomeSame
			}
		}
		fillDefaultTruth(vc)
		cases = append(cases, vc)
	}

	singles := []struct {
		refName     string
		extraSame   []string
		distractors []string
	}{
		{"Παπατζανή Μαρία", nil, []string{"ΦΑΡΜΑΚΕΙΟ ΑΓΓΕΛΟΣ ΜΑΡΙΝΑΚΗΣ"}},
		{"Δαφνομήλη Γεωργία", nil, []string{"ΟΔΟΝΤΙΑΤΡΕΙΟ ΣΤΑΥΡΟΣ ΠΑΠΑΔΑΚΗΣ"}},
		{"Μαστοράκη - Κεραμιανάκη", []string{"ΚΕΡΑΜΙΑΝΑΚΗ"}, nil},
		{"Δρανδάκη - Λιάσκος", []string{"ΛΙΑΣΚΟΣ"}, nil},
		{"Μαγγανιώτης Σταμάτης", []string{"ΜΑΓΙΑΝΙΩΤΗΣ"}, nil},
		{"Καλογεράκης Ιωάννης", nil, []string{"ΕΣΤΙΑΤΟΡΙΟ ΤΟ ΚΑΣΤΡΟ"}},
	}
	for _, s := range singles {
		ref := refByName(t, refs, s.refName)
		phones := normalize.Phones(ref.Phone)
		if len(phones) == 0 {
			t.Fatalf("%s: no phone", s.refName)
		}
		phone := phones[0]
		c := expCase{kind: "single", name: "single: " + ref.Name, truth: map[string]MergeOutcome{}}
		c.group.References = []MergeRef{{Name: ref.Name, Address: ref.Address, Phone: phone}}
		names := append([]string{ref.Name}, variantsFor(ref.Name)...)
		names = append(names, s.extraSame...)
		for _, name := range names {
			id := fmt.Sprintf("s-%s-%d", phone, len(c.group.Candidates))
			c.group.Candidates = append(c.group.Candidates, MergeCandidate{
				ID: id, Name: name, SourceName: name,
				Address: ref.Address, Phone: phone,
			})
			c.truth[pairKey(0, id)] = OutcomeSame
		}
		for _, name := range s.distractors {
			id := fmt.Sprintf("s-%s-d%d", phone, len(c.group.Candidates))
			c.group.Candidates = append(c.group.Candidates, MergeCandidate{
				ID: id, Name: name, SourceName: name,
				Address: "Άλλη Οδός 1", Phone: phone,
			})
			c.truth[pairKey(0, id)] = OutcomeDifferent
		}
		fillDefaultTruth(c)
		cases = append(cases, c)
	}
	return cases
}

// fillDefaultTruth labels every pair not explicitly assigned as different.
func fillDefaultTruth(c expCase) {
	for ri := range c.group.References {
		for _, cand := range c.group.Candidates {
			k := pairKey(ri, cand.ID)
			if _, ok := c.truth[k]; !ok {
				c.truth[k] = OutcomeDifferent
			}
		}
	}
}

// variantsFor returns plausible cross-source name forms of a catalog name.
func variantsFor(name string) []string {
	seen := map[string]bool{name: true}
	var out []string
	add := func(s string) {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
	}
	add("ΦΑΡΜΑΚΕΙΟ " + name)
	if seg := strings.Split(name, " - ")[0]; seg != name {
		add(seg)
	}
	if f := strings.Fields(name); len(f) > 1 {
		add(f[0])
	}
	if l := normalize.GreekToLatin(name); l != name {
		add(l)
	}
	if typo := typoVariant(name); typo != "" {
		add(typo)
	}
	return out
}

func typoVariant(name string) string {
	words := strings.Fields(name)
	best := -1
	for i, w := range words {
		if len([]rune(w)) >= 5 && (best < 0 || len([]rune(w)) > len([]rune(words[best]))) {
			best = i
		}
	}
	if best < 0 {
		return ""
	}
	r := []rune(words[best])
	mid := len(r) / 2
	r[mid-1], r[mid] = r[mid], r[mid-1]
	words[best] = string(r)
	return strings.Join(words, " ")
}

// variantOnlyNames returns short-form names only (no exact form): the case
// where the golden source never states the reference name in full.
func variantOnlyNames(name string) []string {
	seen := map[string]bool{name: true}
	var out []string
	if seg := strings.Split(name, " - ")[0]; seg != name && !seen[seg] {
		seen[seg] = true
		out = append(out, seg)
	}
	if f := strings.Fields(name); len(f) > 1 && !seen[f[0]] {
		out = append(out, f[0])
	}
	if len(out) == 0 {
		if l := normalize.GreekToLatin(name); l != name {
			out = append(out, l)
		}
	}
	return out
}

func refsWithPhone(t *testing.T, refs []validate.Reference, phone string) []validate.Reference {
	t.Helper()
	var out []validate.Reference
	for _, r := range refs {
		for _, p := range normalize.Phones(r.Phone) {
			if p == phone {
				out = append(out, r)
				break
			}
		}
	}
	return out
}

func refByName(t *testing.T, refs []validate.Reference, name string) validate.Reference {
	t.Helper()
	for _, r := range refs {
		if r.Name == name {
			return r
		}
	}
	t.Fatalf("reference %q not found", name)
	return validate.Reference{}
}

// ---- deterministic baseline (mirrors cmd/merge-golden) ----

func goldenNameScore(refName string, c MergeCandidate) float64 {
	a := normalize.GreekToLatin(refName)
	best := normalize.Similarity(a, normalize.GreekToLatin(c.Name))
	if s2 := normalize.Similarity(a, normalize.GreekToLatin(c.SourceName)); s2 > best {
		best = s2
	}
	return best
}

// baselinePair is the pair-level reading of the current thresholds: the 0.55
// gate for duplicate-phone entries and the 0.4 floor of bestGoldenByName.
func baselinePair(refName string, multiRef bool, c MergeCandidate) MergeOutcome {
	s := goldenNameScore(refName, c)
	gate := 0.55
	if !multiRef {
		gate = 0.4
	}
	switch {
	case s >= gate:
		return OutcomeSame
	case s >= 0.4:
		return OutcomeCurator
	default:
		return OutcomeDifferent
	}
}

// baselineRef mirrors the production decision in cmd/merge-golden.
func baselineRef(c expCase, ri int) (MergeOutcome, string) {
	multi := len(c.group.References) > 1
	ref := c.group.References[ri]
	bestID, bestScore, bestOutcome := "", -1.0, OutcomeDifferent
	for _, cand := range c.group.Candidates {
		s := goldenNameScore(ref.Name, cand)
		if s > bestScore {
			bestID, bestScore, bestOutcome = cand.ID, s, baselinePair(ref.Name, multi, cand)
		}
	}
	if bestScore < 0 {
		return OutcomeDifferent, ""
	}
	return bestOutcome, bestID
}

// ---- analysis ----

type refRow struct {
	kind       string
	caseName   string
	refName    string
	expected   MergeOutcome
	baseline   MergeOutcome
	baselineID string
	tsRaw      MergeOutcome
	tsGated    MergeOutcome
	tsID       string
	tsScore    float64
	tsConf     float64
	tsName     float64
}

func experimentReport(
	cases []expCase,
	armA []armResult,
	verdictsB []PairVerdict,
	usageA Usage,
	respB *Response,
	wallA, wallB time.Duration,
) string {
	var b strings.Builder

	// ---- pair level (arm A) ----
	type counts struct{ same, curator, different int }
	byTruth := map[MergeOutcome]map[string]*counts{}
	bump := func(m map[string]*counts, impl string, out MergeOutcome) {
		if m[impl] == nil {
			m[impl] = &counts{}
		}
		switch out {
		case OutcomeSame:
			m[impl].same++
		case OutcomeCurator:
			m[impl].curator++
		default:
			m[impl].different++
		}
	}
	for i, c := range cases {
		for _, v := range armA[i].verdicts {
			truth := c.truth[pairKey(v.RefIndex, v.Candidate)]
			if byTruth[truth] == nil {
				byTruth[truth] = map[string]*counts{}
			}
			cand := c.group.Candidates[indexOfCandidate(c.group.Candidates, v.Candidate)]
			bump(byTruth[truth], "baseline",
				baselinePair(c.group.References[v.RefIndex].Name, len(c.group.References) > 1, cand))
			bump(byTruth[truth], "typesafe", v.Outcome)
		}
	}
	fmt.Fprintf(&b, "pair-level outcomes by ground truth (%d adjudicated pairs, arm A)\n", countPairs(cases))
	for _, truth := range []MergeOutcome{OutcomeSame, OutcomeCurator, OutcomeDifferent} {
		m := byTruth[truth]
		if m == nil {
			continue
		}
		fmt.Fprintf(&b, "  truth=%-9s baseline same/curator/diff = %d/%d/%d   typesafe same/curator/diff = %d/%d/%d\n",
			truth.String(),
			m["baseline"].same, m["baseline"].curator, m["baseline"].different,
			m["typesafe"].same, m["typesafe"].curator, m["typesafe"].different)
	}

	// ---- batched arm B (Score only) ----
	byCaseB := map[int]map[int]map[string]PairVerdict{}
	for _, v := range verdictsB {
		if byCaseB[v.Group] == nil {
			byCaseB[v.Group] = map[int]map[string]PairVerdict{}
		}
		if byCaseB[v.Group][v.RefIndex] == nil {
			byCaseB[v.Group][v.RefIndex] = map[string]PairVerdict{}
		}
		byCaseB[v.Group][v.RefIndex][v.Candidate] = v
	}
	var compared, agree, disagree int
	for i, r := range armA {
		for _, v := range r.verdicts {
			compared++
			bv, ok := byCaseB[i][v.RefIndex][v.Candidate]
			if !ok {
				continue
			}
			if bv.Outcome == v.Outcome {
				agree++
			} else {
				disagree++
			}
		}
	}
	fmt.Fprintf(&b, "\nbatched arm B (all %d groups in one request, Score only): %d pairs compared, %d agree, %d differ\n",
		len(cases), compared, agree, disagree)

	// ---- ref level (arm A) ----
	var rows []refRow
	for i, c := range cases {
		for ri, ref := range c.group.References {
			baseOut, baseID := baselineRef(c, ri)
			tsRaw, tsGated, tsID, score, conf, name := tsRefOutcome(armA[i].verdicts, ri, 0.5)
			rows = append(rows, refRow{
				kind: c.kind, caseName: c.name, refName: ref.Name, expected: expectedRefOutcome(c, ri),
				baseline: baseOut, baselineID: baseID,
				tsRaw: tsRaw, tsGated: tsGated, tsID: tsID,
				tsScore: score, tsConf: conf, tsName: name,
			})
		}
	}
	fmt.Fprintf(&b, "\nref-level decisions\n")
	fmt.Fprintf(&b, "  %-56s %-7s %-16s %-16s %-16s %s\n",
		"case / reference", "expect", "baseline", "typesafe", "ts gated", "ts score/conf/samename")
	for _, r := range rows {
		detail := "-"
		if r.tsID != "" {
			detail = fmt.Sprintf("%.2f/%.2f/%.2f", r.tsScore, r.tsConf, r.tsName)
		}
		fmt.Fprintf(&b, "  %-56s %-7s %-16s %-16s %-16s %s\n",
			truncate(r.caseName+": "+r.refName, 56),
			r.expected.String(),
			fmt.Sprintf("%s %s", r.baseline, shortID(r.baselineID)),
			fmt.Sprintf("%s %s", r.tsRaw, shortID(r.tsID)),
			r.tsGated.String(),
			detail)
	}

	// ---- headline counts ----
	var baseMissed, tsMissed, baseMissedVariant, tsMissedVariant, baseFalse, tsFalse, tsCurated int
	for _, r := range rows {
		if r.expected == OutcomeSame {
			if r.baseline != OutcomeSame {
				baseMissed++
				if r.kind == "variant-only" {
					baseMissedVariant++
				}
			}
			if r.tsRaw != OutcomeSame {
				tsMissed++
				if r.kind == "variant-only" {
					tsMissedVariant++
				}
			}
			if r.tsRaw == OutcomeCurator || r.tsGated == OutcomeCurator {
				tsCurated++
			}
		} else {
			if r.baseline == OutcomeSame {
				baseFalse++
			}
			if r.tsRaw == OutcomeSame {
				tsFalse++
			}
		}
	}
	fmt.Fprintf(&b, "\nmissed merges (valid pair not merged): baseline %d, typesafe %d (of %d expected merges)\n",
		baseMissed, tsMissed, countExpectedSame(rows))
	fmt.Fprintf(&b, "  variant-only cases: baseline missed %d, typesafe missed %d\n",
		baseMissedVariant, tsMissedVariant)
	fmt.Fprintf(&b, "false merges (pair that should not merge): baseline %d, typesafe %d\n", baseFalse, tsFalse)
	fmt.Fprintf(&b, "expected merges that landed on curator: %d\n", tsCurated)

	// ---- cost ----
	var usageB Usage
	if respB != nil {
		usageB = respB.Usage
	}
	fmt.Fprintf(&b, "\ncost: arm A %d requests, %d in / %d out tokens, %.1fs wall\n",
		len(armA), usageA.InputTokens, usageA.OutputTokens, wallA.Seconds())
	fmt.Fprintf(&b, "      arm B 1 request, %d in / %d out tokens, %.1fs wall\n",
		usageB.InputTokens, usageB.OutputTokens, wallB.Seconds())
	return b.String()
}

func countPairs(cases []expCase) int {
	n := 0
	for _, c := range cases {
		n += len(c.group.References) * len(c.group.Candidates)
	}
	return n
}

func countExpectedSame(rows []refRow) int {
	n := 0
	for _, r := range rows {
		if r.expected == OutcomeSame {
			n++
		}
	}
	return n
}

func indexOfCandidate(cands []MergeCandidate, id string) int {
	for i, c := range cands {
		if c.ID == id {
			return i
		}
	}
	return 0
}

func expectedRefOutcome(c expCase, ri int) MergeOutcome {
	out := OutcomeDifferent
	for _, cand := range c.group.Candidates {
		switch c.truth[pairKey(ri, cand.ID)] {
		case OutcomeSame:
			return OutcomeSame
		case OutcomeCurator:
			out = OutcomeCurator
		}
	}
	return out
}

// tsRefOutcome derives the ref-level decision the caller would take from the
// pair verdicts: a same pair merges; otherwise a curator pair escalates;
// otherwise the reference stays unlinked. gated applies MinConfidence 0.5.
func tsRefOutcome(verdicts []PairVerdict, ri int, minConf float64) (raw, gated MergeOutcome, id string, score, conf, name float64) {
	bestScore, bestRaw, bestGated := -1.0, -1.0, -1.0
	curated := false
	for _, v := range verdicts {
		if v.RefIndex != ri {
			continue
		}
		if v.Score > bestScore {
			id, score, conf, name = v.Candidate, v.Score, v.Confidence, v.SameName
			bestScore = v.Score
		}
		if v.Outcome == OutcomeSame && v.Score > bestRaw {
			bestRaw = v.Score
		}
		if RouteMerge(v.Score, v.Confidence, minConf) == OutcomeSame && v.Score > bestGated {
			bestGated = v.Score
		}
		if v.Outcome == OutcomeCurator {
			curated = true
		}
	}
	raw, gated = OutcomeDifferent, OutcomeDifferent
	if bestRaw >= 0 {
		raw = OutcomeSame
	} else if curated {
		raw = OutcomeCurator
	}
	if bestGated >= 0 {
		gated = OutcomeSame
	} else if curated {
		gated = OutcomeCurator
	}
	if bestScore < 0 {
		id = ""
	}
	return raw, gated, id, score, conf, name
}

func shortID(id string) string {
	if id == "" {
		return "-"
	}
	if len(id) <= 14 {
		return id
	}
	return id[:14] + "…"
}

// caseExport is the JSON shape of a case (the Go case fields are unexported).
type caseExport struct {
	Name  string                  `json:"name"`
	Kind  string                  `json:"kind"`
	Group MergeGroup              `json:"group"`
	Truth map[string]MergeOutcome `json:"truth"`
}

// armExport is the JSON shape of one arm A call.
type armExport struct {
	Verdicts []PairVerdict `json:"verdicts"`
	Response *Response     `json:"response"`
	Elapsed  string        `json:"elapsed"`
}

// writeExperimentJSON stores the raw experiment data for later inspection.
func writeExperimentJSON(
	dir string,
	cases []expCase,
	armA []armResult,
	verdictsB []PairVerdict,
	usageA Usage,
	respB *Response,
	elapsedB time.Duration,
) error {
	armExports := make([]armExport, len(armA))
	for i, r := range armA {
		armExports[i] = armExport{Verdicts: r.verdicts, Response: r.resp, Elapsed: r.elapsed.String()}
	}
	caseExports := make([]caseExport, len(cases))
	for i, c := range cases {
		caseExports[i] = caseExport{Name: c.name, Kind: c.kind, Group: c.group, Truth: c.truth}
	}
	out := map[string]any{
		"cases":   caseExports,
		"arm_a":   armExports,
		"usage_a": usageA,
		"arm_b": map[string]any{
			"verdicts": verdictsB,
			"response": respB,
			"elapsed":  elapsedB.String(),
		},
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "merge-adjudication-experiment.json"), data, 0o644)
}
