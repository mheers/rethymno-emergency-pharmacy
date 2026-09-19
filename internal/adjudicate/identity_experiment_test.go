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

	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// TestIdentityAdjudicationExperiment runs the live TypeSafe catalog-identity
// experiment (TYPESAFE_EVALUATION.md §3A).
//
// It is skipped unless TYPESAFE_EXPERIMENT=1 and TYPESAFE_API_KEY are set, so
// `go test ./...` stays hermetic. The dataset is built from the real
// reference catalog: the four phone numbers that resolve to two catalog
// entries, each producing OCR-style readings of both entries (exact name,
// short form, Greeklish transliteration, recognizer look-alike script,
// character noise) plus readings that must not resolve to a candidate
// (address only, street-derived name, a different pharmacy's name, an
// unrelated business). Ground truth is which entry a reading derives from.
//
// The readings are simulated: the real schedule corpus does not contain these
// four numbers, so the recognizer's observed failure modes (mixed
// Greek/Latin look-alikes as in "ΔHMHTPAKAKH", swapped characters) are
// applied deterministically to the catalog names.
//
// Run it with:
//
//	TYPESAFE_EXPERIMENT=1 go test -run TestIdentityAdjudicationExperiment -v ./internal/adjudicate
//
// Set TYPESAFE_EXPERIMENT_OUT to write the raw results as JSON.
func TestIdentityAdjudicationExperiment(t *testing.T) {
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
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	refs, err := validate.LoadReference(validate.ReferenceJSON)
	if err != nil {
		t.Fatalf("load reference catalog: %v", err)
	}
	cases := buildIdentityCases(t, refs)
	t.Logf("model %s; %d readings", client.Model, len(cases))

	// ---- arm A: one request per reading ----
	results := make([]identityArmResult, len(cases))
	var usageA Usage
	var wallA time.Duration
	for i, c := range cases {
		start := time.Now()
		verdicts, resp, err := client.AdjudicateIdentity(ctx, []IdentityGroup{c.group}, IdentityConfig{})
		if err != nil {
			t.Fatalf("case %d (%s): %v", i, c.name, err)
		}
		if len(verdicts) != 1 {
			t.Fatalf("case %d (%s): %d verdicts", i, c.name, len(verdicts))
		}
		results[i] = identityArmResult{verdict: verdicts[0], resp: resp, elapsed: time.Since(start)}
		usageA.InputTokens += resp.Usage.InputTokens
		usageA.OutputTokens += resp.Usage.OutputTokens
		wallA += results[i].elapsed
	}

	// ---- arm B: every reading in one request (granularity check) ----
	groups := make([]IdentityGroup, len(cases))
	for i, c := range cases {
		groups[i] = c.group
	}
	startB := time.Now()
	verdictsB, respB, errB := client.AdjudicateIdentity(ctx, groups, IdentityConfig{})
	elapsedB := time.Since(startB)
	if errB != nil {
		t.Logf("batched arm failed: %v", errB)
	}

	report := identityExperimentReport(cases, results, verdictsB, errB, usageA, respB, wallA, elapsedB)
	t.Logf("\n%s", report)

	if dir := os.Getenv("TYPESAFE_EXPERIMENT_OUT"); dir != "" {
		if err := writeIdentityExperimentJSON(dir, cases, results, verdictsB, respB, errB, usageA, elapsedB); err != nil {
			t.Logf("write report: %v", err)
		} else {
			t.Logf("wrote %s", filepath.Join(dir, "identity-adjudication-experiment.json"))
		}
	}
}

// identityArmResult is one per-case call in arm A.
type identityArmResult struct {
	verdict IdentityVerdict
	resp    *Response
	elapsed time.Duration
}

// identityCase is one OCR-style reading with ground truth. expect is the
// candidate ID the reading derives from, or "" when no candidate should be
// accepted.
type identityCase struct {
	kind   string
	name   string
	group  IdentityGroup
	expect string
}

// identityReadingVariant is one generated reading of a catalog entry.
type identityReadingVariant struct {
	kind string
	name string
}

func buildIdentityCases(t *testing.T, refs []validate.Reference) []identityCase {
	t.Helper()
	shared := []string{"2831055212", "2831054706", "2831025123", "2831027264"}
	var cases []identityCase
	for _, phone := range shared {
		groupRefs := refsWithPhone(t, refs, phone)
		if len(groupRefs) != 2 {
			t.Fatalf("phone %s: expected 2 reference entries, got %d", phone, len(groupRefs))
		}
		candidates := make([]IdentityCandidate, len(groupRefs))
		for i, r := range groupRefs {
			candidates[i] = IdentityCandidate{
				ID:      fmt.Sprintf("cand-%s-%d", phone, i),
				Name:    r.Name,
				Address: r.Address,
			}
		}
		for ri, ref := range groupRefs {
			for _, rv := range identityReadings(ref) {
				cases = append(cases, identityCase{
					kind: rv.kind,
					name: fmt.Sprintf("%s %s: %s", phone, rv.kind, rv.name),
					group: IdentityGroup{
						Reading: IdentityReading{
							Name:    rv.name,
							Address: ref.Address,
							Phone:   phone,
						},
						Candidates: candidates,
					},
					expect: candidates[ri].ID,
				})
			}
		}
		// Readings that must not resolve to a candidate: the name is absent,
		// derived from the shared street, or belongs to another business.
		noPick := []identityReadingVariant{
			{kind: "address-only", name: ""},
			{kind: "street-name", name: "ΦΑΡΜΑΚΕΙΟ " + strings.ToUpper(normalize.NormalizeGreek(streetToken(groupRefs[0].Address)))},
			{kind: "foreign-name", name: foreignRefName(refs, phone)},
			{kind: "unrelated", name: "ΚΑΦΕΝΕΙΟ Ο ΚΗΠΟΣ"},
		}
		for _, np := range noPick {
			cases = append(cases, identityCase{
				kind: np.kind,
				name: fmt.Sprintf("%s %s: %s", phone, np.kind, np.name),
				group: IdentityGroup{
					Reading: IdentityReading{
						Name:    np.name,
						Address: groupRefs[0].Address,
						Phone:   phone,
					},
					Candidates: candidates,
				},
				expect: "",
			})
		}
	}
	return cases
}

// identityReadings returns the OCR-style readings of one catalog entry. Every
// kind is a failure mode observed in the schedule corpus or a plausible
// source variant; duplicates are dropped.
func identityReadings(ref validate.Reference) []identityReadingVariant {
	var out []identityReadingVariant
	seen := map[string]bool{}
	add := func(kind, name string) {
		name = strings.TrimSpace(name)
		if name == "" || seen[name] {
			return
		}
		seen[name] = true
		out = append(out, identityReadingVariant{kind: kind, name: name})
	}
	add("exact", ref.Name)
	if seg := strings.Split(ref.Name, " - ")[0]; seg != ref.Name {
		add("short", seg)
	}
	if f := strings.Fields(ref.Name); len(f) > 1 {
		add("first-word", f[0])
	}
	if latin := normalize.GreekToLatin(ref.Name); latin != ref.Name {
		add("latin", latin)
	}
	add("lookalike", identityLookAlikeVariant(ref.Name))
	add("typo", typoVariant(ref.Name))
	return out
}

// identityLookAlike maps Greek capitals to the Latin look-alikes PP-OCRv6
// emits, keeping the shapes it usually recognises (Δ, Φ). It reproduces
// corpus outputs such as "KAAYΦATAKH" and "ΔHMHTPAKAKH".
var identityLookAlike = map[rune]rune{
	'Α': 'A', 'Β': 'B', 'Γ': 'G', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Θ': 'O',
	'Ι': 'I', 'Κ': 'K', 'Λ': 'A', 'Μ': 'M', 'Ν': 'N', 'Ξ': 'X', 'Ο': 'O',
	'Π': 'P', 'Ρ': 'P', 'Σ': 'S', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X', 'Ψ': 'Y',
	'Ω': 'W',
}

func identityLookAlikeVariant(name string) string {
	up := strings.ToUpper(normalize.NormalizeGreek(name))
	var b strings.Builder
	for _, r := range up {
		if lat, ok := identityLookAlike[r]; ok {
			b.WriteRune(lat)
			continue
		}
		b.WriteRune(r)
	}
	v := b.String()
	if v == name {
		return ""
	}
	return v
}

// streetToken returns the longest word of an address: the street name a
// name-less reading might be misread as ("ΦΑΡΜΑΚΕΙΟ ΔΗΜΗΤΡΑΚΑΚΗ").
func streetToken(addr string) string {
	best := ""
	for _, f := range strings.Fields(addr) {
		f = strings.Trim(f, ".,-·")
		if len([]rune(f)) > len([]rune(best)) {
			best = f
		}
	}
	return best
}

// foreignRefName returns the name of the first catalog entry that does not
// share the phone number: a real pharmacy name that must not resolve against
// this group's candidates.
func foreignRefName(refs []validate.Reference, phone string) string {
	for _, r := range refs {
		if !refHasPhone(r, phone) {
			return r.Name
		}
	}
	return ""
}

func refHasPhone(r validate.Reference, phone string) bool {
	for _, p := range normalize.Phones(r.Phone) {
		if p == phone {
			return true
		}
	}
	return false
}

// ---- deterministic baseline ----

// identityBaselineID is the pick today's code makes: ChoosePhoneReference,
// the same similarity-based selection the validator and FillFromCatalog use.
func identityBaselineID(c identityCase) string {
	refs := make([]validate.Reference, len(c.group.Candidates))
	for i, cand := range c.group.Candidates {
		refs[i] = validate.Reference{Name: cand.Name, Address: cand.Address, Phone: c.group.Reading.Phone}
	}
	pick := validate.ChoosePhoneReference(c.group.Reading.Name, c.group.Reading.Address, refs)
	for _, cand := range c.group.Candidates {
		if cand.Name == pick.Name {
			return cand.ID
		}
	}
	return ""
}

// ---- analysis ----

// identityRow is one reading's outcome. signal is min(confidence, noul of the
// chosen candidate), the quantity a runtime gate would compare; it is -1 when
// the model declined (or the Noul was missing).
type identityRow struct {
	kind       string
	caseName   string
	expect     string
	baselineID string
	tsChoice   string
	tsConf     float64
	tsNoul     float64
	signal     float64
	correct    bool
	wrongPick  bool
	none       bool
}

func identityRows(cases []identityCase, results []identityArmResult) []identityRow {
	rows := make([]identityRow, len(cases))
	for i, c := range cases {
		v := results[i].verdict
		r := identityRow{
			kind:       c.kind,
			caseName:   c.name,
			expect:     c.expect,
			baselineID: identityBaselineID(c),
			tsChoice:   v.Choice,
			tsConf:     v.Confidence,
			tsNoul:     v.SamePharmacy,
			signal:     -1,
		}
		switch {
		case v.Choice == IdentityNone:
			r.none = true
		case c.expect == "":
			r.wrongPick = true
		case v.Choice == c.expect:
			r.correct = true
		default:
			r.wrongPick = true
		}
		if v.Choice != IdentityNone {
			r.signal = v.Confidence
			if v.SamePharmacy >= 0 && v.SamePharmacy < r.signal {
				r.signal = v.SamePharmacy
			}
		}
		rows[i] = r
	}
	return rows
}

type identityKindStats struct {
	n       int
	baseOK  int
	tsOK    int
	tsNone  int
	tsWrong int
}

func identityStatsByKind(rows []identityRow) map[string]*identityKindStats {
	m := map[string]*identityKindStats{}
	for _, r := range rows {
		s := m[r.kind]
		if s == nil {
			s = &identityKindStats{}
			m[r.kind] = s
		}
		s.n++
		if r.expect != "" {
			if r.baselineID == r.expect {
				s.baseOK++
			}
			if r.correct {
				s.tsOK++
			}
		}
		if r.none {
			s.tsNone++
		}
		if r.wrongPick {
			s.tsWrong++
		}
	}
	return m
}

// identitySignal collects the acceptance signals of correct or wrong picks.
func identitySignal(rows []identityRow, wantCorrect bool) []float64 {
	var out []float64
	for _, r := range rows {
		if r.signal < 0 {
			continue
		}
		if wantCorrect && r.correct {
			out = append(out, r.signal)
		}
		if !wantCorrect && r.wrongPick {
			out = append(out, r.signal)
		}
	}
	sort.Float64s(out)
	return out
}

func median(fs []float64) float64 {
	if len(fs) == 0 {
		return 0
	}
	return fs[len(fs)/2]
}

func identityExperimentReport(
	cases []identityCase,
	results []identityArmResult,
	verdictsB []IdentityVerdict,
	errB error,
	usageA Usage,
	respB *Response,
	wallA, wallB time.Duration,
) string {
	rows := identityRows(cases, results)
	var b strings.Builder

	decisive, noPick := 0, 0
	for _, c := range cases {
		if c.expect == "" {
			noPick++
		} else {
			decisive++
		}
	}
	fmt.Fprintf(&b, "identity experiment: %d readings (%d decisive, %d no-pick)\n", len(cases), decisive, noPick)

	// ---- per-kind table ----
	stats := identityStatsByKind(rows)
	kinds := make([]string, 0, len(stats))
	for k := range stats {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	fmt.Fprintf(&b, "\n%-14s %3s %9s %6s %6s %6s\n", "kind", "n", "base ok", "ts ok", "ts none", "ts wrong")
	for _, k := range kinds {
		s := stats[k]
		fmt.Fprintf(&b, "%-14s %3d %9d %6d %6d %6d\n", k, s.n, s.baseOK, s.tsOK, s.tsNone, s.tsWrong)
	}

	// ---- headline counts ----
	var baseOK, tsOK, tsNoneDecisive, tsWrongDecisive, tsNoneNoPick, tsPickedNoPick, baselineForced int
	for _, r := range rows {
		switch {
		case r.expect == "":
			if r.none {
				tsNoneNoPick++
			} else {
				tsPickedNoPick++
			}
			if r.baselineID != "" {
				baselineForced++
			}
		default:
			if r.baselineID == r.expect {
				baseOK++
			}
			switch {
			case r.correct:
				tsOK++
			case r.none:
				tsNoneDecisive++
			default:
				tsWrongDecisive++
			}
		}
	}
	fmt.Fprintf(&b, "\ndecisive readings: baseline picked correctly %d/%d, System One %d/%d (%d declined, %d wrong candidate)\n",
		baseOK, decisive, tsOK, decisive, tsNoneDecisive, tsWrongDecisive)
	fmt.Fprintf(&b, "no-pick readings: System One declined %d/%d, picked a candidate %d/%d (baseline forced a pick %d/%d)\n",
		tsNoneNoPick, noPick, tsPickedNoPick, noPick, baselineForced, noPick)

	// ---- signal separation ----
	right := identitySignal(rows, true)
	wrong := identitySignal(rows, false)
	fmt.Fprintf(&b, "\nacceptance signal min(confidence, same-pharmacy noul) for the chosen candidate:\n")
	fmt.Fprintf(&b, "  correct picks (n=%d): min %.2f, median %.2f, max %.2f\n",
		len(right), first(right), median(right), last(right))
	fmt.Fprintf(&b, "  wrong picks (n=%d):   min %.2f, median %.2f, max %.2f\n",
		len(wrong), first(wrong), median(wrong), last(wrong))
	if len(wrong) > 0 && len(right) > 0 && last(wrong) < first(right) {
		fmt.Fprintf(&b, "  empty band: a gate in (%.2f, %.2f] accepts every correct pick and no observed wrong pick\n",
			last(wrong), first(right))
	} else if len(wrong) == 0 {
		fmt.Fprintf(&b, "  no wrong picks observed\n")
	} else {
		fmt.Fprintf(&b, "  no empty band: wrong picks reach %.2f, correct picks start at %.2f\n", last(wrong), first(right))
	}

	// ---- gate sweep ----
	fmt.Fprintf(&b, "\ngate sweep (accept when signal >= t; below t falls back to the deterministic pick):\n")
	for _, t := range []float64{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} {
		acceptedRight, acceptedWrong := 0, 0
		for _, r := range rows {
			if r.signal < t {
				continue
			}
			if r.correct {
				acceptedRight++
			} else {
				acceptedWrong++
			}
		}
		fmt.Fprintf(&b, "  t=%.1f: accepted correct %d/%d, accepted wrong %d\n", t, acceptedRight, decisive, acceptedWrong)
	}

	// ---- arm B ----
	fmt.Fprintf(&b, "\nbatched arm (all %d readings in one request): ", len(cases))
	if errB != nil {
		fmt.Fprintf(&b, "failed: %v\n", errB)
	} else {
		agree := 0
		for _, v := range verdictsB {
			if v.Group < len(results) && results[v.Group].verdict.Choice == v.Choice {
				agree++
			}
		}
		fmt.Fprintf(&b, "%d/%d picks agree with per-reading requests\n", agree, len(verdictsB))
	}

	// ---- failed readings ----
	fmt.Fprintf(&b, "\nreadings System One did not get right (raw pick):\n")
	failed := 0
	for _, r := range rows {
		if r.correct || (r.none && r.expect == "") {
			continue
		}
		failed++
		fmt.Fprintf(&b, "  %-56s expect %-18s got %-18s conf %.2f noul %.2f\n",
			truncate(r.caseName, 56), shortID(r.expect), shortID(r.tsChoice), r.tsConf, r.tsNoul)
	}
	if failed == 0 {
		fmt.Fprintf(&b, "  none\n")
	}

	// ---- cost ----
	fmt.Fprintf(&b, "\ncost: arm A %d requests, %d in / %d out tokens, %.1fs wall\n",
		len(results), usageA.InputTokens, usageA.OutputTokens, wallA.Seconds())
	if respB != nil {
		fmt.Fprintf(&b, "      arm B 1 request, %d in / %d out tokens, %.1fs wall\n",
			respB.Usage.InputTokens, respB.Usage.OutputTokens, wallB.Seconds())
	}
	return b.String()
}

func first(fs []float64) float64 {
	if len(fs) == 0 {
		return 0
	}
	return fs[0]
}

func last(fs []float64) float64 {
	if len(fs) == 0 {
		return 0
	}
	return fs[len(fs)-1]
}

// identityCaseExport is the JSON shape of a case (the Go fields are unexported).
type identityCaseExport struct {
	Name   string        `json:"name"`
	Kind   string        `json:"kind"`
	Expect string        `json:"expect"`
	Group  IdentityGroup `json:"group"`
}

// identityArmExport is the JSON shape of one arm A call.
type identityArmExport struct {
	Verdict  IdentityVerdict `json:"verdict"`
	Response *Response       `json:"response"`
	Elapsed  string          `json:"elapsed"`
}

func writeIdentityExperimentJSON(
	dir string,
	cases []identityCase,
	results []identityArmResult,
	verdictsB []IdentityVerdict,
	respB *Response,
	errB error,
	usageA Usage,
	elapsedB time.Duration,
) error {
	caseExports := make([]identityCaseExport, len(cases))
	for i, c := range cases {
		caseExports[i] = identityCaseExport{Name: c.name, Kind: c.kind, Expect: c.expect, Group: c.group}
	}
	armExports := make([]identityArmExport, len(results))
	for i, r := range results {
		armExports[i] = identityArmExport{Verdict: r.verdict, Response: r.resp, Elapsed: r.elapsed.String()}
	}
	armB := map[string]any{
		"verdicts": verdictsB,
		"response": respB,
		"elapsed":  elapsedB.String(),
	}
	if errB != nil {
		armB["error"] = errB.Error()
	}
	out := map[string]any{
		"cases":   caseExports,
		"arm_a":   armExports,
		"usage_a": usageA,
		"arm_b":   armB,
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "identity-adjudication-experiment.json"), data, 0o644)
}
