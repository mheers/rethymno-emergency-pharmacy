// Command merge-golden merges the Google-enriched golden pharmacy catalog
// (produced by the guide-maps enrichment workflow in the expat-map-guide
// repository) into the embedded reference catalog used for OCR validation and
// schedule enrichment.
//
// Usage:
//
//	go run ./cmd/merge-golden \
//	  -golden /path/to/expat-map-guide/catalog/pharmacies.json \
//	  -reference internal/validate/reference/rethymno_pharmacies.json \
//	  -out internal/validate/reference/rethymno_pharmacies.json
//
// Matching is by normalized phone digits. Reference entries without a phone
// match keep their existing OpenStreetMap coordinates and no Google block.
//
// Ambiguous phone groups (several reference entries or several golden
// candidates on one number) are adjudicated by TypeSafe System One instead of
// the name-similarity thresholds, and the decisions are written to an optional
// report. -judge is on by default and degrades to the thresholds when
// TYPESAFE_API_KEY is not set or the service is unreachable; -judge=false
// forces the deterministic path. -judge-strict refuses to write the output
// while curator decisions or judge fallbacks are unresolved, so ambiguity
// cannot silently become "unlinked". See TYPESAFE_EVALUATION.md and
// internal/adjudicate for the design and the experiment behind it.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/adjudicate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
)

// goldenCatalog mirrors the expat-map-guide enrichment output envelope. Only
// the fields needed for the reference merge are modeled.
type goldenCatalog struct {
	Pharmacies []goldenPharmacy `json:"pharmacies"`
}

// goldenPharmacy is one enriched record from the golden catalog.
type goldenPharmacy struct {
	Source struct {
		Name  string `json:"name"`
		Phone string `json:"phone"`
	} `json:"source"`
	Matched    bool          `json:"matched"`
	PlaceID    string        `json:"placeId,omitempty"`
	Name       string        `json:"name,omitempty"`
	Phone      string        `json:"phone,omitempty"`
	PhoneIntl  string        `json:"phoneInternational,omitempty"`
	WebsiteURL string        `json:"websiteUrl,omitempty"`
	MapsURL    string        `json:"googleMapsUrl,omitempty"`
	Formatted  string        `json:"formattedAddress,omitempty"`
	Latitude   float64       `json:"latitude"`
	Longitude  float64       `json:"longitude"`
	TimeZone   string        `json:"timeZone,omitempty"`
	OffsetMin  int64         `json:"utcOffsetMinutes,omitempty"`
	Rating     float64       `json:"rating,omitempty"`
	ReviewCnt  int64         `json:"userRatingCount,omitempty"`
	Status     string        `json:"businessStatus,omitempty"`
	Types      []string      `json:"types,omitempty"`
	Hours      *goldenHours  `json:"openingHours,omitempty"`
	CurHours   *goldenHours  `json:"currentOpeningHours,omitempty"`
	Photos     []goldenPhoto `json:"photos,omitempty"`
}

type goldenHours struct {
	WeekdayDescriptions []string       `json:"weekdayDescriptions,omitempty"`
	Periods             []goldenPeriod `json:"periods,omitempty"`
	OpenNow             bool           `json:"openNow,omitempty"`
}

type goldenPeriod struct {
	Open  *goldenPoint `json:"open"`
	Close *goldenPoint `json:"close,omitempty"`
}

type goldenPoint struct {
	Day    int64 `json:"day"`
	Hour   int64 `json:"hour"`
	Minute int64 `json:"minute"`
}

type goldenPhoto struct {
	ContentType string `json:"contentType,omitempty"`
	Base64      string `json:"base64,omitempty"`
}

func main() {
	var (
		goldenPath      string
		refPath         string
		outPath         string
		pretty          bool
		judge           bool
		judgeStrict     bool
		judgeModel      string
		judgeMinConf    float64
		judgeReportPath string
	)
	flag.StringVar(&goldenPath, "golden", "", "golden catalog JSON from the enrichment workflow (required)")
	flag.StringVar(&refPath, "reference", "internal/validate/reference/rethymno_pharmacies.json", "reference catalog JSON to merge into")
	flag.StringVar(&outPath, "out", "", "output path (default: overwrite the reference)")
	flag.BoolVar(&pretty, "pretty", true, "indent the output JSON")
	flag.BoolVar(&judge, "judge", true, "adjudicate ambiguous phone groups with TypeSafe System One; falls back to similarity thresholds when the API is unavailable")
	flag.BoolVar(&judgeStrict, "judge-strict", false, "refuse to write the output while curator decisions or judge fallbacks are unresolved")
	flag.StringVar(&judgeModel, "judge-model", "", "TypeSafe model (default jev-latest; the response reports the versioned ID)")
	flag.Float64Var(&judgeMinConf, "judge-min-confidence", 0.5, "minimum confidence for a judged merge")
	flag.StringVar(&judgeReportPath, "judge-report", "", "write the adjudication report as JSON to this path")
	flag.Parse()

	if goldenPath == "" {
		fmt.Fprintln(os.Stderr, "-golden is required")
		os.Exit(2)
	}
	if outPath == "" {
		outPath = refPath
	}

	golden, err := loadGolden(goldenPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "golden: %v\n", err)
		os.Exit(1)
	}
	refs, err := validate.LoadReference(mustRead(refPath))
	if err != nil {
		fmt.Fprintf(os.Stderr, "reference: %v\n", err)
		os.Exit(1)
	}

	if judgeStrict && !judge {
		fmt.Fprintln(os.Stderr, "-judge-strict requires -judge")
		os.Exit(2)
	}
	judgeCtx := context.Background()
	var judgeClient *adjudicate.Client
	var judgeDisabled string
	if judge {
		client, err := adjudicate.NewClientFromEnv()
		if err != nil {
			if judgeStrict {
				fmt.Fprintf(os.Stderr, "-judge-strict requires a working adjudicator: %v\n", err)
				os.Exit(1)
			}
			judgeDisabled = err.Error()
		} else {
			if judgeModel != "" {
				client.Model = judgeModel
			}
			judgeClient = client
			var cancel context.CancelFunc
			judgeCtx, cancel = context.WithTimeout(judgeCtx, 5*time.Minute)
			defer cancel()
		}
	}
	report := &judgeReport{Model: judgeModel}
	if judgeClient != nil {
		report.Model = judgeClient.Model
	}

	byPhone := map[string][]goldenPharmacy{}
	matched := 0
	for _, g := range golden {
		key := normalize.DigitsOnly(firstNonEmpty(g.Phone, g.Source.Phone))
		if key == "" || !g.Matched {
			continue
		}
		byPhone[key] = append(byPhone[key], g)
		matched++
	}

	merged, curated, skipped := 0, 0, 0
	judgedGroups, deterministicGroups, fallbacks := 0, 0, 0
	// Group reference entries by phone so duplicate reference phones (two
	// entries sharing one duty number) can be disambiguated by name. Dual
	// phone fields (e.g. "28310 51113, 28310 55649") register under every
	// individual number.
	byRefPhone := map[string][]int{}
	for i := range refs {
		for _, key := range normalize.Phones(refs[i].Phone) {
			byRefPhone[key] = append(byRefPhone[key], i)
		}
	}
	for phoneKey, indexes := range byRefPhone {
		candidates := byPhone[phoneKey]
		if len(candidates) == 0 {
			continue
		}
		if judgeClient != nil && (len(indexes) > 1 || len(candidates) > 1) {
			decisions, verdicts, resp, err := adjudicateGroup(judgeCtx, judgeClient, refs, indexes, candidates, judgeMinConf)
			if err != nil {
				fmt.Fprintf(os.Stderr, "judge: %s: %v; falling back to similarity thresholds\n", phoneKey, err)
				fallbacks++
				report.Fallbacks = append(report.Fallbacks, phoneKey)
			} else {
				judgedGroups++
				report.Model = resp.Model
				report.Usage.InputTokens += resp.Usage.InputTokens
				report.Usage.OutputTokens += resp.Usage.OutputTokens
				groupReport := judgeGroupReport{
					Phone:      phoneKey,
					References: refNames(refs, indexes),
					Verdicts:   verdicts,
				}
				for _, d := range decisions {
					ref := &refs[d.RefIndex]
					switch d.Outcome {
					case adjudicate.OutcomeSame:
						applyGolden(ref, d.Golden)
						merged++
					case adjudicate.OutcomeCurator:
						curated++
					default:
						skipped++
					}
					groupReport.Decisions = append(groupReport.Decisions, d.report())
				}
				report.Groups = append(report.Groups, groupReport)
				continue
			}
		}

		// Deterministic path (unchanged): best name match, with the 0.55
		// floor when several reference entries share the phone.
		deterministicGroups++
		for _, idx := range indexes {
			ref := &refs[idx]
			g, ok := bestGoldenByName(ref.Name, candidates)
			if !ok {
				skipped++
				continue
			}
			if len(indexes) > 1 {
				// Two reference entries share this phone: only the best
				// name match receives the enrichment; the other keeps its
				// OSM data.
				if goldenNameScore(ref.Name, g) < 0.55 {
					skipped++
					continue
				}
			}
			applyGolden(ref, g)
			merged++
		}
	}

	fmt.Printf("golden records: %d (matched: %d)\n", len(golden), matched)
	fmt.Printf("reference entries: %d (merged: %d, curator queue: %d, unlinked: %d)\n",
		len(refs), merged, curated, skipped)
	if judgeClient != nil {
		fmt.Printf("judge: %d groups adjudicated, %d groups left deterministic, %d fallbacks\n",
			judgedGroups, deterministicGroups, fallbacks)
		fmt.Printf("judge: model %s, %d in / %d out tokens\n",
			report.Model, report.Usage.InputTokens, report.Usage.OutputTokens)
	} else if judgeDisabled != "" {
		fmt.Printf("judge: disabled (%s); similarity thresholds used for every group\n", judgeDisabled)
	} else {
		fmt.Println("judge: disabled by flag; similarity thresholds used for every group")
	}

	if judgeReportPath != "" {
		if !judge && judgeDisabled == "" {
			fmt.Fprintln(os.Stderr, "-judge-report requires an adjudicated run (-judge)")
			os.Exit(2)
		}
		if err := writeJudgeReport(judgeReportPath, report); err != nil {
			fmt.Fprintf(os.Stderr, "write judge report: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("judge report: %s\n", judgeReportPath)
	}

	if err := strictError(judgeStrict, curated, fallbacks); err != nil {
		fmt.Fprintf(os.Stderr, "%v; refusing to write %s\n", err, outPath)
		os.Exit(1)
	}

	data, err := json.MarshalIndent(refs, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	if pretty {
		data = append(data, '\n')
	}
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write %s: %v\n", outPath, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s\n", outPath)
}

// strictError returns a non-nil error when strict mode must block the output
// write: curator decisions are unresolved matches, and fallbacks mean the
// adjudicator never answered for a group.
func strictError(strict bool, curated, fallbacks int) error {
	if !strict || (curated == 0 && fallbacks == 0) {
		return nil
	}
	return fmt.Errorf("-judge-strict: %d curator decision(s) and %d fallback(s) unresolved", curated, fallbacks)
}

// applyGolden copies the enrichment from a golden record onto a reference entry.
func applyGolden(ref *validate.Reference, g goldenPharmacy) {
	ref.Lat = g.Latitude
	ref.Lon = g.Longitude
	ref.Google = toGoogleDetails(g)
}

// judgedDecision is the per-reference outcome of one adjudicated group.
type judgedDecision struct {
	RefIndex int
	Outcome  adjudicate.MergeOutcome
	Golden   goldenPharmacy
	Verdict  adjudicate.PairVerdict
}

func (d judgedDecision) report() judgeDecisionReport {
	return judgeDecisionReport{
		Reference:   d.RefIndex,
		Outcome:     d.Outcome.String(),
		Golden:      d.Golden.Name,
		Score:       d.Verdict.Score,
		Confidence:  d.Verdict.Confidence,
		SameName:    d.Verdict.SameName,
		SameAddress: d.Verdict.SameAddress,
	}
}

// adjudicateGroup sends one phone group to System One and reduces the pair
// verdicts to one decision per reference entry: the best same pair wins,
// otherwise any curator pair escalates, otherwise the entry stays unlinked.
func adjudicateGroup(
	ctx context.Context,
	client *adjudicate.Client,
	refs []validate.Reference,
	indexes []int,
	candidates []goldenPharmacy,
	minConfidence float64,
) ([]judgedDecision, []adjudicate.PairVerdict, *adjudicate.Response, error) {
	group := adjudicate.MergeGroup{}
	for _, idx := range indexes {
		group.References = append(group.References, adjudicate.MergeRef{
			Name:    refs[idx].Name,
			Address: refs[idx].Address,
			Phone:   refs[idx].Phone,
		})
	}
	idFor := make(map[string]goldenPharmacy, len(candidates))
	for ci, g := range candidates {
		id := fmt.Sprintf("cand-%d", ci)
		idFor[id] = g
		group.Candidates = append(group.Candidates, adjudicate.MergeCandidate{
			ID:         id,
			Name:       g.Name,
			SourceName: g.Source.Name,
			Address:    g.Formatted,
			Phone:      firstNonEmpty(g.Phone, g.Source.Phone),
		})
	}

	verdicts, resp, err := client.AdjudicateMerge(ctx, []adjudicate.MergeGroup{group},
		adjudicate.MergeConfig{MinConfidence: minConfidence})
	if err != nil {
		return nil, nil, nil, err
	}

	decisions := make([]judgedDecision, 0, len(indexes))
	for ri := range indexes {
		var best, bestCurator *adjudicate.PairVerdict
		for vi := range verdicts {
			v := &verdicts[vi]
			if v.RefIndex != ri {
				continue
			}
			switch v.Outcome {
			case adjudicate.OutcomeSame:
				if best == nil || v.Score > best.Score ||
					(v.Score == best.Score && v.Candidate < best.Candidate) {
					best = v
				}
			case adjudicate.OutcomeCurator:
				if bestCurator == nil || v.Score > bestCurator.Score ||
					(v.Score == bestCurator.Score && v.Candidate < bestCurator.Candidate) {
					bestCurator = v
				}
			}
		}
		d := judgedDecision{RefIndex: indexes[ri], Outcome: adjudicate.OutcomeDifferent}
		switch {
		case best != nil:
			d.Outcome = adjudicate.OutcomeSame
			d.Golden = idFor[best.Candidate]
			d.Verdict = *best
		case bestCurator != nil:
			d.Outcome = adjudicate.OutcomeCurator
			d.Golden = idFor[bestCurator.Candidate]
			d.Verdict = *bestCurator
		}
		decisions = append(decisions, d)
	}
	return decisions, verdicts, resp, nil
}

func refNames(refs []validate.Reference, indexes []int) []string {
	out := make([]string, 0, len(indexes))
	for _, idx := range indexes {
		out = append(out, refs[idx].Name)
	}
	return out
}

// judgeReport is the JSON written by -judge-report.
type judgeReport struct {
	Model     string             `json:"model"`
	Usage     adjudicate.Usage   `json:"usage"`
	Groups    []judgeGroupReport `json:"groups,omitempty"`
	Fallbacks []string           `json:"fallbacks,omitempty"`
}

type judgeGroupReport struct {
	Phone      string                   `json:"phone"`
	References []string                 `json:"references"`
	Decisions  []judgeDecisionReport    `json:"decisions"`
	Verdicts   []adjudicate.PairVerdict `json:"verdicts"`
}

type judgeDecisionReport struct {
	Reference   int     `json:"reference_index"`
	Outcome     string  `json:"outcome"`
	Golden      string  `json:"golden,omitempty"`
	Score       float64 `json:"score,omitempty"`
	Confidence  float64 `json:"confidence,omitempty"`
	SameName    float64 `json:"same_name,omitempty"`
	SameAddress float64 `json:"same_address,omitempty"`
}

func writeJudgeReport(path string, report *judgeReport) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func loadGolden(path string) ([]goldenPharmacy, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var catalog goldenCatalog
	if err := json.Unmarshal(data, &catalog); err != nil {
		return nil, err
	}
	return catalog.Pharmacies, nil
}

// goldenNameScore measures how well a reference name matches a golden record.
func goldenNameScore(refName string, g goldenPharmacy) float64 {
	score := normalize.Similarity(normalize.GreekToLatin(refName), normalize.GreekToLatin(g.Name))
	if s2 := normalize.Similarity(normalize.GreekToLatin(refName), normalize.GreekToLatin(g.Source.Name)); s2 > score {
		score = s2
	}
	return score
}

// bestGoldenByName picks the golden record whose name best matches a reference
// name. When the reference name is unknown or the best match is too weak, it
// reports false so the caller skips the ambiguous entry.
func bestGoldenByName(refName string, candidates []goldenPharmacy) (goldenPharmacy, bool) {
	if len(candidates) == 1 {
		return candidates[0], true
	}
	best := candidates[0]
	bestScore := -1.0
	for _, g := range candidates {
		score := goldenNameScore(refName, g)
		if score > bestScore {
			best, bestScore = g, score
		}
	}
	if bestScore < 0.4 {
		return goldenPharmacy{}, false
	}
	return best, true
}

func toGoogleDetails(g goldenPharmacy) *validate.GoogleDetails {
	detail := &validate.GoogleDetails{
		PlaceID:             g.PlaceID,
		FormattedAddress:    g.Formatted,
		PhoneInternational:  g.PhoneIntl,
		Website:             g.WebsiteURL,
		GoogleMapsURL:       g.MapsURL,
		Rating:              g.Rating,
		UserRatingCount:     g.ReviewCnt,
		BusinessStatus:      g.Status,
		Types:               g.Types,
		OpeningHours:        toOpenHours(g.Hours),
		CurrentOpeningHours: toOpenHours(g.CurHours),
	}
	for _, p := range g.Photos {
		if p.Base64 == "" {
			continue
		}
		detail.Photos = append(detail.Photos, validate.PhotoDetail{ContentType: p.ContentType, Base64: p.Base64})
	}
	if detail.PlaceID == "" && detail.FormattedAddress == "" && len(detail.Photos) == 0 {
		return nil
	}
	return detail
}

func toOpenHours(h *goldenHours) *validate.OpenHours {
	if h == nil {
		return nil
	}
	out := &validate.OpenHours{
		WeekdayDescriptions: h.WeekdayDescriptions,
		OpenNow:             h.OpenNow,
	}
	for _, p := range h.Periods {
		period := validate.Period{}
		if p.Open != nil {
			period.Open = &validate.PeriodPoint{Day: p.Open.Day, Hour: p.Open.Hour, Minute: p.Open.Minute}
		}
		if p.Close != nil {
			period.Close = &validate.PeriodPoint{Day: p.Close.Day, Hour: p.Close.Hour, Minute: p.Close.Minute}
		}
		out.Periods = append(out.Periods, period)
	}
	return out
}

func mustRead(path string) []byte {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", path, err)
		os.Exit(1)
	}
	return data
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
