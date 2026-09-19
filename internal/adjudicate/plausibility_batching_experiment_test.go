package adjudicate

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	jev "github.com/mheers/typesafeai-systemone-jev-go"
)

// TestPlausibilityBatchingExperiment measures how request granularity and
// concurrency change the plausibility judgment's wall time and decisions.
//
// The plausibility experiment (§4.2) runs one request per entry; that was 79
// requests and ~28 s for 79 entries. This experiment asks the operations
// question instead of the modeling question: can several entries share a
// request (chunks), and can independent requests run in parallel, without
// changing the per-field decisions? Sequential per-entry requests are the
// reference; every arm is compared against the reference at the 0.5 and 0.7
// gates, and at entry level against the dataset's ground truth.
//
// It is skipped unless TYPESAFE_EXPERIMENT=1 and TYPESAFE_API_KEY are set.
//
//	TYPESAFE_EXPERIMENT=1 TYPESAFE_EXPERIMENT_OUT=/tmp/ts-plausibility \
//	  go test -run TestPlausibilityBatchingExperiment -v ./internal/adjudicate
func TestPlausibilityBatchingExperiment(t *testing.T) {
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
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	cases, err := loadPlausibilityCases()
	if err != nil {
		t.Fatalf("load dataset: %v", err)
	}
	entries := make([]PlausibilityEntry, len(cases))
	for i, c := range cases {
		entries[i] = c.Entry
	}

	// ---- reference: one request per entry, sequential ----
	refVerdicts, refUsage, refWall := runPlausibilityChunked(t, ctx, client, entries, 1, 1)
	ref := &batchingArm{name: "sequential (1/request)", verdicts: refVerdicts, requests: len(entries), usage: refUsage, wall: refWall}

	var arms []*batchingArm
	arms = append(arms, ref)
	for _, chunk := range []int{2, 4, 8, 16, len(entries)} {
		v, usage, wall := runPlausibilityChunked(t, ctx, client, entries, chunk, 1)
		arms = append(arms, &batchingArm{
			name:     fmt.Sprintf("chunks of %d", chunk),
			verdicts: v,
			requests: (len(entries) + chunk - 1) / chunk,
			usage:    usage,
			wall:     wall,
		})
	}
	for _, workers := range []int{4, 8} {
		v, usage, wall := runPlausibilityChunked(t, ctx, client, entries, 1, workers)
		arms = append(arms, &batchingArm{
			name:     fmt.Sprintf("parallel 1/request, %d workers", workers),
			verdicts: v,
			requests: len(entries),
			usage:    usage,
			wall:     wall,
		})
	}
	for _, workers := range []int{4, 8} {
		v, usage, wall := runPlausibilityChunked(t, ctx, client, entries, 4, workers)
		arms = append(arms, &batchingArm{
			name:     fmt.Sprintf("chunks of 4, %d workers", workers),
			verdicts: v,
			requests: (len(entries) + 3) / 4,
			usage:    usage,
			wall:     wall,
		})
	}

	report := batchingExperimentReport(cases, ref, arms)
	t.Logf("\n%s", report)

	if dir := os.Getenv("TYPESAFE_EXPERIMENT_OUT"); dir != "" {
		if err := writeBatchingExperimentJSON(dir, cases, ref, arms); err != nil {
			t.Logf("write report: %v", err)
		} else {
			t.Logf("wrote %s", dir+"/plausibility-batching-experiment.json")
		}
	}
}

// batchingArm is one measured request strategy.
type batchingArm struct {
	name     string
	verdicts []PlausibilityVerdict
	requests int
	usage    Usage
	wall     time.Duration
}

// runPlausibilityChunked evaluates all entries in chunks of the given size,
// distributing chunks over workers (one worker = sequential). Verdicts are in
// entry order. Each chunk is one System One request.
func runPlausibilityChunked(t *testing.T, ctx context.Context, client *Client, entries []PlausibilityEntry, chunkSize, workers int) ([]PlausibilityVerdict, Usage, time.Duration) {
	t.Helper()
	if chunkSize < 1 {
		chunkSize = 1
	}
	if workers < 1 {
		workers = 1
	}
	start := time.Now()
	verdicts := make([]PlausibilityVerdict, len(entries))
	var usage Usage

	type chunk struct{ from, to int }
	var chunks []chunk
	for from := 0; from < len(entries); from += chunkSize {
		to := from + chunkSize
		if to > len(entries) {
			to = len(entries)
		}
		chunks = append(chunks, chunk{from, to})
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	jobs := make(chan chunk, len(chunks))
	for _, c := range chunks {
		jobs <- c
	}
	close(jobs)
	errs := make(chan error, workers)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range jobs {
				got, resp, err := client.AdjudicatePlausibility(ctx, entries[c.from:c.to])
				if err != nil {
					errs <- fmt.Errorf("entries %d-%d: %w", c.from, c.to, err)
					return
				}
				if len(got) != c.to-c.from {
					errs <- fmt.Errorf("entries %d-%d: %d verdicts, want %d", c.from, c.to, len(got), c.to-c.from)
					return
				}
				mu.Lock()
				copy(verdicts[c.from:c.to], got)
				usage.InputTokens += resp.Usage.InputTokens
				usage.OutputTokens += resp.Usage.OutputTokens
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	close(errs)
	if err := <-errs; err != nil {
		t.Fatal(err)
	}
	return verdicts, usage, time.Since(start)
}

// ---- analysis ----

// batchingAgreement counts per-field decision matches against the reference at
// a gate.
func batchingAgreement(ref, arm *batchingArm, name bool, gate float64) (agree, total int) {
	for i := range ref.verdicts {
		r, a := ref.verdicts[i], arm.verdicts[i]
		rv, av := r.AddressPlausible, a.AddressPlausible
		if name {
			rv, av = r.NamePlausible, a.NamePlausible
		}
		total++
		if (rv >= gate) == (av >= gate) {
			agree++
		}
	}
	return agree, total
}

// batchingTruth counts entry-level ground-truth outcomes at a gate: entries
// with a garbage field that the arm flags, and all-clean entries it flags
// (false alarms).
func batchingTruth(cases []plausibilityCase, arm *batchingArm, gate float64) (garbageFlagged, garbageTotal, cleanFlagged, cleanTotal int) {
	for i, c := range cases {
		v := arm.verdicts[i]
		flag := v.NamePlausible < gate || v.AddressPlausible < gate
		if c.NameQuality == qualityGarbage || c.AddressQuality == qualityGarbage {
			garbageTotal++
			if flag {
				garbageFlagged++
			}
		}
		if c.NameQuality == qualityClean && c.AddressQuality == qualityClean {
			cleanTotal++
			if flag {
				cleanFlagged++
			}
		}
	}
	return garbageFlagged, garbageTotal, cleanFlagged, cleanTotal
}

// batchingDelta reports the mean and max absolute probability difference from
// the reference across all fields.
func batchingDelta(ref, arm *batchingArm) (mean, max float64) {
	n := 0
	for i := range ref.verdicts {
		for _, pair := range [][2]float64{
			{ref.verdicts[i].NamePlausible, arm.verdicts[i].NamePlausible},
			{ref.verdicts[i].AddressPlausible, arm.verdicts[i].AddressPlausible},
		} {
			d := abs(pair[0] - pair[1])
			mean += d
			if d > max {
				max = d
			}
			n++
		}
	}
	if n == 0 {
		return 0, 0
	}
	return mean / float64(n), max
}

func batchingExperimentReport(cases []plausibilityCase, ref *batchingArm, arms []*batchingArm) string {
	var b strings.Builder
	fmt.Fprintf(&b, "granularity experiment: %d entries, model answers per field; reference = sequential one request per entry\n", len(cases))
	fmt.Fprintf(&b, "name=0.5/addr=0.5: decision agreement with the reference at t=0.5; garb=0.7: entries with a garbage field flagged at t=0.7; clean=0.7: all-clean entries flagged (false alarms)\n\n")
	fmt.Fprintf(&b, "%-32s %5s %8s %8s %9s %9s %8s %8s %8s\n",
		"arm", "reqs", "wall", "rel", "name=0.5", "addr=0.5", "garb=0.7", "clean=0.7", "max|d|")
	for _, arm := range arms {
		rel := "-"
		if ref.wall > 0 && arm != ref {
			rel = fmt.Sprintf("%.1fx", float64(ref.wall)/float64(arm.wall))
		}
		na, nt := batchingAgreement(ref, arm, true, 0.5)
		aa, at := batchingAgreement(ref, arm, false, 0.5)
		_, maxDelta := batchingDelta(ref, arm)
		gf, gt, cf, ct := batchingTruth(cases, arm, 0.7)
		fmt.Fprintf(&b, "%-32s %5d %8s %8s %9s %9s %8s %8s %8.2f\n",
			arm.name, arm.requests, arm.wall.Round(time.Millisecond*100), rel,
			fmt.Sprintf("%d/%d", na, nt), fmt.Sprintf("%d/%d", aa, at),
			fmt.Sprintf("%d/%d", gf, gt), fmt.Sprintf("%d/%d", cf, ct), maxDelta)
	}
	// 0.7-gate decision agreement, where the address gate is measured.
	fmt.Fprintf(&b, "\ndecision agreement with sequential at the 0.7 gate:\n")
	for _, arm := range arms {
		if arm == ref {
			continue
		}
		na, nt := batchingAgreement(ref, arm, true, 0.7)
		aa, at := batchingAgreement(ref, arm, false, 0.7)
		meanDelta, maxDelta := batchingDelta(ref, arm)
		// probability differences of fields that flipped at 0.7
		var flipped []string
		for i := range ref.verdicts {
			if (ref.verdicts[i].NamePlausible >= 0.7) != (arm.verdicts[i].NamePlausible >= 0.7) {
				flipped = append(flipped, fmt.Sprintf("%s name %.2f->%.2f", cases[i].ID, ref.verdicts[i].NamePlausible, arm.verdicts[i].NamePlausible))
			}
			if (ref.verdicts[i].AddressPlausible >= 0.7) != (arm.verdicts[i].AddressPlausible >= 0.7) {
				flipped = append(flipped, fmt.Sprintf("%s address %.2f->%.2f", cases[i].ID, ref.verdicts[i].AddressPlausible, arm.verdicts[i].AddressPlausible))
			}
		}
		fmt.Fprintf(&b, "  %-32s name %d/%d, address %d/%d, mean|d| %.3f, max|d| %.2f\n",
			arm.name, na, nt, aa, at, meanDelta, maxDelta)
		if len(flipped) > 0 {
			sort.Strings(flipped)
			for _, f := range flipped {
				fmt.Fprintf(&b, "      flip: %s\n", f)
			}
		}
	}
	fmt.Fprintf(&b, "\ncost and speed:\n")
	for _, arm := range arms {
		fmt.Fprintf(&b, "  %-32s %3d reqs, %6d in / %5d out tokens, %s wall\n",
			arm.name, arm.requests, arm.usage.InputTokens, arm.usage.OutputTokens, arm.wall.Round(time.Millisecond*100))
	}
	return b.String()
}

// batchingCaseExport is the JSON shape of one arm's dataset-aligned verdicts.
type batchingCaseExport struct {
	ID        string                         `json:"id"`
	Entry     PlausibilityEntry              `json:"entry"`
	Reference PlausibilityVerdict            `json:"reference"`
	Verdicts  map[string]PlausibilityVerdict `json:"verdicts"`
}

func writeBatchingExperimentJSON(dir string, cases []plausibilityCase, ref *batchingArm, arms []*batchingArm) error {
	exports := make([]batchingCaseExport, len(cases))
	for i, c := range cases {
		e := batchingCaseExport{
			ID:        c.ID,
			Entry:     c.Entry,
			Reference: ref.verdicts[i],
			Verdicts:  map[string]PlausibilityVerdict{},
		}
		for _, arm := range arms {
			e.Verdicts[arm.name] = arm.verdicts[i]
		}
		exports[i] = e
	}
	armExport := map[string]any{}
	for _, arm := range arms {
		armExport[arm.name] = map[string]any{
			"requests": arm.requests,
			"wall":     arm.wall.String(),
			"usage":    arm.usage,
		}
	}
	out := map[string]any{"cases": exports, "arms": armExport}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "plausibility-batching-experiment.json"), data, 0o644)
}
