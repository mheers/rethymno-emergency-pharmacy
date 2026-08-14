// Package pipeline orchestrates the full OCR ingestion pipeline:
// fetch → extract → vision → ocr → layout → parse → validate.
package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"log"
	"os"
	"path/filepath"
	"sort"
	"time"

	"gocv.io/x/gocv"

	"github.com/mheers/rethymno-emergency-pharmacy/internal/layout"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/normalize"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/ocr"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/parse"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/validate"
	"github.com/mheers/rethymno-emergency-pharmacy/internal/vision"
	"strings"
)

// StageTimings records per-stage elapsed times for the benchmark output.
type StageTimings struct {
	Decode      time.Duration
	Perspective time.Duration
	Deskew      time.Duration
	Grid        time.Duration
	Columns     time.Duration
	Preprocess  time.Duration
	OCRA        time.Duration
	Layout      time.Duration
	Parse       time.Duration
	Validate    time.Duration
	Total       time.Duration
}

// Result is the full pipeline output.
type Result struct {
	Schedule    parse.Schedule       `json:"schedule"`
	RawOCR      []ocr.OCRLine        `json:"-"`
	RawOCRText  string               `json:"raw_ocr,omitempty"`
	ImageSHA256 string               `json:"image_sha256"`
	Timings     StageTimings         `json:"timings"`
	Validations []PharmacyValidation `json:"validations,omitempty"`
	Warnings    []string             `json:"warnings,omitempty"`
	DebugDir    string               `json:"debug_dir,omitempty"`
}

// PharmacyValidation couples a parsed pharmacy with its validation outcome.
type PharmacyValidation struct {
	Phone      string              `json:"phone"`
	Name       string              `json:"name"`
	Validation validate.Validation `json:"validation"`
}

// Options configures a pipeline run.
type Options struct {
	SourceURL string
	City      string
	DebugDir  string // when set, debug images are written here
}

// OCRBackend is the OCR stage interface. The production implementation is
// the worker subprocess (ocr.Remote); tests may use the in-process engine
// (which must not be combined with gocv in one process).
type OCRBackend interface {
	Recognize(img *ocr.Image) ([]ocr.OCRLine, error)
	Close() error
}

// Pipeline runs the stages in order.
type Pipeline struct {
	Vision *vision.Processor
	OCR    OCRBackend
	Log    *log.Logger
}

// New creates a Pipeline.
func New(v *vision.Processor, e OCRBackend, l *log.Logger) *Pipeline {
	if l == nil {
		l = log.New(os.Stderr, "[pipe] ", log.LstdFlags)
	}
	return &Pipeline{Vision: v, OCR: e, Log: l}
}

// Run executes the full pipeline on an already-decoded image.
func (p *Pipeline) Run(img gocv.Mat, opts Options) (*Result, error) {
	res := &Result{}
	t0 := time.Now()

	// ---- vision ----
	proc, err := p.Vision.Process(img)
	if err != nil {
		return nil, fmt.Errorf("pipeline: vision: %w", err)
	}
	res.Timings.Perspective = time.Since(t0)
	t1 := time.Now()

	// ---- OCR (det + rec on detected boxes) ----
	lines, err := p.OCRColumnLines(proc)
	if err != nil {
		CloseProcessed(proc)
		return nil, fmt.Errorf("pipeline: ocr: %w", err)
	}
	res.Timings.OCRA = time.Since(t1)
	t1 = time.Now()

	// ---- layout ----
	doc := layout.Reconstruct(columnRects(proc.Columns), toLayoutLines(lines))
	res.Timings.Layout = time.Since(t1)
	t1 = time.Now()

	// ---- parse ----
	sched := parse.ParseDocument(&doc, opts.SourceURL, opts.City)
	res.Timings.Parse = time.Since(t1)
	t1 = time.Now()

	res.RawOCRText = rawOCRTxt(lines)
	res.Timings.Validate = time.Since(t1)

	// ---- debug output ----
	if opts.DebugDir != "" {
		res.DebugDir = opts.DebugDir
		writeDebug(proc, lines, &doc, opts.DebugDir, p.Log)
	}
	CloseProcessed(proc)

	res.Schedule = sched
	res.Timings.Total = time.Since(t0)
	return res, nil
}

// RunBytes executes the pipeline on raw encoded image bytes.
func (p *Pipeline) RunBytes(ctx context.Context, data []byte, opts Options) (*Result, error) {
	_ = ctx
	t0 := time.Now()
	img, err := p.Vision.Decode(data)
	if err != nil {
		return nil, err
	}
	defer img.Close()
	res, err := p.Run(img, opts)
	if err != nil {
		return nil, err
	}
	res.Timings.Decode = time.Since(t0) - res.Timings.Total
	// image hash
	h := sha256.Sum256(data)
	res.ImageSHA256 = hex.EncodeToString(h[:])
	return res, nil
}

// OCRColumnLines runs detection and recognition per grid column at (near)
// full resolution on the grid-cleaned column crops, then offsets the boxes
// back into the image coordinate space. Columns are small enough that no
// det downscaling is needed, keeping small phone digits legible.
func (p *Pipeline) OCRColumnLines(proc *vision.ProcessedSchedule) ([]ocr.OCRLine, error) {
	var all []ocr.OCRLine
	columnOCR, fixedLayout := p.OCR.(interface {
		RecognizeColumn(*ocr.Image) ([]ocr.OCRLine, error)
	})
	for _, col := range proc.Columns {
		if col.Image.Empty() {
			continue
		}
		img := matToImage(col.Image)
		var lines []ocr.OCRLine
		var err error
		if fixedLayout {
			lines, err = columnOCR.RecognizeColumn(img)
		} else {
			lines, err = p.OCR.Recognize(img)
		}
		if err != nil {
			return nil, err
		}
		for i := range lines {
			lines[i].Box.Min.X += col.X0
			lines[i].Box.Max.X += col.X0
			lines[i].Box.Min.Y += col.Rect.Min.Y
			lines[i].Box.Max.Y += col.Rect.Min.Y
		}
		all = append(all, lines...)
	}
	return all, nil
}

// matToImage converts a CV_8U (or CV_8UC3) gocv.Mat to a pure-Go BGR
// Image buffer. This is the only point where gocv memory crosses into the
// OCR stage.
func matToImage(m gocv.Mat) *ocr.Image {
	if m.Empty() {
		return nil
	}
	w, h := m.Cols(), m.Rows()
	bytes := m.ToBytes()
	if m.Channels() == 1 {
		return ocr.GrayToBGR(bytes, w, h)
	}
	return ocr.NewImage(bytes, w, h)
}

// filterByColumns drops OCR lines outside the detected table columns
// (title/footer junk) and lines overlapping grid lines.
func filterByColumns(lines []ocr.OCRLine, cols []vision.Column) []ocr.OCRLine {
	if len(cols) == 0 {
		return lines
	}
	minY, maxY := 1<<30, -1
	for _, c := range cols {
		if c.Rect.Min.Y < minY {
			minY = c.Rect.Min.Y
		}
		if c.Rect.Max.Y > maxY {
			maxY = c.Rect.Max.Y
		}
	}
	var out []ocr.OCRLine
	for _, l := range lines {
		cx := (l.Box.Min.X + l.Box.Max.X) / 2
		cy := (l.Box.Min.Y + l.Box.Max.Y) / 2
		if cy < minY-10 || cy > maxY+10 {
			continue
		}
		// must fall inside or near a column x-range
		inside := false
		for _, c := range cols {
			if cx >= c.Rect.Min.X-8 && cx <= c.Rect.Max.X+8 {
				inside = true
				break
			}
		}
		if inside {
			out = append(out, l)
		}
	}
	return out
}

func columnRects(cols []vision.Column) []layout.ColumnRect {
	out := make([]layout.ColumnRect, 0, len(cols))
	for _, c := range cols {
		out = append(out, layout.ColumnRect{Index: c.Index, Rect: c.Rect})
	}
	return out
}

func toLayoutLines(lines []ocr.OCRLine) []layout.Line {
	out := make([]layout.Line, 0, len(lines))
	for _, l := range lines {
		raw := strings.TrimSpace(l.Text)
		norm := normalize.NormalizeGreek(raw)
		out = append(out, layout.Line{
			Text:       norm,
			RawText:    raw,
			Confidence: l.Confidence,
			Box:        l.Box,
		})
	}
	return out
}

func rawOCRTxt(lines []ocr.OCRLine) string {
	sorted := make([]ocr.OCRLine, len(lines))
	copy(sorted, lines)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Box.Min.Y != sorted[j].Box.Min.Y {
			return sorted[i].Box.Min.Y < sorted[j].Box.Min.Y
		}
		return sorted[i].Box.Min.X < sorted[j].Box.Min.X
	})
	var b strings.Builder
	for _, l := range sorted {
		b.WriteString(fmt.Sprintf("y=%d x=%d c=%.3f %s\n", l.Box.Min.Y, l.Box.Min.X, l.Confidence, l.Text))
	}
	return b.String()
}

// CloseProcessed frees all Mats owned by a ProcessedSchedule.
func CloseProcessed(proc *vision.ProcessedSchedule) {
	proc.Corrected.Close()
	for i := range proc.Columns {
		proc.Columns[i].Image.Close()
	}
	for _, m := range proc.DebugImages {
		m.Close()
	}
	proc.Original.Close()
}

// writeDebug emits the debugging artifacts described in the project spec.
func writeDebug(proc *vision.ProcessedSchedule, lines []ocr.OCRLine, doc *layout.Document, dir string, log *log.Logger) {
	if err := os.MkdirAll(filepath.Join(dir, "columns"), 0o755); err != nil {
		log.Printf("debug: %v", err)
		return
	}
	if err := os.MkdirAll(filepath.Join(dir, "ocr"), 0o755); err != nil {
		log.Printf("debug: %v", err)
		return
	}

	save := func(name string, m gocv.Mat) {
		ok := gocv.IMWrite(filepath.Join(dir, name), m)
		if !ok {
			log.Printf("debug: failed to write %s", name)
		}
	}

	if !proc.Original.Empty() {
		save("original.jpg", proc.Original)
	}
	if !proc.Corrected.Empty() {
		save("deskewed.jpg", proc.Corrected)
	}
	for name, m := range proc.DebugImages {
		save(name, m)
	}
	for i, c := range proc.Columns {
		if !c.Image.Empty() {
			save(filepath.Join("columns", fmt.Sprintf("%02d.jpg", i+1)), c.Image)
		}
	}

	// annotated OCR image
	annotated := proc.Corrected.Clone()
	defer annotated.Close()
	ll := make([]layout.Line, 0, len(lines))
	for _, l := range lines {
		ll = append(ll, layout.Line{Text: l.Text, Box: l.Box})
	}
	layout.DrawBoxes(&annotated, ll)
	save("ocr/annotated.jpg", annotated)

	// per-column annotated images
	for _, col := range doc.Columns {
		if col.Rect.Dx() <= 0 || col.Rect.Dy() <= 0 {
			continue
		}
		crop := proc.Corrected.Region(col.Rect)
		var colLines []layout.Line
		for _, l := range ll {
			cx := (l.Box.Min.X + l.Box.Max.X) / 2
			if cx >= col.Rect.Min.X-8 && cx <= col.Rect.Max.X+8 {
				colLines = append(colLines, l)
			}
		}
		layout.DrawBoxes(&crop, colLines)
		save(filepath.Join("ocr", fmt.Sprintf("column-%02d-annotated.jpg", col.Index+1)), crop)
		crop.Close()
	}
}

// ValidateResult runs semantic validation over the parsed schedule and
// fills missing names/addresses from the reference catalog when the phone
// number identifies a known pharmacy.
func ValidateResult(sched *parse.Schedule, refs []validate.Reference) ([]PharmacyValidation, []string) {
	v := validate.NewValidator(refs)
	var out []PharmacyValidation
	var warnings []string
	for i := range sched.Days {
		for j := range sched.Days[i].Shifts {
			shift := &sched.Days[i].Shifts[j]
			if !validate.ValidateTime(shift.From) || !validate.ValidateTime(shift.To) {
				warnings = append(warnings, fmt.Sprintf("invalid shift times %s-%s on %s", shift.From, shift.To, sched.Days[i].Date.Format("02/01/2006")))
			}
			for k := range shift.Pharmacies {
				ph := &shift.Pharmacies[k]
				res := v.ValidatePharmacy(ph.Name, ph.Address, ph.Phone)
				out = append(out, PharmacyValidation{
					Phone:      ph.Phone,
					Name:       ph.Name,
					Validation: res,
				})
				if res.Discrepancy != "" {
					warnings = append(warnings, res.Discrepancy)
				}
				for _, w := range res.Warnings {
					warnings = append(warnings, w)
				}
			}
		}
	}
	return out, warnings
}

// nameSimilarity compares the OCR name against the catalog name using the
// normalized (Greeklish-transliterated) forms.
func nameSimilarity(ocr, catalog string) float64 {
	a := normalize.GreekToLatin(ocr)
	b := normalize.GreekToLatin(catalog)
	s := normalize.Similarity(a, b)
	if s2 := normalize.Similarity(ocr, catalog); s2 > s {
		s = s2
	}
	return s
}

// FillFromCatalog completes missing name/address fields using the reference
// catalog, keyed by the normalized phone number, and attaches the catalog's
// Latin name/address and OpenStreetMap coordinates to every matched pharmacy.
func FillFromCatalog(sched *parse.Schedule, refs []validate.Reference) int {
	byPhone := map[string][]validate.Reference{}
	for _, r := range refs {
		for _, d := range normalize.Phones(r.Phone) {
			byPhone[d] = append(byPhone[d], r)
		}
	}
	filled := 0
	for i := range sched.Days {
		for j := range sched.Days[i].Shifts {
			for k := range sched.Days[i].Shifts[j].Pharmacies {
				ph := &sched.Days[i].Shifts[j].Pharmacies[k]
				candidates := byPhone[normalize.DigitsOnly(ph.Phone)]
				if len(candidates) == 0 {
					continue
				}
				ref := chooseCatalogReference(ph, candidates)
				ph.NameLat = ref.NameLat
				ph.AddressLat = ref.AddressLat
				ph.Lat = ref.Lat
				ph.Lon = ref.Lon
				catName := strings.ToUpper(normalize.NormalizeGreek(ref.Name))
				catAddr := strings.ToUpper(normalize.NormalizeGreek(ref.Address))
				if ph.Name != catName || ph.Address != catAddr {
					ph.Name = catName
					ph.Address = catAddr
					filled++
				}
			}
		}
	}
	return filled
}

func chooseCatalogReference(ph *parse.Pharmacy, refs []validate.Reference) validate.Reference {
	if len(refs) == 1 {
		return refs[0]
	}
	best := refs[0]
	bestScore := -1.0
	for _, ref := range refs {
		score := nameSimilarity(ph.Name, strings.ToUpper(normalize.NormalizeGreek(ref.Name)))
		if addressScore := normalize.Similarity(
			normalize.GreekToLatin(ph.Address),
			normalize.GreekToLatin(ref.Address),
		); addressScore > score {
			score = addressScore
		}
		if score > bestScore || (score == bestScore && ref.Name < best.Name) {
			best, bestScore = ref, score
		}
	}
	return best
}

// dedupePharmacies removes duplicate entries within the same shift using
// normalized name+address+phone.
func dedupePharmacies(shift *parse.Shift) {
	seen := map[string]bool{}
	var out []parse.Pharmacy
	for _, ph := range shift.Pharmacies {
		key := normalize.NormalizeGreek(strings.ToUpper(ph.Name)) + "|" + normalize.NormalizeGreek(strings.ToUpper(ph.Address)) + "|" + ph.Phone
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, ph)
	}
	shift.Pharmacies = out
}

// ApplyValidation merges validation results back into the schedule (score
// per pharmacy) and deduplicates.
func ApplyValidation(sched *parse.Schedule, vals []PharmacyValidation) {
	byPhone := map[string]float32{}
	for _, v := range vals {
		byPhone[v.Phone] = v.Validation.Score
	}
	for i := range sched.Days {
		for j := range sched.Days[i].Shifts {
			shift := &sched.Days[i].Shifts[j]
			for k := range shift.Pharmacies {
				ph := &shift.Pharmacies[k]
				if s, ok := byPhone[ph.Phone]; ok {
					ph.Confidence = s
				} else {
					ph.Confidence = 0.6
				}
			}
			dedupePharmacies(shift)
		}
	}
}

// JSON marshals the schedule deterministically (sorted keys, stable
// ordering) — Go's encoding/json already sorts map keys and field order is
// struct order, so output is deterministic for equal inputs.
func (r *Result) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
